use rand_distr::Uniform;

use crate::{geoalg::f32_math::{shape::Shape, tensor::Tensor}, nn::learning_rate::LearningRate};

use super::{max_pooling::MaxPooling, LayerPropagates};

//use crate::prettify::*;

pub struct Dimensions {
    pub height: usize,
    pub width: usize
}

impl Dimensions {
    pub fn shape(&self) -> (usize, usize) {
        (self.height, self.width)
    }
}

///
pub struct Convolution2d {
    pub filters: usize,
    pub kernels: Tensor,        // Same as weights
    pub biases: Tensor,
    pub k_d: Dimensions,
    pub i_d: Dimensions,
    //pub stride: usize,        // Assumes stride of 1 for now.
}

impl Convolution2d {
    /// Set channels to 1 for greyscale, 3 for RGB.
    /// * RGB support not implemented yet, so ensure channels is 1.
    /// * Might consider allowing for more flexibility here, but have to think carefuly about the cleanest way to do this.
    pub fn new(filters: usize, channels: usize, k_d: Dimensions, i_d: Dimensions) -> Self {
        let biases = vec![0.; filters];

        let fanin = (k_d.height * k_d.width * channels) as f32;
        let term = (6. / fanin).sqrt();
        //let normal = Normal::new(0., normal_term).unwrap();
        let uniform = Uniform::new_inclusive(-term, term);
        
        Convolution2d {
            filters,
            //kernels: Matrix::new_randomized_normal(filters, channels * k_d.height * k_d.width, normal),
            kernels: Tensor::new_randomized_uniform(Shape::d4(filters, k_d.height, k_d.width, channels), uniform),
            biases: Tensor::vector(biases),
            k_d,
            i_d
        }
    }

    pub fn backward_dims(&self) -> Dimensions {
        Dimensions {
            height: self.i_d.height - self.k_d.height + 1,
            width: self.i_d.width - self.k_d.width + 1
        }
    }

    pub fn feed_into_maxpool(&self, p_d: Dimensions, stride: usize) -> MaxPooling {
        MaxPooling::new(
            self.filters,
            p_d,
            Dimensions { height: self.i_d.height - self.k_d.height + 1, width: self.i_d.width - self.k_d.width + 1 },
            stride)
    }
}

impl LayerPropagates for Convolution2d {
    fn forward(&mut self, inputs: &Tensor) -> Tensor {
        let r = inputs 
            .batch_valid_cross_correlation_simd(&self.kernels);//, &self.k_d, &self.i_d);

            // let msg = format!("F -> {:?} x {:?}", r.row_count(), r.column_count()).bright_red();
            // println!("{msg}");
        r
    }

    #[allow(unused_variables)]
    fn backward<'a>(&'a mut self, learning_rate: &mut LearningRate, dvalues: &Tensor, inputs: &Tensor) -> Tensor {

        let dims = self.backward_dims();
        let size = dims.height * dims.width;
        let k_size = self.k_d.height * self.k_d.width;

        for filter in 0..self.filters {
            let filter_offset = filter * size;

            let mut gradient = Tensor::matrix(1, k_size, vec![0.; k_size]);
            for image in 0..dvalues.shape[0] {
                let delta_image = dvalues.dim_slice(0, image);
                let d_i = Tensor::new(Shape::d4(1, self.i_d.height, self.i_d.width, 1), delta_image.to_vec());

                let delta_filter = &delta_image[filter_offset..filter_offset + size];
                let d_f = Tensor::new(Shape::d4(1, dims.height, dims.width, 1), delta_filter.to_vec());

                let grad = d_i.batch_valid_cross_correlation_simd(&d_f);//, &dims, &self.i_d);

                gradient = gradient.add_element_wise_simd(&grad);
            }

            gradient = gradient.scale_simd(1. / dvalues.shape[0] as f32);
            for i in 0..k_size {
                self.kernels[filter * k_size + i] -= learning_rate.rate() * gradient[i];
            }
        }

        Tensor::new(Shape::d4(1,1,1,1), vec![0.; 1])
        //todo!()
        //inputs.full_outer_convolution(&self.kernels, &self.k_d, &self.i_d)
    }
}

#[cfg(test)]

mod tests {
    use crate::prettify::*;
    use crate::nn::layers::LayerPropagates;

    use super::*;

    #[test]
    fn test_forward_and_back() {
        let batch_size = 2;
        let filters = 3;
        let s3x3 = 9;

        // Test two different images
        let inputs = Tensor::new(Shape::d4(batch_size, 4, 4, 1), vec![

            1., 2., 3., 4., 
            5., 6., 7., 8.,
            9., 10., 11., 12.,
            13., 14., 15., 16.,

            10., 20., 30., 40.,
            50., 60., 70., 80.,
            90., 100., 110., 120.,
            130., 140., 150., 160.
        ]);

        // Have two filters
        let mut cv2d = Convolution2d::new(
            3,
            1, 
            Dimensions { width: 3, height: 3 } , 
            Dimensions { width: 4, height: 4 });

        cv2d.kernels = Tensor::new(Shape::d4(filters, 3, 3, 1), vec![

            0., 0.15, 0.,
            0.15, 0.4, 0.15,
            0., 0.15, 0., 

            0., 0.10, 0.,
            0.10, 0.6, 0.10,
            0., 0.10, 0.,

            0., 0., 0.,
            0., 2., 0.,
            0., 0., 0.
        ]);

        let output = cv2d.forward(&inputs);
        println!("{BRIGHT_CYAN}{:?}{RESET}", output);

        let mut learning_rate = LearningRate::new(0.01);

        let dvalues = &Tensor::new(Shape::d4(batch_size, filters, 3, 3), vec![0.; batch_size * filters * s3x3]);
        
        println!("{:?}", dvalues.shape);
        
        let _dvalues = cv2d.backward(&mut learning_rate, dvalues, &inputs);
    }

    #[test]
    fn test_influences_maxpool() {
        let batch_size = 2;
        let filters = 3;
        let s3x3 = 9;
        let s13x13 = 169;
        let s28x28 = 784;

        // Test two different images
        let inputs = Tensor::new(Shape::d4(batch_size, 28, 28, 1), vec![1.0; batch_size * s28x28]);

        let mut cv2d = Convolution2d::new(
            filters,
            1, 
            Dimensions { width: 3, height: 3 } , 
            Dimensions { width: 28, height: 28 });
        cv2d.kernels = Tensor::new(Shape::d4(filters, 3,3, 1), vec![1.2; filters * s3x3]);

        let output = &cv2d.forward(&inputs);
        println!("{BRIGHT_CYAN}{:?}{RESET}", output);

        let mut maxpool = cv2d.feed_into_maxpool(Dimensions { width: 2, height: 2 }, 2);
        let _output2 = maxpool.forward(output);

        let dvalues = &Tensor::new(Shape::new(vec![batch_size, filters, 13, 13, 1] ), vec![-0.1; batch_size * filters * s13x13]);

        let mut learning_rate = LearningRate::new(0.01);
        let dvalues = &maxpool.backward(&mut learning_rate, dvalues, output);

        let _dvalues = cv2d.backward(&mut learning_rate, dvalues, &inputs);
      }
}