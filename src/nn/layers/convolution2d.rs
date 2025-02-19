use rand_distr::Uniform;

use crate::nn::learning_rate::LearningRate;

use super::{max_pooling::MaxPooling, Matrix, Propagates};

use crate::prettify::*;

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
pub struct Convolution2dDeprecated {
    pub filters: usize,
    pub kernels: Matrix,        // Same as weights
    pub biases: Matrix,
    pub k_d: Dimensions,
    pub i_d: Dimensions,
    //pub stride: usize,        // Assumes stride of 1 for now.
}

impl Convolution2dDeprecated {
    /// Set channels to 1 for greyscale, 3 for RGB.
    /// * RGB support not implemented yet, so ensure channels is 1.
    /// * Might consider allowing for more flexibility here, but have to think carefuly about the cleanest way to do this.
    pub fn new(filters: usize, channels: usize, k_d: Dimensions, i_d: Dimensions) -> Self {
        let biases = vec![0.; filters];

        let fanin = (k_d.height * k_d.width * channels) as f32;
        let term = (6. / fanin).sqrt();
        //let normal = Normal::new(0., normal_term).unwrap();
        let uniform = Uniform::new_inclusive(-term, term);
        
        Convolution2dDeprecated {
            filters,
            //kernels: Matrix::new_randomized_normal(filters, channels * k_d.height * k_d.width, normal),
            kernels: Matrix::new_randomized_uniform(filters, channels * k_d.height * k_d.width, uniform),
            biases: Matrix::from(1, filters, biases),
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

    pub fn influences_maxpool(&self, p_d: Dimensions, stride: usize) -> MaxPooling {
        MaxPooling::new(
            self.filters,
            p_d,
            Dimensions { height: self.i_d.height - self.k_d.height + 1, width: self.i_d.width - self.k_d.width + 1 },
            stride)
    }
}

impl Propagates for Convolution2dDeprecated {
    fn forward(&mut self, inputs: &Matrix) -> Matrix {
        let r = inputs 
            .valid_cross_correlation(&self.kernels, &self.k_d, &self.i_d);

            // let msg = format!("F -> {:?} x {:?}", r.row_count(), r.column_count()).bright_red();
            // println!("{msg}");
        r
    }

    fn backward<'a>(&'a mut self, learning_rate: &mut LearningRate, dvalues: &Matrix, inputs: &Matrix) -> Matrix {
        println!("{BRIGHT_GREEN} dvalues shape: {:?} x {:?}{RESET}", dvalues.row_count(), dvalues.column_count());
        let dims = self.backward_dims();
        let size = dims.height * dims.width;
        let k_size = self.k_d.height * self.k_d.width;

        for filter in 0..self.filters {
            let filter_offset = filter * size;

            let mut gradient = Matrix::from(1, k_size, vec![0.; k_size]);
            for image in 0..dvalues.row_count() {
                let delta_image = dvalues.row(image);
                let d_i = Matrix::from(1, self.i_d.height * self.i_d.width, delta_image.to_vec());

                let delta_filter = &delta_image[filter_offset..filter_offset + size];
                let d_f = Matrix::from(1, size, delta_filter.to_vec());

                let grad = d_i.valid_cross_correlation(&d_f, &dims, &self.i_d);

                gradient = gradient.add(&grad);
            }

            gradient = gradient.scale(1. / dvalues.row_count() as f32);
            for i in 0..k_size {
                self.kernels.add_at(filter * k_size + i, -learning_rate.rate() * gradient.read_at(i));
            }
        }

        inputs.full_outer_convolution(&self.kernels, &self.k_d, &self.i_d)
    }
}

#[cfg(test)]

mod tests {
    use crate::prettify::*;
    use crate::{geoalg::f32_math::matrix::Matrix, nn::layers::Propagates};

    use super::*;

    #[test]
    fn test_forward_and_back() {
        let batch_size = 2;
        let filters = 3;
        let s3x3 = 9;
        let s4x4 = 16;

        // Test two different images
        let inputs = Matrix::from(batch_size, s4x4, vec![
            1., 2., 3., 4., 
            5., 6., 7., 8.,
            9., 10., 11., 12.,
            13., 14., 15., 16.,

            10., 20., 30., 40.,
            50., 60., 70., 80.,
            90., 100., 110., 120.,
            130., 140., 150., 160.
        ]);

        let mut cv2d = Convolution2dDeprecated::new(
            filters,
            1, 
            Dimensions { width: 3, height: 3 } , 
            Dimensions { width: 4, height: 4 });
        cv2d.kernels = Matrix::from(filters, s3x3, vec![
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

        let dvalues = &Matrix::from(batch_size, filters * s3x3, vec![0.; batch_size * filters * s3x3]);
        let _dvalues = cv2d.backward(&mut learning_rate, &dvalues, &inputs);
    }

    #[test]
    fn test_influences_maxpool() {
        let batch_size = 2;
        let filters = 3;
        let s3x3 = 9;
        let s13x13 = 169;
        let s28x28 = 784;

        // Test two different images
        let inputs = Matrix::from(batch_size, s28x28, vec![1.0; batch_size * s28x28]);

        let mut cv2d = Convolution2dDeprecated::new(
            filters,
            1, 
            Dimensions { width: 3, height: 3 } , 
            Dimensions { width: 28, height: 28 });
        cv2d.kernels = Matrix::from(filters, s3x3, vec![1.2; filters * s3x3]);

        let output = &cv2d.forward(&inputs);
        println!("{BRIGHT_CYAN}{:?}{RESET}", output);

        let mut maxpool = cv2d.influences_maxpool(Dimensions { width: 2, height: 2 }, 2);
        let _output2 = maxpool.forward(output);

        let dvalues = &Matrix::from(batch_size, filters * s13x13, vec![-0.1; batch_size * filters * s13x13]);

        let mut learning_rate = LearningRate::new(0.01);
        let dvalues = &maxpool.backward(&mut learning_rate, dvalues, output);

        let _dvalues = cv2d.backward(&mut learning_rate, dvalues, &inputs);
      }
}