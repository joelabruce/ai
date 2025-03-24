use rand_distr::Uniform;

use crate::nn::learning_rate::LearningRate;

use super::{max_pooling::MaxPooling, Matrix, Propagates};

// Useful for debugging
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
            biases: Matrix::new(1, filters, biases),
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
        let original = false;
        if original {
        // Known to work
            let r = inputs.valid_cross_correlation(&self.kernels, &self.k_d, &self.i_d);
            r
        } else {
        // Seems to be working, and is much faster!
           let r = inputs.par_cc_im2col(&self.kernels, &self.k_d, &self.i_d);
           r
        }
    }

    fn backward<'a>(&'a mut self, learning_rate: &mut LearningRate, dvalues: &Matrix, inputs: &Matrix) -> Matrix {
        //println!("{BRIGHT_GREEN} dvalues shape: {:?} x {:?}{RESET}", dvalues.row_count(), dvalues.column_count());
        let dims = self.backward_dims();
        let f_size = dims.height * dims.width;
        let k_size = self.k_d.height * self.k_d.width;
        let i_size = self.i_d.height * self.i_d.width;

        for filter in 0..self.filters {
            let filter_offset = filter * f_size;

            let mut gradient = Matrix::new(1, k_size, vec![0.; k_size]);
            for image in 0..dvalues.row_count() {
                let delta_image = dvalues.row(image);
                let d_i = Matrix::new(1, i_size, delta_image.to_vec());

                let delta_filter = &delta_image[filter_offset..filter_offset + f_size];
                let d_f = Matrix::new(1, f_size, delta_filter.to_vec());

                let grad = d_i.cc_im2col(&d_f, &dims, &self.i_d);

                gradient = gradient.add(&grad);
            }

            gradient = gradient.scale(learning_rate.rate() / dvalues.row_count() as f32);
            for i in 0..k_size {
                self.kernels[filter * k_size + i] -= gradient[i];
            }
        }

        inputs.full_outer_convolution(&self.kernels, &self.k_d, &self.i_d)
    }
}

#[cfg(test)]

mod tests {
    //use crate::prettify::*;
    use crate::{geoalg::f32_math::{matrix::Matrix, simd_extensions::{im2col_transposed, SliceExt}}, nn::layers::Propagates};

    use super::*;

    #[test]
    fn test_forward_and_back() {
        let batch_size = 2;
        let filters = 3;
        let s3x3 = 9;
        let s4x4 = 16;

        // Test two different images
        let inputs = Matrix::new(batch_size, s4x4, vec![
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
        cv2d.kernels = Matrix::new(filters, s3x3, vec![
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

        let _output = cv2d.forward(&inputs);
        //println!("{BRIGHT_CYAN}{:?}{RESET}", output);

        let mut learning_rate = LearningRate::new(0.01);

        let dvalues = &Matrix::new(batch_size, filters * s3x3, vec![0.; batch_size * filters * s3x3]);
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
        let inputs = Matrix::new(batch_size, s28x28, vec![1.0; batch_size * s28x28]);

        let mut cv2d = Convolution2dDeprecated::new(
            filters,
            1, 
            Dimensions { width: 3, height: 3 } , 
            Dimensions { width: 28, height: 28 });
        cv2d.kernels = Matrix::new(filters, s3x3, vec![1.2; filters * s3x3]);

        let output = &cv2d.forward(&inputs);
        //println!("{BRIGHT_CYAN}{:?}{RESET}", output);

        let mut maxpool = cv2d.influences_maxpool(Dimensions { width: 2, height: 2 }, 2);
        let _output2 = maxpool.forward(output);

        let dvalues = &Matrix::new(batch_size, filters * s13x13, vec![-0.1; batch_size * filters * s13x13]);

        let mut learning_rate = LearningRate::new(0.01);
        let dvalues = &maxpool.backward(&mut learning_rate, dvalues, output);

        let _dvalues = cv2d.backward(&mut learning_rate, dvalues, &inputs);
    }

    #[test]
    #[allow(unused)]
    fn test_matches_other_libs() {
        // 2, 1, 6, 6
        let batch_size = 2;
        let in_channels = 1;
        let image_height = 6;
        let image_width = 6;
        let images = vec![
            0.5463, 0.7340, 0.0529, 0.3832, 0.0270, 0.0762,
            0.8616, 0.4619, 0.0695, 0.2323, 0.3086, 0.1032,
            0.4636, 0.2489, 0.1030, 0.5925, 0.8976, 0.3730,
            0.1945, 0.5421, 0.6404, 0.4513, 0.6649, 0.8863,
            0.7978, 0.1297, 0.7143, 0.4763, 0.7851, 0.6691,
            0.2878, 0.9141, 0.4255, 0.4009, 0.6997, 0.7527,  
  
            0.6216, 0.5607, 0.9803, 0.7456, 0.4444, 0.6561,
            0.1483, 0.4778, 0.0693, 0.7137, 0.3872, 0.3660,
            0.8805, 0.1232, 0.2228, 0.4856, 0.0156, 0.6362,
            0.8774, 0.1551, 0.5922, 0.4736, 0.5383, 0.8704,
            0.8192, 0.9175, 0.8558, 0.5901, 0.9186, 0.8109,
            0.0681, 0.9770, 0.4072, 0.3955, 0.9965, 0.8976
        ];

        let kernel_height = 3;
        let kernel_width = 3;

        let im2col_result = im2col_transposed(&*images, 2, image_height, image_width, kernel_height, kernel_width);

        // 2, 9, 16
        let expected = vec![
            0.5463, 0.7340, 0.0529, 0.3832, 0.8616, 0.4619, 0.0695, 0.2323, 0.4636, 0.2489, 0.1030, 0.5925, 0.1945, 0.5421, 0.6404, 0.4513,
            0.7340, 0.0529, 0.3832, 0.0270, 0.4619, 0.0695, 0.2323, 0.3086, 0.2489, 0.1030, 0.5925, 0.8976, 0.5421, 0.6404, 0.4513, 0.6649,
            0.0529, 0.3832, 0.0270, 0.0762, 0.0695, 0.2323, 0.3086, 0.1032, 0.1030, 0.5925, 0.8976, 0.3730, 0.6404, 0.4513, 0.6649, 0.8863,
            0.8616, 0.4619, 0.0695, 0.2323, 0.4636, 0.2489, 0.1030, 0.5925, 0.1945, 0.5421, 0.6404, 0.4513, 0.7978, 0.1297, 0.7143, 0.4763,
            0.4619, 0.0695, 0.2323, 0.3086, 0.2489, 0.1030, 0.5925, 0.8976, 0.5421, 0.6404, 0.4513, 0.6649, 0.1297, 0.7143, 0.4763, 0.7851,
            0.0695, 0.2323, 0.3086, 0.1032, 0.1030, 0.5925, 0.8976, 0.3730, 0.6404, 0.4513, 0.6649, 0.8863, 0.7143, 0.4763, 0.7851, 0.6691,
            0.4636, 0.2489, 0.1030, 0.5925, 0.1945, 0.5421, 0.6404, 0.4513, 0.7978, 0.1297, 0.7143, 0.4763, 0.2878, 0.9141, 0.4255, 0.4009,
            0.2489, 0.1030, 0.5925, 0.8976, 0.5421, 0.6404, 0.4513, 0.6649, 0.1297, 0.7143, 0.4763, 0.7851, 0.9141, 0.4255, 0.4009, 0.6997,
            0.1030, 0.5925, 0.8976, 0.3730, 0.6404, 0.4513, 0.6649, 0.8863, 0.7143, 0.4763, 0.7851, 0.6691, 0.4255, 0.4009, 0.6997, 0.7527,
  
            0.6216, 0.5607, 0.9803, 0.7456, 0.1483, 0.4778, 0.0693, 0.7137, 0.8805, 0.1232, 0.2228, 0.4856, 0.8774, 0.1551, 0.5922, 0.4736,
            0.5607, 0.9803, 0.7456, 0.4444, 0.4778, 0.0693, 0.7137, 0.3872, 0.1232, 0.2228, 0.4856, 0.0156, 0.1551, 0.5922, 0.4736, 0.5383,
            0.9803, 0.7456, 0.4444, 0.6561, 0.0693, 0.7137, 0.3872, 0.3660, 0.2228, 0.4856, 0.0156, 0.6362, 0.5922, 0.4736, 0.5383, 0.8704,
            0.1483, 0.4778, 0.0693, 0.7137, 0.8805, 0.1232, 0.2228, 0.4856, 0.8774, 0.1551, 0.5922, 0.4736, 0.8192, 0.9175, 0.8558, 0.5901,
            0.4778, 0.0693, 0.7137, 0.3872, 0.1232, 0.2228, 0.4856, 0.0156, 0.1551, 0.5922, 0.4736, 0.5383, 0.9175, 0.8558, 0.5901, 0.9186,
            0.0693, 0.7137, 0.3872, 0.3660, 0.2228, 0.4856, 0.0156, 0.6362, 0.5922, 0.4736, 0.5383, 0.8704, 0.8558, 0.5901, 0.9186, 0.8109,
            0.8805, 0.1232, 0.2228, 0.4856, 0.8774, 0.1551, 0.5922, 0.4736, 0.8192, 0.9175, 0.8558, 0.5901, 0.0681, 0.9770, 0.4072, 0.3955,
            0.1232, 0.2228, 0.4856, 0.0156, 0.1551, 0.5922, 0.4736, 0.5383, 0.9175, 0.8558, 0.5901, 0.9186, 0.9770, 0.4072, 0.3955, 0.9965,
            0.2228, 0.4856, 0.0156, 0.6362, 0.5922, 0.4736, 0.5383, 0.8704, 0.8558, 0.5901, 0.9186, 0.8109, 0.4072, 0.3955, 0.9965, 0.8976
        ];

        println!("im2col = {:?}", im2col_result);

        //assert_eq!(im2col_result, expected);

        // 4, 9
        let out_channels = 4;
        let kernel_matrix = vec![
             0.0903,  1.6831, -1.2517,  1.4519, -0.4311, -1.9387, -0.5444,  0.6553, -0.7051,
             0.2976,  0.4306,  2.2662,  1.4804, -0.6277, -0.7304,  0.8510,  1.0613, -0.4742,
             0.3410,  2.0507, -1.1369, -0.2150,  0.6438,  1.4654, -0.1203, -0.1643,  0.8366,
             0.3783, -1.2285, -0.2971,  1.1674,  0.0966,  1.3630, -0.2323,  1.2406,  2.0313
        ];

        let actual1 = (&*kernel_matrix).mm_transpose(&*&im2col_result, out_channels, kernel_height * kernel_width, 9);
        println!("kernels x im2col = {:?}", actual1);

        // 2, 4, 16
        let expected_with_im2col_mult = vec![
             1.9738, -0.6198, -0.2823, -0.0087,  0.9321, -1.1576, -2.3568, -0.2736, -1.7140, -0.8481, -1.3011, -0.4691, -0.0118, -0.9827, -1.1674, -1.1842,
             2.1431,  1.6201,  0.2644,  1.6531,  1.5048,  1.4917,  0.6536,  1.1467,  0.4364,  2.1749,  3.2402,  1.9328,  3.3337,  1.8950,  2.5326,  2.8796,
             1.8347,  0.6577,  2.0010,  0.4927,  1.7967,  1.1241,  2.2288,  2.1696,  2.2772,  0.8449,  1.8495,  3.6219,  1.5799,  2.2705,  2.1596,  2.3598,
             0.8447,  2.2349,  2.5997,  2.2643,  2.3717,  2.7137,  2.8119,  3.4853,  2.4180,  2.9250,  2.7615,  3.0323,  3.0663,  1.2839,  3.2601,  2.9384,
  
            -0.9080, -0.2159,  0.0155, -0.5491,  0.7306, -1.6226,  0.4152, -0.7958, -0.3812, -1.5248, -0.2970, -2.2492, -0.9501, -0.3069, -1.2811, -1.4474,
             3.2915,  2.5324,  1.6893,  2.5744,  2.1009,  2.0130,  1.9705,  2.0148,  2.8543,  2.1543,  1.4158,  2.4130,  2.5834,  2.8407,  2.1194,  3.0185,
             0.6847,  2.6963,  2.2766,  1.5230,  1.5326,  0.6025,  1.6358,  2.0419,  1.5451,  1.2315,  2.5892,  1.4510,  1.7844,  2.0935,  2.8228,  2.4682,
            -0.0302,  1.5578,  0.5827,  2.1095,  1.9835,  2.3715,  0.9061,  3.4470,  4.6482,  2.5595,  3.3530,  4.4166,  4.2003,  2.2301,  4.2105,  4.1097
        ];

        println!("Expected = {:?}", expected_with_im2col_mult);
        //println!("{:?}", im2col_result);
    }
}