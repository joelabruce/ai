use rand_distr::Normal;

use super::{max_pooling::MaxPooling, Matrix, Propagates};

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
    pub kernels: Matrix,        // Same as weights
    pub biases: Matrix,
    pub k_d: Dimensions,
    pub i_d: Dimensions,
    //pub stride: usize,        // Assumes stride of 1 for now.
}

impl Convolution2d {
    /// Set channels to 1 for greyscale, 3 for RGB.
    /// *RGB support not implemented yet, so ensure channels is 1.
    pub fn new(filters: usize, channels: usize, k_d: Dimensions, i_d: Dimensions) -> Self {
        let biases = vec![0.; filters];

        let fanin = (k_d.height * k_d.width * channels) as f32;
        let normal_term = (2. / fanin).sqrt();
        let normal = Normal::new(0., normal_term).unwrap();
        
        Convolution2d {
            filters,
            kernels: Matrix::new_randomized_normal(filters, channels * k_d.height * k_d.width, normal),
            biases: Matrix::from(1, filters, biases),
            k_d,
            i_d
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

impl Propagates for Convolution2d {
    fn forward(&mut self, inputs: &Matrix) -> Matrix {
        let r = inputs 
            .valid_cross_correlation(&self.kernels, &self.k_d, &self.i_d);

        
        //println!()
            //.add_row_partitioned(&self.biases);
        r
    }

    fn backward<'a>(&'a mut self, dvalues: &Matrix, inputs: &Matrix) -> Matrix {
        dvalues.len();
        inputs.len();
        //todo!()

        let capacity = self.kernels.row_count() * self.i_d.height * self.i_d.width;

        Matrix::from(inputs.row_count(), self.i_d.height * self.i_d.width, vec![0.; capacity])
    }
}

#[cfg(test)]

mod tests {
    use colored::Colorize;

    use crate::{geoalg::f32_math::matrix::Matrix, nn::layers::Propagates};

    use super::*;

    #[test]
    fn test_forward() {
        // Test two different images
        let inputs = Matrix::from(2, 4 * 4, vec![
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
        cv2d.kernels = Matrix::from(3, 3 * 3, vec![
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
        let conv_result = format!("{:?}", output).bright_cyan();
        println!("{conv_result}");
    }
}