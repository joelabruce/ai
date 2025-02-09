use rand::prelude::*;
use rand_distr::Normal;
//use rand::distributions::;


use crate::geoalg::f32_math::{simd_extensions::dot_product_simd3, tensor::Tensor};

use super::{Matrix, Propagates};

///
pub struct Convolution2dLayer {
    pub kernels: Matrix,
    pub kernel_width: usize,
    pub kernel_height: usize,
    pub biases: Matrix,
    pub image_width: usize,
    pub image_height: usize,
    //pub image_height: usize   // inferred for now
    //pub stride: usize,        // Assumes stride of 1 for now.
}

impl Convolution2dLayer {
    /// Set channels to 1 for greyscale, 3 for RGB.
    /// *RGB support not implemented yet, so ensure channels is 1.
    pub fn new(filters: usize, channels: usize, kernel_width: usize, kernel_height: usize, image_width: usize, image_height: usize) -> Self {
        let biases = vec![0.; filters];

        let fanin = (kernel_height * kernel_width * channels) as f32;
        let normal_term = (2. / fanin).sqrt();
        let normal = Normal::new(0., normal_term).unwrap();
        
        Convolution2dLayer {
            kernels: Matrix::new_randomized_normal(filters, channels * kernel_height * kernel_width, normal),
            kernel_width,
            kernel_height,
            biases: Matrix::from(1, filters, biases),
            image_width,
            image_height
        }
    }
}

impl Propagates for Convolution2dLayer {
    fn forward(&mut self, inputs: &Matrix) -> Matrix {
        let batches = inputs.row_count();

        // Adjust for valid convolution (no padding)
        let n_rows = self.image_height - self.kernel_height + 1;

        // Columns are calculated based on remaining items
        // Might consider having it as parameter when creating the layer.
        let n_columns = self.image_width - self.kernel_width + 1;

        let mut values = Vec::with_capacity(self.kernels.len());
        for batch_index in 0..batches {
            let input = inputs.row(batch_index);
            for filter_index in 0..self.kernels.row_count() {
                let filter = self.kernels.row(filter_index);

                for row in 0..n_rows {
                    for column in 0..n_columns {
                        let mut c_accum = 0.;

                        // Slide kernel window horizontally and then vertically
                        // Since we are doing optimized dot_products, only need to move down the rows
                        for kernel_row in 0..self.kernel_height {
                            // Get the row of the input, offset by kernel row, and start the row at the column.
                            let input_row_start_index = (row + kernel_row) * self.image_width + column;
                            // Only get as many columns as are in the kernel for the convolution.
                            let input_row_end_index = input_row_start_index + self.kernel_width;

                            let kernel_row_start_index = kernel_row * self.kernel_width;
                            let kernel_row_end_index = kernel_row_start_index + self.kernel_width;

                            let x = &input[input_row_start_index..input_row_end_index];
                            let y = &filter[kernel_row_start_index..kernel_row_end_index];

                            c_accum += dot_product_simd3(x, y)
                        }

                        values.push(c_accum);
                    }
                }
            }
        }

        Matrix::from(batches, n_rows * n_columns, values)
    }

    fn backward<'a>(&'a mut self, dvalues: &Matrix, inputs: &Matrix) -> Matrix {
        dvalues.len();
        inputs.len();
        todo!()
    }
}