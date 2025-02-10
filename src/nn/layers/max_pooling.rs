use crate::geoalg::f32_math::matrix::Matrix;
use super::{dense::Dense, Propagates};

/// Currently only supports valid pooling layers, with no padding.
pub struct MaxPooling {
    filters: usize,
    pooling_height: usize,
    pooling_width: usize,
    stride: usize,
    input_width: usize,
    input_height: usize,

    max_indices: Vec<usize>
}

impl MaxPooling {
    pub fn new(filters: usize, pooling_height: usize, pooling_width: usize, input_width: usize, input_height: usize, stride: usize) -> Self {
        MaxPooling {
            filters, pooling_width, pooling_height, input_height, input_width, stride, max_indices: vec![0]
        }
    }

    pub fn influences_dense(&self) -> Dense {
        let (rows, columns) = self.output_shape();

        let features = self.filters * rows * columns;
        //Dense::new(self.input_size,features)
        Dense::new(features, 100)
    }

    fn output_shape(&self) -> (usize, usize) {
        (
            (self.input_height - self.pooling_height) / self.stride + 1,
            (self.input_width - self.pooling_width) / self.stride + 1
        )
    }
}

impl Propagates for MaxPooling {
    fn forward(&mut self, inputs: &Matrix) -> Matrix {
        let batches = inputs.row_count();

        //assert_eq!(batches, self.input_size, "Input size does not equal expected rows count passed into MaxPooling.");

        let (rows, columns) = self.output_shape();
        let rows_x_columns = rows * columns;

        let mut values = Vec::with_capacity(batches * self.filters * rows_x_columns);        
        self.max_indices = Vec::with_capacity(batches * self.filters * rows_x_columns);

        for batch in 0..batches {
            let input_row = inputs.row(batch);
            for filter in 0..self.filters {
                let filter_offset = filter * self.input_height * self.input_width;
                for row in 0..rows {
                    let w_row = row * self.stride;
                    for column in 0..columns {
                        let w_column = column * self.stride;

                        let mut max = f32::MIN;
                        let mut max_index = 0;
                        for k_row in 0..self.pooling_height {
                            for k_column in 0..self.pooling_width {
                                let index = filter_offset + (w_column + k_column) + (w_row + k_row) * self.input_width;
                                let index_value = input_row[index];
                                if index_value > max { 
                                    max = index_value;
                                    max_index = index;
                                } 
                            }
                        }

                        values.push(max);
                        self.max_indices.push(max_index);
                    }
                }
            }
        }

        Matrix::from(batches, self.filters * rows_x_columns, values)
    }

    fn backward<'a>(&'a mut self, dvalues: &Matrix, inputs: &Matrix) -> Matrix {
        let batches = inputs.row_count();
        let columns = self.filters * self.input_height * self.input_width;

        let mut values = vec![0.; batches * columns];
        for i in 0..self.max_indices.len() {
            values[self.max_indices[i]] += dvalues.read_at(i);
        }

        Matrix::from(batches, columns, values)
    }
}

#[cfg(test)]
mod tests {
    use colored::Colorize;

    use crate::{geoalg::f32_math::matrix::Matrix, nn::layers::Propagates};

    use super::MaxPooling;

    #[test]
    fn test_forward2x2() {
        // Will need to update code if we ever change to use tensors instead of matrices everywhere.
        // Create a matrix that represents 2 filters that are 4 x 4 for a single image.
        let tc = Matrix::from(1, 2 * 4 * 4, vec![
            1., 3., 2., 1.,  
            4., 2., 1., 5.,
            3., 1., 4., 2.,
            8., 6., 7., 9.,

            6., 4., 3., 8.,
            2., 4., 3., 7.,
            1., 5., 4., 3.,
            4., 7., 6., 4.
        ]);

        let mut pooling = MaxPooling::new(
            2, 
            2, 2, 
            4, 4,
            2);

        // Assume 2 filters
        let forward_output = pooling.forward(&tc);
        let expected = Matrix::from(1, 2 * 2 * 2, vec![
            4.0, 5.0,
            8.0, 9.0,
            
            6.0, 8.0,
            7.0, 6.0]);

        assert_eq!(forward_output, expected);

        // After the forward pass, the output should be a matrix that represents 2 filters of 2x2.
        let msg = format!("Forward pooling: {:?} <{:?}>", forward_output, pooling.max_indices).bright_magenta();
        println!("{msg}");

        // Assume these values came from the previous layers back-propagation.
        let dvalues = Matrix::from(1, 2 * 4, vec![
            0.2, -0.5,
            0.3, 0.1,

            -1., 4.,
            3., 8.
        ]);

        let backward_output = pooling.backward(&dvalues, &forward_output);
        let expected = Matrix::from(1, 2 * 4 * 4, vec![
            0.0, 0.0, 0.0, 0.0,
            0.2, 0.0, 0.0, -0.5,
            0.0, 0.0, 0.0, 0.0,
            0.3, 0.0, 0.0, 0.1,
            
            -1.0, 0.0, 0.0, 4.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 3.0, 8.0, 0.0
        ]);
        assert_eq!(backward_output, expected);

        // After backprop, we should get back to 2 filters of 4x4
        let msg = format!("Backprop pooling: {:?}", backward_output).bright_magenta();
        println!("{msg}");
    }
}