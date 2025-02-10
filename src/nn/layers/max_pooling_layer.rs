use crate::geoalg::f32_math::matrix::Matrix;
use super::Propagates;

/// Currently only supports valid pooling layers, with no padding.
pub struct MaxPoolingLayer {
    pooling_height: usize,
    pooling_width: usize,
    stride: usize,
    input_width: usize,
    input_height: usize,

    max_indices: Vec<usize>
}

impl MaxPoolingLayer {
    pub fn new(pooling_height: usize, pooling_width: usize, input_width: usize, input_height: usize, stride: usize) -> Self {
        MaxPoolingLayer {
            pooling_width, pooling_height, input_height, input_width, stride, max_indices: vec![0]
        }
    }
}

impl Propagates for MaxPoolingLayer {
    fn forward(&mut self, inputs: &Matrix) -> Matrix {
        let columns = (self.input_width - self.pooling_width) / self.stride + 1;
        let rows = (self.input_width - self.pooling_height) / self.stride + 1;
        let batches = inputs.row_count();
        let rows_x_columns = rows * columns;

        let mut values = Vec::with_capacity(batches * rows_x_columns);
        self.max_indices = Vec::with_capacity(batches * rows_x_columns);
        for batch in 0..batches {
            let batch_offset = batch * self.input_height * self.input_width;
            for row in 0..rows {
                let w_row = row * self.stride;
                for column in 0..columns {
                    let w_column = column * self.stride;

                    let mut max = f32::MIN;
                    let mut max_index = 0;
                    for k_row in 0..self.pooling_height {
                        for k_column in 0..self.pooling_width {
                            let index = batch_offset + (w_column + k_column) + (w_row + k_row) * self.input_width;
                            let index_value = inputs.read_at(index);
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

        Matrix::from(batches, rows_x_columns, values)
    }

    fn backward<'a>(&'a mut self, dvalues: &Matrix, inputs: &Matrix) -> Matrix {
        let batches = inputs.row_count();

        let mut values = vec![0.; batches * self.input_width * self.input_height];

        for i in 0..self.max_indices.len() {
            values[self.max_indices[i]] += dvalues.read_at(i);
        }

        Matrix::from(batches, self.input_width * self.input_height, values)
    }
}

#[cfg(test)]
mod tests {
    use colored::Colorize;

    use crate::{geoalg::f32_math::matrix::Matrix, nn::layers::Propagates};

    use super::MaxPoolingLayer;

    #[test]
    fn test_forward2x2() {
        let tc = Matrix::from(2, 4 * 4, vec![
            1., 3., 2., 1.,  
            4., 2., 1., 5.,
            3., 1., 4., 2.,
            8., 6., 7., 9.,

            6., 4., 3., 8.,
            2., 4., 3., 7.,
            1., 5., 4., 3.,
            4., 7., 6., 4.
        ]);

        let mut pooling = MaxPoolingLayer::new(
            2, 2, 
            4, 4, 2);

        let forward_output = pooling.forward(&tc);

        let msg = format!("Forward pooling: {:?} <{:?}>", forward_output, pooling.max_indices).bright_magenta();
        println!("{msg}");

        let dvalues = Matrix::from(1, 4, vec![
            0.2, -0.5,
            0.3, 0.1,

            -1., 4.,
            3., 8.
        ]);

        let x = pooling.backward(&dvalues, &forward_output);
        let msg = format!("Backprop pooling: {:?}", x).bright_magenta();
        println!("{msg}");
    }
}