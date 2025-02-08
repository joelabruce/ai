use crate::geoalg::f32_math::matrix::Matrix;

use super::Propagates;

/// Currently only supports valid pooling layers, with no padding.
pub struct MaxPoolingLayer {
    height: usize,
    width: usize,
    stride: usize,

    max_indices: Vec<usize>
}

impl MaxPoolingLayer {
    pub fn new(height: usize, width: usize, stride: usize) -> Self {
        MaxPoolingLayer {
            width, height, stride, max_indices: vec![0]
        }
    }
}

impl Propagates for MaxPoolingLayer {
    fn forward(&mut self, inputs: &Matrix) -> Matrix {
        let columns = (inputs.column_count() - self.width) / self.stride + 1;
        let rows = (inputs.row_count() - self.height) / self.stride + 1; 

        let mut values = Vec::with_capacity(rows * columns);
        self.max_indices = Vec::with_capacity(rows * columns);
        for row in 0..rows {
            let w_row = row * self.stride;
            for column in 0..columns {
                let w_column = column * self.stride;

                let mut max = f32::MIN;
                let mut max_index = 0;
                for k_row in 0..self.height {
                    for k_column in 0..self.width {
                        let index = (w_column + k_column) + (w_row + k_row) * inputs.column_count();
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

        Matrix::from(rows, columns, values)
    }

    fn backward<'a>(&'a mut self, dvalues: &Matrix, inputs: &Matrix) -> Matrix {
        let columns = self.stride * (inputs.column_count() - 1) + self.width;
        let rows = self.stride * (inputs.row_count() - 1) + self.height;
        
        let mut values = vec![0.; rows * columns];

        for i in 0..self.max_indices.len() {
            values[self.max_indices[i]] += dvalues.read_at(i);
        }

        Matrix::from(rows, columns, values)
    }
}

#[cfg(test)]
mod tests {
    use crate::{geoalg::f32_math::matrix::Matrix, nn::layers::Propagates};

    use super::MaxPoolingLayer;

    #[test]
    fn test_forward2x2() {
        let tc = Matrix::from(4, 4, vec![
            1., 3., 2., 1.,  
            4., 2., 1., 5.,
            3., 1., 4., 2.,
            8., 6., 7., 9.
        ]);

        let mut pooling = MaxPoolingLayer::new(2, 2, 2);
        let forward_output = pooling.forward(&tc);

        println!("Forward pooling {:?} <{:?}>", forward_output, pooling.max_indices);

        let dvalues = Matrix::from(2, 2, vec![
            0.2, -0.5,
            0.3, 0.1
        ]);

        let x = pooling.backward(&dvalues, &forward_output);
        println!("Backprop pooling: {:?}", x);
    }
}