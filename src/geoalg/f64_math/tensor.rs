use super::matrix::Matrix;

const COLUMN:usize = 0;
const ROW:usize = 1;
const DEPTH:usize = 2;

#[derive(PartialEq, Debug, Clone)]
pub struct Tensor {
    pub values: Vec<f64>,
    pub shape: Vec<usize>,   // Rows, columns, depth. Everything after is arbitrary
    //dim_size: vec<usize>
}

impl Tensor {
    pub fn new(values: Vec<f64>, shape: Vec<usize>) -> Tensor {
        let zero = 0;
        let no_zeroes = !shape.contains(&zero);
        assert!(no_zeroes);

        let len = shape.iter().product();
        assert_eq!(values.len(), len, "The length of tensor doesn't match its shape.");

        Tensor {
            values,
            shape
        }
    }

    /// Helps with calculations
    pub fn get_dims(&self) -> Vec<usize> {
        //let x = self.shape.clone().iter().rev()
        let r = Vec::new();
        //for 
        r
    }

    /// Gets a specific value inside tensor.
    pub fn get_at(&self, coordinate: Vec<usize>) -> f64 {
        assert_eq!(self.shape.len(), coordinate.len());


        1.
    }

    /// Gets a matrix from this tensor at specified depth
    pub fn get_matrix(&self, depth: usize) -> Matrix{
        assert!(depth < self.shape[DEPTH], "Tensor not deep enough to get requested matrix.");

        let value_size = self.shape[ROW] * self.shape[COLUMN];
        let start = value_size * depth;
        let end = start + value_size;

        let mut r = Matrix {
            values: Vec::with_capacity(value_size),
            rows: self.shape[ROW],
            columns: self.shape[COLUMN]
        };

        r.values.copy_from_slice(&self.values[start..end]);
        r
    }

    /// Simple and naive valid cross correlation implementation with little to no optimizations.
    /// Assumes stride of 1
    /// Values are stored row major, then column, then depth
    /// So at depth 1, it is stored a 2d data set
    /// Will use to verify correctness of optimized versions
    /// Will assume depth of 1 for first pass
    pub fn valid_cross_correlation(&self, kernel: &Tensor) -> Tensor {
        assert!(self.shape[ROW] >= kernel.shape[ROW], "Lhs must have same or more rows than kernel.");
        assert!(self.shape[COLUMN] >= kernel.shape[COLUMN], "Lhs must have same or more columns than kernels.");

        let new_shape = vec![
            self.shape[ROW] - kernel.shape[ROW] + 1,
            self.shape[COLUMN] - kernel.shape[COLUMN] + 1
        ];

        let mut value_stream = Vec::with_capacity(new_shape[ROW] * new_shape[COLUMN]);

        // Sliding row window for lhs
        for row in 0..new_shape[ROW] {
            // Sliding column window for lhs
            for column in 0..new_shape[COLUMN] {
                let mut c_accum = 0.;
                for kernel_row in 0..kernel.shape[ROW] {
                    for kernel_column in 0..kernel.shape[COLUMN] {
                        let x = self.get_at(vec![row + kernel_row, column + kernel_column]);
                        let y = kernel.get_at(vec![kernel_row, kernel_column]);                        
                        
                        c_accum += x * y;
                    }
                }

                value_stream.push(c_accum);
            }
        }

        Tensor {
            values: value_stream,
            shape: new_shape
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_correlation() {
        let _lhs = Tensor {
            values: vec![
                0., 0.15, 0.,
                0.15, 0.4, 0.15,
                0., 0.15, 0. 
            ],
            shape: vec![3, 3, 1]
        };
    }
}