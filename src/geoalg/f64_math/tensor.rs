use super::matrix::Matrix;

// Dimension constants
const COLUMN:usize = 0;
const ROW:usize = 1;
const DEPTH:usize = 2;

#[derive(PartialEq, Debug, Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,   // Rows, columns, depth. Everything after is arbitrary
    pub values: Vec<f64>
}

impl Tensor {
    pub fn new(shape: Vec<usize>, values: Vec<f64>, ) -> Tensor {
        let zero = 0;
        let no_zeroes = !shape.contains(&zero);
        assert!(no_zeroes, "Shape cannot have dimensions of zero.");

        let len = shape.iter().product();
        assert_eq!(values.len(), len, "The length of tensor doesn't match specified shape.");

        Tensor {
            values,
            shape
        }
    }

    /// Unstable internal
    /// Helps with calculations
    fn get_dims(&self) -> Vec<usize> {
        let mut r = Vec::with_capacity(self.shape.len());
        r.push(1);

        for i in 0..self.shape.len() {
            r.push(self.shape[i] * r[i]);
        }

        r
    }

    /// Gets a specific value inside tensor.
    /// Does not check each individual shape's bounds (yet?)
    pub fn get_at(&self, coordinate: Vec<usize>) -> f64 {
        assert_eq!(self.shape.len(), coordinate.len());

        let dims = self.get_dims();

        let index: usize = dims.iter().zip(coordinate.iter()).map(|(&a, &b)| a*b).sum();

        self.values[index]
    }

    /// Performs no bounds checking.
    fn get_at_3(&self, column: usize, row: usize, depth: usize) -> f64 {
        self.values[column + row * self.shape[0] + depth * self.shape[0] * self.shape[1]]
    }

    /// Simple and naive valid cross correlation implementation with little to no optimizations.
    /// Assumes stride of 1
    /// Values are stored row major, then column, then depth
    /// So at depth 1, it is stored as a 2d data set
    /// Will use to verify correctness of optimized versions
    /// Will assume depth of 1 for first pass
    pub fn valid_cross_correlation(&self, kernel: &Tensor) -> Tensor {
        assert!(self.shape[ROW] >= kernel.shape[ROW], "Lhs must have same or more rows than kernel.");
        assert!(self.shape[COLUMN] >= kernel.shape[COLUMN], "Lhs must have same or more columns than kernels.");
        assert_eq!(self.shape[DEPTH], kernel.shape[DEPTH], "Lhs and kernel must have same depth.");

        let new_shape = vec![
            self.shape[ROW] - kernel.shape[ROW] + 1,
            self.shape[COLUMN] - kernel.shape[COLUMN] + 1
        ];

        let mut value_stream = Vec::with_capacity(new_shape[ROW] * new_shape[COLUMN]);
        //let lhs_dims = self.get_dims();
        //let kernel_dims = kernel.get_dims();

        for row in 0..new_shape[ROW] {
            for column in 0..new_shape[COLUMN] {
                let mut c_accum = 0.;

                // Slide kernel window horizontally and then vertically
                for kernel_row in 0..kernel.shape[ROW] {
                    for kernel_column in 0..kernel.shape[COLUMN] {
                        let x = self.get_at_3(row + kernel_row, column + kernel_column, 0);
                        let y = kernel.get_at_3(kernel_row, kernel_column, 0);                        
                        
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
    pub fn test_get_at() {
        let tc = Tensor {
            shape: vec![4,3,2],
            values: vec![
                1., 2., 3., 4.,
                5., 6., 7., 8.,
                9., 10., 11., 12.,

                13., 14., 15., 16.,
                17., 18., 19., 20.,
                21., 22., 23., 24.
            ]
        };

        let actual = tc.get_at(vec![2, 1, 0]);
        let expected = 7.;
        assert_eq!(actual, expected);

        let actual = tc.get_at(vec![1,2,1]);
        let expected = 22.;
        assert_eq!(actual, expected);
    }

    #[test]
    pub fn test_get_dims() {
        let tc1 = Tensor {
            shape: vec![4, 3, 2],
            values: vec![0.;24]
        };

        let actual = tc1.get_dims();
        let expected = vec![1, 4, 12, 24];

        assert_eq!(actual, expected);
    }

    #[ignore = "reason"]
    #[test]
    fn test_cross_correlation() {
        let lhs = Tensor {
            shape: vec![4, 4, 2],
            values: vec![
                1., 2., 3., 4., 
                5., 6., 7., 8.,
                9., 10., 11., 12.,
                13., 14., 15., 16.,

                10., 20., 30., 40.,
                50., 60., 70., 80.,
                90., 100., 110., 120.,
                130., 140., 150., 160.
            ]
        };

        let kernel = Tensor {
            shape: vec![3, 3, 2],
            values: vec![
                0., 0.15, 0.,
                0.15, 0.4, 0.15,
                0., 0.15, 0., 

                0., 0.10, 0.,
                0.10, 0.6, 0.10,
                0., 0.10, 0., 
            ]
        };

        let actual = lhs.valid_cross_correlation(&kernel);
        let expected = Tensor {
            shape: vec![2, 2, 1],
            values: vec![]
        };

        assert_eq!(actual, expected);
    }
}