use std::thread;
use rand::distributions::{Distribution, Uniform};
use crate::geoalg::f64_math::optimized_functions::*;

/// To be adapted later.
/// WIP
// pub fn argmax(values: &[f64]) -> usize {
//     values
//         .iter()
//         .enumerate()
//         .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
//         .map(|(index, _)| index).unwrap()
// }

/// Matrix is implemented as a single dimensional vector of f64s.
/// This implementation of Matrix is row-major. 
/// Row-major is specified so certain optimizations and parallelization can be performed.
/// Column-major is not yet implemented.
#[derive(PartialEq, Debug, Clone)]
pub struct Matrix {
    rows: usize,
    columns: usize,
    values: Vec<f64>
}

impl Matrix {
    pub fn row_count(&self) -> usize {
        self.rows
    }

    pub fn column_count(&self) -> usize {
        self.columns
    }

    pub fn read_values(&self) -> &[f64] {
        &self.values
    }

    /// Create a matrix from a vector.
    /// # Arguments
    /// # Returns
    pub fn from_vec(values: Vec<f64>, rows: usize, columns: usize) -> Self {
        assert_eq!(rows * columns, values.len());
        
        Matrix {
            rows,
            columns,
            values
        }
    }

    /// Creates a matrix with columns * size elements where every element is zero
    /// # Arguments
    /// # Returns
    pub fn new_zeroed(rows: usize, columns: usize) -> Self {
        assert!(columns > 0);
        assert!(rows > 0);

        let capacity = rows * columns;
        let values = vec![0.0f64; capacity];

        Self {
            columns,
            rows,
            values: values
        }
    }

    /// Possibly faster way to implement an identity matrix.
    pub fn new_identity(n: usize) -> Self {
        assert!(n > 0);

        let values = (0..n*n).map(|i| kronecker_delta_f64(i % (n + 1), 0)).collect();

        Self {
            columns: n,
            rows: n,
            values
        }
    }

    /// Returns an row x column matrix filled with random values between -1.0 and 1.0 inclusive.
    /// # Arguments
    /// # Returns
    pub fn new_randomized_z(rows: usize, columns: usize) -> Self {
        assert!(columns > 0);
        assert!(rows > 0);

        let mut rng = rand::thread_rng();
        let step = Uniform::new_inclusive(-1.0, 1.0);
        let element_count = columns * rows;
        let values = step.sample_iter(&mut rng).take(element_count).collect();

        Self {
            columns,
            rows,
            values,
        }
    }

    /// Returns an ixj matrix filled with random values specified by uniform distribution.
    /// # Arguments
    /// # Returns
    pub fn new_randomized_uniform(rows: usize, columns: usize, uniform: Uniform<f64>) -> Self {
        assert!(columns > 0);
        assert!(rows > 0);

        let mut rng = rand::thread_rng();
        let step = uniform;// Uniform::new_inclusive(-0.15f64, 0.15f64);
        let element_count = columns * rows;
        let values = step.sample_iter(&mut rng).take(element_count).collect();

        Self {
            columns,
            rows,
            values,
        }
    }

    /// Returns slice of matrix that is a row of the matrix
    /// # Arguments
    /// # Returns
    pub fn get_row_vector_slice(&self, row: usize) -> &[f64] {
        assert!(row < self.rows, "Tried to get a row that was out of bounds.");

        let start = row * self.columns;
        let end = start + self.columns;
        &self.values[start..end]
    }

    /// Returns a newly allocated matrix that is the transpose of the matrix operated on.
    /// # Arguments
    /// # Returns
    pub fn get_transpose(&self) -> Matrix {
        let capacity = self.rows * self.columns;
        let mut transposed = Vec::with_capacity(capacity);

        for i in 0..capacity {
            let index_to_push = self.columns * (i % self.rows) + i / self.rows;

            transposed.push(self.values[index_to_push]);
        }

        Matrix {
            columns: self.rows,
            rows: self.columns,
            values: transposed
        }
    }

    /// Element-wise multiplication.
    pub fn elementwise_multiply(&self, rhs: &Matrix) -> Matrix {
        assert_eq!(self.columns, rhs.columns, "Columns must match for elementwise multiply.");
        assert_eq!(self.rows, rhs.rows, "Rows must match for elementwise multiply.");

        let values = self.values
            .iter()
            .zip(rhs.values.iter())
            .map(|(&x, &y)| x * y)
            .collect();

        Matrix {
            rows: self.rows,
            columns: self.columns,
            values
        }
    }

    /// Row-wise multi-threaded element-wise multiply
    pub fn elementwise_multiply_threaded(&self, rhs: &Matrix) -> Matrix {
        assert_eq!(self.columns, rhs.columns, "Columns must match for elementwise multiply.");
        assert_eq!(self.rows, rhs.rows, "Rows must match for elementwise multiply.");

        let num_cores = thread::available_parallelism().unwrap().get();
        let rows_per_core = self.rows / num_cores;

        if rows_per_core < 1 {
        // If there are not enough rows, must use regular multiplication
        // Otherwise it fails because there is not enough data to parallelize
            return self.elementwise_multiply(&rhs);
        }

        let mut thread_join_handles = vec![];
        let spread = self.rows % num_cores;

        let mut rows_curosr = 0;
        for core in 0..num_cores {
            let lhs = self.clone(); // How do I do avoid cloning?
            let rhs = rhs.clone();  // How do I avoid cloning?

            let rows_to_handle = rows_per_core + if core < spread { 1 } else { 0 };
            let partition_start = rows_curosr;
            let partition_end = partition_start + rows_to_handle - 1;

            rows_curosr = partition_end + 1;

            thread_join_handles.push(thread::spawn(move || {
                let mut partition_values: Vec<f64> = Vec::with_capacity(rows_to_handle * rhs.rows);

                for row in partition_start..=partition_end {
                    //////
                    // Actual work inside of thread to be done
                    let ls = lhs.get_row_vector_slice(row);
                    let rs = rhs.get_row_vector_slice(row);
                        
                    let values = ls
                        .iter()
                        .zip(rs)
                        .map(|(&l, &r)| l * r);
                    // End of actual parallelized work
                    //////

                    partition_values.extend(values);
                }                
                
                partition_values
            }));
        }

        let mut values = Vec::with_capacity(self.rows * rhs.rows);
        for thread in thread_join_handles {
            match thread.join() {
                Ok(r) => { values.extend(r); },
                Err(_e) => { } 
            }
        }

        Matrix {
            rows: self.rows,
            columns: self.columns,
            values
        }
    }

    /// Multiplies two matrices using transpose operation for efficiency.
    /// Willing to take performance hit [O(m x n)] for small matrices for creating transposed matrix.
    /// 
    pub fn mul(&self, rhs: &Matrix) -> Matrix {
        assert_eq!(self.columns, rhs.rows, "When multiplying matrices, lhs columns must equal rhs rows.");
        let transposed = rhs.get_transpose();
        self.mul_with_transposed(&transposed)
    }

    /// Will fallback to normal implementation if not enough rows to parallelize operations on
    pub fn mul_threaded_rowwise(&self, rhs: &Matrix) -> Matrix {
        assert_eq!(self.columns, rhs.rows, "When multiplying matrices, lhs columns must equal rhs rows.");
        let transposed = rhs.get_transpose();
        self.mul_with_transposed_threaded_rowwise(&transposed)
     }

    /// Faster multiplcation when you need to multiply the transposed matrix of rhs.
    /// Avoids calculating the transpose twice.
    pub fn mul_with_transposed(&self, rhs: &Matrix) -> Matrix {
        assert_eq!(self.columns, rhs.columns, "When multiplying with transposed, columns must be equal for lhs and rhs.");

        let r_size = self.rows * rhs.rows;
        let mut values = Vec::with_capacity(r_size);

        for row in 0..self.rows {
            let ls = self.get_row_vector_slice(row);
            for transposed_row in 0..rhs.rows {
                let rs = rhs.get_row_vector_slice(transposed_row);
                let x = dot_product_of_vector_slices(&ls, &rs);
                values.push(x);
            }
        }

        Matrix {
            columns: rhs.rows,
            rows: self.rows,
            values
        }        
    }

    /// Matrix multiplication using transposed row-wise multi-threading.
    pub fn mul_with_transposed_threaded_rowwise(&self, rhs: &Matrix) -> Matrix {
        assert_eq!(self.columns, rhs.columns, "When multiplying with transposed, columns must be equal for lhs and rhs.");
        let num_cores = thread::available_parallelism().unwrap().get();
        let rows_per_core = self.rows / num_cores;

        if rows_per_core < 1 {
        // If there are not enough rows, must use regular multiplication
        // Otherwise it fails because there is not enough data to parallelize
            return self.mul_with_transposed(&rhs);
        }

        let mut thread_join_handles = vec![];
        let spread = self.rows % num_cores;

        let mut rows_curosr = 0;
        for core in 0..num_cores {
            let lhs = self.clone(); // How do I do avoid cloning?
            let rhs = rhs.clone();  // How do I avoid cloning?

            let rows_to_handle = rows_per_core + if core < spread { 1 } else { 0 };
            let partition_start = rows_curosr;
            let partition_end = partition_start + rows_to_handle - 1;

            rows_curosr = partition_end + 1;

            thread_join_handles.push(thread::spawn(move || {
                 let mut partition_values: Vec<f64> = Vec::with_capacity(rows_to_handle * rhs.rows);

                for row in partition_start..=partition_end {
                    //////
                    // Actual work inside of thread to be done
                    let ls = lhs.get_row_vector_slice(row);
                    for transposed_row in 0..rhs.rows {
                        let rs = rhs.get_row_vector_slice(transposed_row);
                        let x = dot_product_of_vector_slices(&ls, &rs);
                        partition_values.push(x);
                    }
                    // End of actual parallelized work
                    //////
                }                
                
                partition_values
            }));
        }

        let mut values = Vec::with_capacity(self.rows * rhs.rows);
        for thread in thread_join_handles {
            match thread.join() {
                Ok(r) => { values.extend(r); },
                Err(_e) => { } 
            }
        }

        Matrix {
            rows: self.rows,
            columns: rhs.rows,
            values
        }
    }

    /// Returns size of underlying vector.
    pub fn len(&self) -> usize {
        self.columns * self.rows
    }

    /// Useful for applying an activation function to the entire matrix.
    /// *Allows to map capturing a variable outside of the closure.
    /// # Arguments
    /// # Returns
    pub fn map_with_capture(&self, func: impl Fn(&f64) -> f64) -> Matrix {
        let values = self.values.iter().map(|val| func(val)).collect();
        
        Matrix {
            rows: self.rows,
            columns: self.columns,
            values: values
        } 
    }

    /// Useful for applying an activation function to the entire matrix.
    /// # Arguments
    /// # Returns
    pub fn map(&self, func: fn(&f64) -> f64) -> Matrix {
        let values = self.values.iter().map(|&val| func(&val)).collect();
        
        Matrix {
            rows: self.rows,
            columns: self.columns,
            values: values
        } 
    }

    /// Elementwise difference of two matrices.
    /// # Arguments
    /// # Returns
    pub fn sub(&self, rhs: &Matrix) -> Matrix {
        assert!(self.columns == rhs.columns && self.rows == rhs.rows, "Cannot subtract matrices with different orders.");

        let values = self.values
            .iter()
            .zip(rhs.values.iter())
            .map(|(x, y)| x - y)
            .collect();

        Matrix {
            columns: rhs.columns,
            rows: rhs.rows,
            values
        }
    }

    /// Adds two matrices together. Efficient and easy because both matrices must have same order.
    pub fn add(&self, rhs: &Matrix) -> Matrix {
        assert_eq!(self.columns, rhs.columns, "Columns of lhs and rhs must be equal when adding matrices.");
        assert_eq!(self.rows, rhs.rows, "Rows of lhs and rhs must be equal when adding matrices.");

        let values = self.values
            .iter()
            .zip(rhs.values.iter())
            .map(|(x, y)| x + y)
            .collect();

        Matrix {
            columns: rhs.columns,
            rows: rhs.rows,
            values
        }
    }

    /// Elementwise division of matrix by scalar.
    /// # Arguments
    /// # Returns
    pub fn div_by_scalar(&self, scalar: f64) -> Matrix {
        assert_ne!(scalar, 0.0, "Cannot divide matrix elements by zero.");
        let values = self.values
            .iter()
            .map(|x| x / scalar)
            .collect();

        Matrix {
            rows: self.rows,
            columns: self.columns,
            values
        }
    }

    /// Adds a Matrix of shape 1xn to every column. Each matrix must have same number of rows and rhs must have exactly 1 column.
    pub fn add_column_vector(&self, rhs: &Matrix) -> Matrix {
        assert_eq!(rhs.columns, 1, "Rhs matrix must have 1 column.");
        assert_eq!(self.rows, rhs.rows, "Lhs and rhs must have equal number of rows.");

        let mut r = Vec::with_capacity(self.len());
        
        for row in 0..self.rows {
            let lhs_row = self.get_row_vector_slice(row);
            let new_row = lhs_row.iter().map(|f| f + rhs.values[row]);
            r.extend(new_row);
        }

        Matrix {
            columns: self.columns,
            rows: self.rows,
            values: r
        }
    }

    /// Adds a given row to each row in lhs matrix.
    pub fn add_row_vector(&self, rhs: &Matrix) -> Matrix {
        assert_eq!(rhs.rows, 1, "Rhs matrix must have 1 row.");
        assert_eq!(self.columns, rhs.columns, "Lhs and rhs must have equal number of columns.");

        let mut r = Vec::with_capacity(self.len());
        for row in 0..self.rows{
            let x = self.get_row_vector_slice(row);
            let y = rhs.get_row_vector_slice(0);

            let xplusy = x.iter().zip(y.iter()).map(|(a, b)| a + b).collect::<Vec<f64>>();

            r.extend(xplusy);
        }

        Matrix {
            rows: self.rows,
            columns: self.columns,
            values: r
        }
    }

    /// Sums each row in self, and outputs a new matrix that is 1 row but same number of columns.
    pub fn shrink_rows_by_add(&self) -> Matrix {
        let mut v = Vec::with_capacity(self.columns);

        let t = self.get_transpose();
        for row in 0..t.rows {
            let x = t.get_row_vector_slice(row).iter().sum();
            v.push(x);
        }

        Matrix {
            rows: 1,
            columns: self.columns,
            values: v
        }
    }

    /// Scales matrix by scalar.
    pub fn scale(&self, scalar: f64) -> Matrix {
        let values = self.values.iter().map(|&x| x * scalar).collect();

        Matrix {
            rows: self.rows,
            columns: self.columns,
            values
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_vec() {
        let actual = Matrix::from_vec(vec![
            1., 2., 3.,
            4., 5., 6.
        ], 2, 3);

        let expected = Matrix {
            rows: 2,
            columns: 3,
            values: vec![
                1., 2., 3.,
                4., 5., 6.
            ]
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_add_column() {
        let lhs = Matrix {
            rows: 3,
            columns: 3,
            values: vec![
                1f64, 2f64, 3f64,
                4f64, 5f64, 6f64,
                7f64, 8f64, 9f64
            ]
        };

        let rhs = Matrix {
            rows: 3,
            columns: 1,
            values: vec![
                10.0,
                20.0,
                30.0
            ]
        };

        let expected = Matrix {
            rows: 3,
            columns: 3,
            values: vec![
                11.0, 12.0, 13.0,
                24.0, 25.0, 26.0,
                37.0, 38.0, 39.0
            ]
        };

        let actual = lhs.add_column_vector(&rhs);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_matrix_add() {
        let lhs = Matrix {
            rows: 3,
            columns: 3,
            values: vec![
                1f64, 2f64, 3f64,
                4f64, 5f64, 6f64,
                7f64, 8f64, 9f64
            ]
        };

        let rhs = Matrix {
            rows: 3,
            columns: 3,
            values: vec![
                -1f64, -2f64, -3f64,
                -4f64, -5f64, -6f64,
                -7f64, -8f64, -9f64
            ]
        };

        let actual = lhs.add(&rhs);

        let expected = Matrix {
            rows: 3,
            columns: 3,
            values: vec![
                0f64, 0f64, 0f64,
                0f64, 0f64, 0f64,
                0f64, 0f64, 0f64
            ]
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn transpose_test() {
        let m = Matrix {
            rows: 5,
            columns: 4,
            values: vec![
                0f64, 1f64, 2f64, 3f64,
                4f64, 5f64, 6f64, 7f64,
                8f64, 9f64, 10f64, 11f64,
                12f64, 13f64, 14f64, 15f64,
                16f64, 17f64, 18f64, 19f64
            ]
        };

        let expected = Matrix {
            rows: 4,
            columns: 5,
            values: vec![
                0f64, 4f64, 8f64, 12f64, 16f64,
                1f64, 5f64, 9f64, 13f64, 17f64,
                2f64, 6f64, 10f64, 14f64, 18f64,
                3f64, 7f64, 11f64, 15f64, 19f64
            ]
        };
        
        let actual = m.get_transpose();
        assert_eq!(actual, expected);
    }

    #[test]
    fn identity_1() {
        let actual = Matrix::new_identity(1);
        let expected = Matrix{
            rows: 1,
            columns: 1,
            values: vec![1.0f64]
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn identity_2() {
        let actual = Matrix::new_identity(2);
        let expected = Matrix {
            rows: 2,
            columns: 2,
            values: vec![
                1.0f64, 0.0f64,
                0.0f64, 1.0f64]
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn identity_3() {
        let actual = Matrix::new_identity(3);
        let expected = Matrix {
            rows: 3,
            columns: 3,
            values: vec![
                1.0f64, 0.0f64, 0.0f64, 
                0.0f64, 1.0f64, 0.0f64, 
                0.0f64, 0.0f64, 1.0f64]
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn random_matrix() {
        let m28x28 = Matrix::new_randomized_z(28, 28);

        let _r =m28x28.values.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ");
    }

     #[test]
    fn matrix_mul() {
        // Given
        let lhs = Matrix {
            rows: 4,
            columns: 3,
            values: vec![
                1f64, 2f64, 3f64,
                4f64, 5f64, 6f64,
                7f64, 8f64, 9f64,
                10f64, 11f64, 12f64]
        };

        let rhs = Matrix {
            rows: 3,
            columns: 5,
            values: vec![
                1f64, 2f64, 3f64, 4f64, 5f64,
                6f64, 7f64, 8f64, 9f64, 10f64,
                11f64, 12f64, 13f64, 14f64, 15f64
            ]
        };

        let actual = Matrix::mul(&lhs, &rhs);

        // Resultant matrix needs to have aas many rows as lhs, and as many columns as rhs.
        let expected = Matrix {
            rows: 4,
            columns: 5,
            values: vec! [
                46f64, 52f64, 58f64, 64f64, 70f64,
                100f64, 115f64, 130f64, 145f64, 160f64,
                154f64, 178f64, 202f64, 226f64, 250f64,
                208f64, 241f64, 274f64, 307f64, 340f64
            ]
        };

        assert!(actual == expected);

        let actual_threaded = Matrix::mul_threaded_rowwise(&lhs, &rhs);
        assert_eq!(actual_threaded, expected);
    }

    #[test]
    fn threaded_mul_equals_regular_mul() {
        let lhs = Matrix::new_randomized_z(96, 784);
        let rhs = Matrix::new_randomized_z(128, 784);

        let actual = lhs.mul_with_transposed_threaded_rowwise(&rhs);
        let expected = lhs.mul_with_transposed(&rhs);

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_scale() {
        let tc = Matrix {
            rows: 2,
            columns: 3,
            values: vec![
                1., 2., 3.,
                4., 5., 6.
            ]
        };

        let actual = tc.scale(3.);
        let expected = Matrix {
            rows: 2,
            columns: 3,
            values: vec![
                3., 6., 9.,
                12., 15., 18.
            ]
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_shrink_rows_by_add() {
        let tc = Matrix {
            columns: 3,
            rows: 3,
            values: vec![
                1., 1., 2.,
                2., 3., 3.,
                4., 4., 5.
            ]
        };

        let actual = tc.shrink_rows_by_add();
        let expected = Matrix {
            columns: 3,
            rows: 1,
            values: vec![
                7., 8., 10.
            ]
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_get_row_vector_slice() {
        let tc = Matrix {
            columns: 3,
            rows: 4,
            values: vec![
                1., 2., 3.,
                10., 20., 30.,
                100., 200., 300.,
                1000., 2000., 3000.
            ]
        };

        let actual = tc.get_row_vector_slice(2);
        let expected = &[100., 200., 300.];

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_element_wise_multiply_threaded() {
        let lhs = Matrix::new_randomized_z(1000, 3001);
        let rhs = Matrix::new_randomized_z(1000, 3001);

        let expected = lhs.elementwise_multiply(&rhs);
        let actual = lhs.elementwise_multiply_threaded(&rhs);

        assert_eq!(actual, expected);

    }
}
