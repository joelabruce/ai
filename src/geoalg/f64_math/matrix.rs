use std::thread;
use rand::distributions::{Distribution, Uniform};
use crate::{geoalg::f64_math::optimized_functions::*, Partitioner, Partition};

/// Matrix is implemented as a single dimensional vector of f64s.
/// This implementation of Matrix is row-major. 
/// Row-major is specified so certain optimizations and parallelization can be performed.
/// Column-major is not yet implemented.
#[derive(PartialEq, Debug, Clone, Default)]
pub struct Matrix {
    rows: usize,
    columns: usize,
    values: Vec<f64>,
    rowwise_partitioner: Partitioner
}

impl Matrix {
    /// Returns number of rows this matrix has.
    pub fn row_count(&self) -> usize { self.rows }

    /// Returns number of columns this matrix has.
    pub fn column_count(&self) -> usize { self.columns }

    /// Returns a slice of the values this matrix has.
    pub fn read_values(&self) -> &[f64] { &self.values }

    /// Create a matrix from a vector. Move the values into the matrix.
    /// Creates a default partitoner to use.
    /// Use this instead of manually doing struct everywhere if you do not want to specify a partitioner for parallelism.
    pub fn from_vec(values: Vec<f64>, rows: usize, columns: usize) -> Self {
        assert_eq!(rows * columns, values.len());

        let partitioner = Partitioner::with_partitions(rows, thread::available_parallelism().unwrap().get());
        
        Self::from_vec_with_partitioner(values, rows, columns, &partitioner)
    }

    /// Exhaustive Matrix instantiation.
    /// Do not move partitioner into Matrix, make a clone of it.
    /// Partitioner meant to be lightweight, so hopefully the performance improvement is more than the clone.
    /// Might look into something that allows for sharing a reference later for even more performance.
    pub fn from_vec_with_partitioner(values: Vec<f64>, rows: usize, columns: usize, rowwise_partitioner: &Partitioner) -> Self {
        assert_eq!(rows * columns, values.len());

        Self {
            rows,
            columns,
            values,
            rowwise_partitioner: rowwise_partitioner.clone()
        }
    }

    /// Possibly faster way to implement an identity matrix.
    pub fn new_identity(n: usize) -> Self {
        assert!(n > 0);

        let values = (0..n*n).map(|i| kronecker_delta_f64(i % (n + 1), 0)).collect();

        Self::from_vec(values, n, n)
    }

    /// Returns a row x column matrix filled with random values between -1.0 and 1.0 inclusive.
    pub fn new_randomized_z(rows: usize, columns: usize) -> Self {
        assert!(columns > 0);
        assert!(rows > 0);

        let mut rng = rand::thread_rng();
        let step = Uniform::new_inclusive(-1.0, 1.0);
        let element_count = columns * rows;
        let values = step.sample_iter(&mut rng).take(element_count).collect();

        Self::from_vec(values, rows, columns)
    }

    /// Returns an rows x column matrix filled with random values specified by uniform distribution.
    pub fn new_randomized_uniform(rows: usize, columns: usize, uniform: Uniform<f64>) -> Self {
        assert!(columns > 0);
        assert!(rows > 0);

        let mut rng = rand::thread_rng();
        let element_count = columns * rows;
        let values = uniform.sample_iter(&mut rng).take(element_count).collect();
        
        Self::from_vec(values, rows, columns)
    }

    /// Returns a contiguous slice of data representing columns in the matrix.
    pub fn get_row_vector_slice(&self, row: usize) -> &[f64] {
        assert!(row < self.rows, "Tried to get a row that was out of bounds.");

        let start = row * self.columns;
        let end = start + self.columns;
        &self.values[start..end]
    }

    /// Returns a newly allocated matrix that is the transpose of the matrix operated on.
    pub fn get_transpose(&self) -> Self {
        let capacity = self.values.len();
        let mut transposed = Vec::with_capacity(capacity);

        for i in 0..capacity {
            let index_to_push = self.columns * (i % self.rows) + i / self.rows;

            transposed.push(self.values[index_to_push]);
        }

        Self::from_vec(transposed, self.column_count(), self.row_count())
    }

    /// Element-wise multiplication.
    pub fn elementwise_multiply(&self, rhs: &Matrix) -> Self {
        assert_eq!(self.columns, rhs.columns, "Columns must match for elementwise multiply.");
        assert_eq!(self.rows, rhs.rows, "Rows must match for elementwise multiply.");

        let values = self.values
            .iter()
            .zip(rhs.values.iter())
            .map(|(&x, &y)| x * y)
            .collect();

        Self::from_vec(values, self.row_count(), self.column_count())
    }

    // /// Row-wise multi-threaded element-wise multiply
    pub fn elementwise_multiply_threaded(&self, rhs: &Matrix) -> Self {
        assert_eq!(self.columns, rhs.columns, "Columns must match for elementwise multiply.");
        assert_eq!(self.rows, rhs.rows, "Rows must match for elementwise multiply.");

        let num_cores = thread::available_parallelism().unwrap().get();
        let rows_per_core = self.rows / num_cores;

        if rows_per_core < 1 {
        // If there are not enough rows, must use regular multiplication
        // Otherwise it fails because there is not enough data to parallelize
            return self.elementwise_multiply(&rhs);
        }

        let spread = self.rows % num_cores;

        let mut rows_curosr = 0;
        let mut values = Vec::with_capacity(self.rows * rhs.rows);

        thread::scope(|s| {
            let mut thread_join_handles = Vec::with_capacity(num_cores);
            for core in 0..num_cores {
                let rows_to_handle = rows_per_core + if core < spread { 1 } else { 0 };
                let partition_start = rows_curosr;
                let partition_end = partition_start + rows_to_handle - 1;

                rows_curosr = partition_end + 1;

                thread_join_handles.push(s.spawn(move || {
                    let mut partition_values: Vec<f64> = Vec::with_capacity(rows_to_handle * rhs.rows);

                    for row in partition_start..=partition_end {
                        //////
                        // Actual work inside of thread to be done
                        let ls = self.get_row_vector_slice(row);
                        let rs = rhs.get_row_vector_slice(row);
                            
                        let values = ls
                            .iter()
                            .zip(rs)
                            .map(|(&l, &r)| l * r);
                        partition_values.extend(values);
                        // End of actual parallelized work
                        //////
                    }                
                    
                    partition_values
                }));
            }

            for thread in thread_join_handles {
                match thread.join() {
                    Ok(r) => { values.extend(r); },
                    Err(_e) => { } 
                }
            }
        });

        Self::from_vec(values, self.row_count(), self.column_count())
    }

    /// Makes use of supplied partitions to parallelize the operation.
    /// If partitions is cached, can be reused (to hopefully save even more time).
    pub fn mul_element_wise_partitioned(&self, rhs: &Matrix, partitioner: &Partitioner) -> Self {
        let inner_process = |partition: &Partition| {
            // We know in advance the exact capacity of result.
            // It will always be partition size * number of inner rows.
            let mut partition_values: Vec<f64> = Vec::with_capacity(partition.get_size() * self.columns);
            for row in partition.get_range() {
                let ls = self.get_row_vector_slice(row);
                let rs = rhs.get_row_vector_slice(row);

                for column in 0..ls.len() {
                    partition_values.push(rs[column] * ls[column]);
                }
            }

            partition_values
        };

        let values = partitioner.parallelized(inner_process);

        Self::from_vec(values, self.row_count(), self.column_count())//, &partitioner)
    }

    /// Multiplies two matrices using transpose operation for efficiency.
    /// Willing to take performance hit [O(m x n)] for small matrices for creating transposed matrix. 
    pub fn mul(&self, rhs: &Matrix) -> Self {
        assert_eq!(self.columns, rhs.rows, "When multiplying matrices, lhs columns must equal rhs rows.");
        let transposed = rhs.get_transpose();
        self.mul_with_transposed(&transposed)
    }

    /// Faster multiplcation when you need to multiply the transposed matrix of rhs.
    /// Avoids calculating the transpose twice.
    pub fn mul_with_transposed(&self, rhs: &Matrix) -> Self {
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

        Self::from_vec(values, self.row_count(), rhs.row_count())
    }

    /// Computes matrix multiplication and divying up work amongst partitions.
    pub fn mul_with_transposed_partitioned(&self, rhs: &Matrix, partitioner: &Partitioner) -> Matrix {
        let inner_process = move |partition: &Partition| {
            // We know in advance the exact capacity of result.
            // It will always be partition size * number of inner rows.
            let mut partition_values: Vec<f64> = Vec::with_capacity(partition.get_size() * rhs.rows);

            // This is start of the main logic of the code we care about
            for row in partition.get_range() {
                let ls = self.get_row_vector_slice(row);
                for transposed_row in 0..rhs.rows {
                    let rs = rhs.get_row_vector_slice(transposed_row);
                    let dot_product = dot_product_of_vector_slices(&ls, &rs);
                    partition_values.push(dot_product);
                }
            }
            // End of the main code we care abour

            partition_values
        };

        let values = partitioner.parallelized(inner_process);

        Self::from_vec(values, self.row_count(), rhs.row_count())
    }

    /// Returns size of underlying vector.
    pub fn len(&self) -> usize {
        self.columns * self.rows
    }

    /// Useful for applying an activation function to the entire matrix.
    /// *Allows to map capturing a variable outside of the closure.
    pub fn map_with_capture(&self, func: impl Fn(&f64) -> f64) -> Self {
        let values = self.values.iter().map(|val| func(val)).collect();
        
        Self::from_vec(values, self.row_count(), self.column_count())
    }

    /// Useful for applying an activation function to the entire matrix.
    pub fn map(&self, func: fn(&f64) -> f64) -> Self {
        let values = self.values.iter().map(|&val| func(&val)).collect();
        
        Self::from_vec(values, self.row_count(), self.column_count())
    }

    /// TODO: write in new _Matrix.
    /// Elementwise difference of two matrices.
    pub fn sub(&self, rhs: &Matrix) -> Self {
        assert!(self.columns == rhs.columns && self.rows == rhs.rows, "Cannot subtract matrices with different orders.");

        let values = self.values
            .iter()
            .zip(rhs.values.iter())
            .map(|(x, y)| x - y)
            .collect();

        Self::from_vec(values, self.row_count(), self.column_count())
    }

    /// Adds two matrices together. Efficient and easy because both matrices must have same order.
    pub fn add(&self, rhs: &Matrix) -> Self {
        assert_eq!(self.columns, rhs.columns, "Columns of lhs and rhs must be equal when adding matrices.");
        assert_eq!(self.rows, rhs.rows, "Rows of lhs and rhs must be equal when adding matrices.");

        let values = self.values
            .iter()
            .zip(rhs.values.iter())
            .map(|(&x, &y)| x + y)
            .collect();

        Self::from_vec(values, self.row_count(), self.column_count())
    }

    /// Adds a given row to each row in lhs matrix.
    pub fn add_row_vector(&self, rhs: &Matrix) -> Self {
        assert_eq!(rhs.rows, 1, "Rhs matrix must have 1 row.");
        assert_eq!(self.columns, rhs.columns, "Lhs and rhs must have equal number of columns.");

        let mut values = Vec::with_capacity(self.len());
        for row in 0..self.rows{
            let x = self.get_row_vector_slice(row);
            let y = rhs.get_row_vector_slice(0);

            let xplusy = x.iter().zip(y.iter()).map(|(a, b)| a + b).collect::<Vec<f64>>();

            values.extend(xplusy);
        }

        Self::from_vec(values, self.row_count(), self.column_count())
    }

    /// TODO: write in new _Matrix.
    pub fn add_row_partitioned(&self, rhs: &Matrix, partitioner: &Partitioner) -> Self {
        assert_eq!(rhs.rows, 1, "Rhs matrix must have 1 row.");
        assert_eq!(self.columns, rhs.columns, "Lhs and rhs must have equal number of columns.");

        let inner_process = move |partition: &Partition| {
            let mut partition_values: Vec<f64> = Vec::with_capacity(partition.get_size() * self.columns);
            let rs = rhs.get_row_vector_slice(0); 
            for row in partition.get_range() {
                let ls = self.get_row_vector_slice(row);

                for column in 0..ls.len() {
                    partition_values.push(ls[column] + rs[column]);
                }
            }

            partition_values
        };

        let values = partitioner.parallelized(inner_process);

        Self::from_vec(values, self.row_count(), self.column_count())
    }

    /// TODO: write in new _Matrix.
    /// Sums each row in self, and outputs a new matrix that is 1 row but same number of columns.
    pub fn shrink_rows_by_add(&self) -> Self {
        let t = self.get_transpose();
        let mut values = Vec::with_capacity(self.columns);
        for row in 0..t.rows {
            let x = t.get_row_vector_slice(row).iter().sum();
            values.push(x);
        }

        Self::from_vec(values, 1, self.column_count())
    }

    /// TODO: write in new _Matrix.
    /// Scales matrix by a scalar.
    /// Instead of making a division operator, please pass. reciprocal of scalar
    pub fn scale(&self, scalar: f64) -> Self {
        let values = self.values.iter().map(|&x| x * scalar).collect();

        Self::from_vec(values, self.row_count(), self.column_count())
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
            ],
            rowwise_partitioner: Partitioner::with_partitions(2, thread::available_parallelism().unwrap().get())
        };

        assert_eq!(actual, expected);
    }

    #[test]
    #[should_panic]
    fn test_invalid_row_slice() {
        let tc = Matrix {
            values: vec![],
            rows: 1,
            columns: 3,
            rowwise_partitioner: Partitioner::with_partitions(1, 1)
        };

        tc.get_row_vector_slice(1);
    }

    #[test]
    fn test_matrix_add() {
        let p = thread::available_parallelism().unwrap().get();

        let lhs = Matrix {
            rows: 3,
            columns: 3,
            values: vec![
                1f64, 2f64, 3f64,
                4f64, 5f64, 6f64,
                7f64, 8f64, 9f64
            ],
            rowwise_partitioner: Partitioner::with_partitions(3, p)

        };

        let rhs = Matrix {
            rows: 3,
            columns: 3,
            values: vec![
                -1f64, -2f64, -3f64,
                -4f64, -5f64, -6f64,
                -7f64, -8f64, -9f64
            ],
            rowwise_partitioner: Partitioner::with_partitions(3, p)
        };

        let actual = lhs.add(&rhs);

        let expected = Matrix {
            rows: 3,
            columns: 3,
            values: vec![
                0f64, 0f64, 0f64,
                0f64, 0f64, 0f64,
                0f64, 0f64, 0f64
            ],
            rowwise_partitioner: Partitioner::with_partitions(3, p)
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn transpose_test() {
        let p = thread::available_parallelism().unwrap().get();

        let m = Matrix {
            rows: 5,
            columns: 4,
            values: vec![
                0f64, 1f64, 2f64, 3f64,
                4f64, 5f64, 6f64, 7f64,
                8f64, 9f64, 10f64, 11f64,
                12f64, 13f64, 14f64, 15f64,
                16f64, 17f64, 18f64, 19f64
            ],
            rowwise_partitioner: Partitioner::with_partitions(5, p)
        };

        let expected = Matrix {
            rows: 4,
            columns: 5,
            values: vec![
                0f64, 4f64, 8f64, 12f64, 16f64,
                1f64, 5f64, 9f64, 13f64, 17f64,
                2f64, 6f64, 10f64, 14f64, 18f64,
                3f64, 7f64, 11f64, 15f64, 19f64
            ],
            rowwise_partitioner: Partitioner::with_partitions(4, p)
        };
        
        let actual = m.get_transpose();
        assert_eq!(actual, expected);
    }

    #[test]
    fn identity_1() {
        let p = thread::available_parallelism().unwrap().get();

        let actual = Matrix::new_identity(1);
        let expected = Matrix{
            rows: 1,
            columns: 1,
            values: vec![1.0f64],
            rowwise_partitioner: Partitioner::with_partitions(1, p)
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn identity_2() {
        let p = thread::available_parallelism().unwrap().get();

        let actual = Matrix::new_identity(2);
        let expected = Matrix {
            rows: 2,
            columns: 2,
            values: vec![
                1.0f64, 0.0f64,
                0.0f64, 1.0f64],
            rowwise_partitioner: Partitioner::with_partitions(2, p)
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn identity_3() {
        let p = thread::available_parallelism().unwrap().get();

        let actual = Matrix::new_identity(3);
        let expected = Matrix {
            rows: 3,
            columns: 3,
            values: vec![
                1.0f64, 0.0f64, 0.0f64, 
                0.0f64, 1.0f64, 0.0f64, 
                0.0f64, 0.0f64, 1.0f64],
                rowwise_partitioner: Partitioner::with_partitions(3, p)
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_random_matrix() {
        let m28x28 = Matrix::new_randomized_z(28, 28);

        let _r =m28x28.values.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ");
    }

     #[test]
    fn matrix_mul() {
        let p = thread::available_parallelism().unwrap().get();

        // Given
        let lhs = Matrix {
            rows: 4,
            columns: 3,
            values: vec![
                1f64, 2f64, 3f64,
                4f64, 5f64, 6f64,
                7f64, 8f64, 9f64,
                10f64, 11f64, 12f64],
                rowwise_partitioner: Partitioner::with_partitions(4, p)
        };

        let rhs = Matrix {
            rows: 3,
            columns: 5,
            values: vec![
                1f64, 2f64, 3f64, 4f64, 5f64,
                6f64, 7f64, 8f64, 9f64, 10f64,
                11f64, 12f64, 13f64, 14f64, 15f64
            ],
            rowwise_partitioner: Partitioner::with_partitions(3, p)
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
            ],
            rowwise_partitioner: Partitioner::with_partitions(4, p)
        };

        assert!(actual == expected);
    }

    #[test]
    fn test_threaded_mul_equals_regular_mul() {
        let lhs = Matrix::new_randomized_z(96, 784);
        let rhs = Matrix::new_randomized_z(128, 784);

        let expected = lhs.mul_with_transposed(&rhs);

        let partition_count = thread::available_parallelism().unwrap().get();
        let partitioner = Partitioner::with_partitions(lhs.rows, partition_count);
        let actual = lhs.mul_with_transposed_partitioned(&rhs, &partitioner);

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_scale() {
        let p = thread::available_parallelism().unwrap().get();

        let tc = Matrix {
            rows: 2,
            columns: 3,
            values: vec![
                1., 2., 3.,
                4., 5., 6.
            ],
            rowwise_partitioner: Partitioner::with_partitions(2, p)
        };

        let actual = tc.scale(3.);
        let expected = Matrix {
            rows: 2,
            columns: 3,
            values: vec![
                3., 6., 9.,
                12., 15., 18.
            ],
            rowwise_partitioner: Partitioner::with_partitions(2, p)
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_shrink_rows_by_add() {
        let p = thread::available_parallelism().unwrap().get();

        let tc = Matrix {
            columns: 3,
            rows: 3,
            values: vec![
                1., 1., 2.,
                2., 3., 3.,
                4., 4., 5.
            ],
            rowwise_partitioner: Partitioner::with_partitions(3, p)
        };

        let actual = tc.shrink_rows_by_add();
        let expected = Matrix {
            columns: 3,
            rows: 1,
            values: vec![
                7., 8., 10.
            ],
            rowwise_partitioner: Partitioner::with_partitions(1, p)
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_get_row_vector_slice() {
        let p = thread::available_parallelism().unwrap().get();

        let tc = Matrix {
            columns: 3,
            rows: 4,
            values: vec![
                1., 2., 3.,
                10., 20., 30.,
                100., 200., 300.,
                1000., 2000., 3000.
            ],
            rowwise_partitioner: Partitioner::with_partitions(4, p)
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

        let partition_count = thread::available_parallelism().unwrap().get();
        let partitioner = Partitioner::with_partitions(lhs.rows, partition_count);
        let actual = lhs.mul_element_wise_partitioned(&rhs, &partitioner);

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_add_row_vector() {
        let p = thread::available_parallelism().unwrap().get();

        let tc = Matrix {
            rows: 3,
            columns: 4,
            values: vec![
                0., 0., 0., 0.,
                1., 1., 1., 1.,
                2., 2., 2., 2.
            ],
            rowwise_partitioner: Partitioner::with_partitions(3, p)
        };

        let row_to_add = Matrix {
            rows: 1,
            columns: 4,
            values: vec![10., 20., 30., 40.],
            rowwise_partitioner: Partitioner::with_partitions(1, p)
        };

        let expected = Matrix {
            rows: 3,
            columns: 4,
            values: vec![
                10., 20., 30., 40.,
                11., 21., 31., 41.,
                12., 22., 32., 42.
            ],
            rowwise_partitioner: Partitioner::with_partitions(3, p)
        };

        let actual = tc.add_row_vector(&row_to_add);
        assert_eq!(actual, expected);        
    }

    #[test]
    fn test_add_row_partitioned() {
        let lhs = Matrix::new_randomized_z(500, 600);
        let rhs = &Matrix::new_randomized_z(1, 600);

        let partitioner = Partitioner::with_partitions(500, 16);
        let actual = lhs.add_row_partitioned(rhs, &partitioner);

        let expected = lhs.add_row_vector(rhs);

        assert_eq!(actual, expected);
    }
}
