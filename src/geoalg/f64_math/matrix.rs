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
    all_partitioner: Option<Partitioner>,
    row_partitioner: Option<Partitioner>
}

impl Matrix {
    /// Returns number of rows this matrix has.
    pub fn row_count(&self) -> usize { self.rows }

    /// Returns number of columns this matrix has.
    pub fn column_count(&self) -> usize { self.columns }

    /// Returns a slice of the values this matrix has.
    pub fn read_values(&self) -> &[f64] { &self.values }

    pub fn from(rows: usize, columns: usize, values: Vec<f64>) -> Self {
        Self {
            rows, columns, values,
            //rowwise_partitioner_old = None
            all_partitioner: None,
            row_partitioner: None
        }
    }

    /// Returns a row x column matrix filled with random values between -1.0 and 1.0 inclusive.
    pub fn new_randomized_z(rows: usize, columns: usize) -> Self {
        assert!(columns > 0);
        assert!(rows > 0);

        let mut rng = rand::thread_rng();
        let step = Uniform::new_inclusive(-1.0, 1.0);
        let element_count = columns * rows;
        let values = step.sample_iter(&mut rng).take(element_count).collect();

        Self::from(rows, columns, values)
    }

    /// Returns an rows x column matrix filled with random values specified by uniform distribution.
    pub fn new_randomized_uniform(rows: usize, columns: usize, uniform: Uniform<f64>) -> Self {
        assert!(columns > 0);
        assert!(rows > 0);

        let mut rng = rand::thread_rng();
        let element_count = columns * rows;
        let values = uniform.sample_iter(&mut rng).take(element_count).collect();
        
        Self::from(rows, columns, values)
    }

    /// Returns a contiguous slice of data representing columns in the matrix.
    pub fn get_row_vector_slice(&self, row: usize) -> &[f64] {
        assert!(row < self.rows, "Tried to get a row that was out of bounds.");

        let start = row * self.columns;
        let end = start + self.columns;
        &self.values[start..end]
    }

    /// Returns a newly allocated matrix that is the transpose of the matrix operated on.
    pub fn transpose_s(&self) -> Self {
        let capacity = self.values.len();
        let mut transposed = Vec::with_capacity(capacity);

        for i in 0..capacity {
            let index_to_push = self.columns * (i % self.rows) + i / self.rows;

            transposed.push(self.values[index_to_push]);
        }

        Self::from(self.columns, self.rows, transposed)
    }

    /// Returns transpose of matrix.
    pub fn transpose(&self) -> Self {
        let inner_process = move |partition: &Partition| {
            let mut partition_values = Vec::with_capacity(partition.get_size());
            for i in partition.get_range() {
                let index_to_read = self.columns * (i % self.rows) + i / self.rows;
                let value = self.values[index_to_read];                        
                partition_values.push(value);
            }

            partition_values
        };

        let partition_strategy = match self.all_partitioner.as_ref() {
            Some(p) => p,
            None => {
                &Partitioner::with_partitions(self.len(), thread::available_parallelism().unwrap().get())
            }
        };

        let values = partition_strategy.parallelized(inner_process);
        Self::from(self.columns, self.rows, values)
    }

    /// Element-wise multiplication.
    pub fn elementwise_multiply_s(&self, rhs: &Matrix) -> Self {
        assert_eq!(self.columns, rhs.columns, "Columns must match for elementwise multiply.");
        assert_eq!(self.rows, rhs.rows, "Rows must match for elementwise multiply.");

        let values = self.values
            .iter()
            .zip(rhs.values.iter())
            .map(|(&x, &y)| x * y)
            .collect();

        Self::from(self.rows, self.columns, values)
    }

    /// Makes use of supplied partitions to parallelize the operation.
    /// If partitions is cached, can be reused (to hopefully save even more time).
    pub fn mul_element_wise_partitioned(&self, rhs: &Matrix) -> Self {
        let inner_process = |partition: &Partition| {
            // We know in advance the exact capacity of result.
            // It will always be partition size * number of inner rows.
            let mut partition_values: Vec<f64> = Vec::with_capacity(partition.get_size());
            for i in partition.get_range() {
                partition_values.push(self.values[i] * rhs.values[i]);
            }

            partition_values
        };

        let partition_strategy = match self.all_partitioner.as_ref() {
            Some(p) => p,
            None => {
                &Partitioner::with_partitions(self.len(), thread::available_parallelism().unwrap().get())
            }
        };

        let values = partition_strategy.parallelized(inner_process);

        Self::from(self.row_count(), self.column_count(), values)
    }

    /// Multiplies two matrices using transpose operation for efficiency.
    /// Willing to take performance hit [O(m x n)] for small matrices for creating transposed matrix. 
    pub fn mul(&self, rhs: &Matrix) -> Self {
        assert_eq!(self.columns, rhs.rows, "When multiplying matrices, lhs columns must equal rhs rows.");
        let transposed = rhs.transpose_s();
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

        Self::from(self.row_count(), rhs.row_count(), values)
    }

    /// Computes matrix multiplication and divying up work amongst partitions.
    /// Partitioner implementation complete.
    pub fn mul_with_transposed_partitioned(&self, rhs: &Matrix) -> Matrix {
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

        let partition_strategy = match self.row_partitioner.as_ref() {
            Some(p) => p,
            None => {
                &Partitioner::with_partitions(self.rows, thread::available_parallelism().unwrap().get())
            }
        };

        let values = partition_strategy.parallelized(inner_process);
        Self::from(self.rows, rhs.rows, values)
    }

    /// Returns size of underlying vector.
    pub fn len(&self) -> usize {
        self.columns * self.rows
    }

    /// Useful for applying an activation function to the entire matrix.
    /// *Allows to map capturing a variable outside of the closure.
    pub fn map_with_capture(&self, func: impl Fn(&f64) -> f64) -> Self {
        let values = self.values.iter().map(|val| func(val)).collect();
        
        Self::from(self.row_count(), self.column_count(), values)
    }

    /// Useful for applying an activation function to the entire matrix.
    pub fn map(&self, func: fn(&f64) -> f64) -> Self {
        let values = self.values.iter().map(|&val| func(&val)).collect();
        
        Self::from(self.row_count(), self.column_count(), values)
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

        Self::from(self.row_count(), self.column_count(), values)
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

        Self::from(self.row_count(), self.column_count(), values)
    }

    /// Adds a row to each row in matrix.
    /// Partitioner implementation complete.
    pub fn add_row_partitioned(&self, rhs: &Matrix) -> Self {
        assert_eq!(rhs.rows, 1, "Rhs matrix must have 1 row.");
        assert_eq!(self.columns, rhs.columns, "Lhs and rhs must have equal number of columns.");

        let inner_process = move |partition: &Partition| {
            let mut partition_values= Vec::with_capacity(partition.get_size() * self.columns);
            let rs = rhs.get_row_vector_slice(0); 
            for row in partition.get_range() {
                let ls = self.get_row_vector_slice(row);

                for column in 0..ls.len() {
                    partition_values.push(ls[column] + rs[column]);
                }
            }

            partition_values
        };

        let partition_strategy = match self.row_partitioner.as_ref() {
            Some(p) => p,
            None => {
                &Partitioner::with_partitions(self.rows, thread::available_parallelism().unwrap().get())
            }
        };

        let values = partition_strategy.parallelized(inner_process);
        Self::from(self.row_count(), self.column_count(), values)
    }

    /// TODO: write in new _Matrix.
    /// Sums each row in self, and outputs a new matrix that is 1 row but same number of columns.
    pub fn shrink_rows_by_add(&self) -> Self {
        let t = self.transpose_s();
        let mut values = Vec::with_capacity(self.columns);
        for row in 0..t.rows {
            let x = t.get_row_vector_slice(row).iter().sum();
            values.push(x);
        }

        Self::from(1, self.column_count(), values)
    }

    /// Scales matrix by a scalar.
    /// Instead of making a division operator, please pass in reciprocal of scalar.
    /// Partitioner implementation complete.
    pub fn scale(&self, scalar: f64) -> Self {
        let inner_process = move |partition: &Partition| {
            let mut partition_values = Vec::with_capacity(partition.get_size());
            for i in partition.get_range() {
                partition_values.push(self.values[i] * scalar);
            }

            partition_values
        };

        let partition_strategy = match self.all_partitioner.as_ref() {
            Some(p) => p,
            None => {
                &Partitioner::with_partitions(self.len(), thread::available_parallelism().unwrap().get())
            }
        };

        let values = partition_strategy.parallelized(inner_process);
        Self::from(self.row_count(), self.column_count(), values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_vec() {
        let actual = Matrix {
            values: vec![
                1., 2., 3.,
                4., 5., 6.
                ], 
            rows: 2,
            columns: 3,
            all_partitioner: None,
            row_partitioner: None};

        let expected = Matrix::from(
            2,
            3,
            vec![
                1., 2., 3.,
                4., 5., 6.
            ]);

        assert_eq!(actual, expected);
    }

    #[test]
    #[should_panic]
    fn test_invalid_row_slice() {
        let tc = Matrix::from(1, 3, vec![]);

        tc.get_row_vector_slice(1);
    }

    #[test]
    fn transpose_test() {
        let m = Matrix::from(5, 4, vec![
                0f64, 1f64, 2f64, 3f64,
                4f64, 5f64, 6f64, 7f64,
                8f64, 9f64, 10f64, 11f64,
                12f64, 13f64, 14f64, 15f64,
                16f64, 17f64, 18f64, 19f64
            ]);

        let expected = Matrix::from(4, 5, vec![
                0f64, 4f64, 8f64, 12f64, 16f64,
                1f64, 5f64, 9f64, 13f64, 17f64,
                2f64, 6f64, 10f64, 14f64, 18f64,
                3f64, 7f64, 11f64, 15f64, 19f64
            ]);
        
        let actual = m.transpose_s();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_random_matrix() {
        let m28x28 = Matrix::new_randomized_z(28, 28);

        let _r =m28x28.values.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ");
    }

     #[test]
    fn matrix_mul() {
        // Given
        let lhs = Matrix::from(4, 3, vec![
                1f64, 2f64, 3f64,
                4f64, 5f64, 6f64,
                7f64, 8f64, 9f64,
                10f64, 11f64, 12f64]);

        let rhs = Matrix::from(3, 5, vec![
                1f64, 2f64, 3f64, 4f64, 5f64,
                6f64, 7f64, 8f64, 9f64, 10f64,
                11f64, 12f64, 13f64, 14f64, 15f64
            ]);

        let actual = Matrix::mul(&lhs, &rhs);

        // Resultant matrix needs to have aas many rows as lhs, and as many columns as rhs.
        let expected = Matrix::from(4, 5, vec! [
                46f64, 52f64, 58f64, 64f64, 70f64,
                100f64, 115f64, 130f64, 145f64, 160f64,
                154f64, 178f64, 202f64, 226f64, 250f64,
                208f64, 241f64, 274f64, 307f64, 340f64
            ]);

        assert!(actual == expected);
    }

    #[test]
    fn test_threaded_mul_equals_regular_mul() {
        let lhs = Matrix::new_randomized_z(96, 784);
        let rhs = Matrix::new_randomized_z(128, 784);

        let expected = lhs.mul_with_transposed(&rhs);

        let actual = lhs.mul_with_transposed_partitioned(&rhs);

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_scale() {
        let tc = Matrix::from(2, 3, vec![
                1., 2., 3.,
                4., 5., 6.
            ]);

        let actual = tc.scale(3.);
        let expected = Matrix::from(2, 3, vec![
                3., 6., 9.,
                12., 15., 18.
            ]);

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_shrink_rows_by_add() {
        let tc = Matrix::from(3, 3, vec![
                1., 1., 2.,
                2., 3., 3.,
                4., 4., 5.
            ]);

        let actual = tc.shrink_rows_by_add();
        let expected = Matrix::from(1, 3, vec![
                7., 8., 10.
            ]);

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_get_row_vector_slice() {
        let tc = Matrix::from(4, 3, vec![
                1., 2., 3.,
                10., 20., 30.,
                100., 200., 300.,
                1000., 2000., 3000.
            ]);

        let actual = tc.get_row_vector_slice(2);
        let expected = &[100., 200., 300.];

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_element_wise_multiply_threaded() {
        let lhs = Matrix::new_randomized_z(1000, 3001);
        let rhs = Matrix::new_randomized_z(1000, 3001);

        let expected = lhs.elementwise_multiply_s(&rhs);

        let actual = lhs.mul_element_wise_partitioned(&rhs);

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_add_row_vector() {
        let tc = Matrix::from(3, 4, vec![
                0., 0., 0., 0.,
                1., 1., 1., 1.,
                2., 2., 2., 2.
            ]);

        let row_to_add = Matrix::from(1, 4, vec![10., 20., 30., 40.]);

        let expected = Matrix::from(3, 4, vec![
                10., 20., 30., 40.,
                11., 21., 31., 41.,
                12., 22., 32., 42.
            ]);

        let actual = tc.add_row_vector(&row_to_add);
        assert_eq!(actual, expected);        
    }

    #[test]
    fn test_add_row_partitioned() {
        let lhs = Matrix::new_randomized_z(500, 600);
        let rhs = &Matrix::new_randomized_z(1, 600);

        let actual = lhs.add_row_partitioned(rhs);

        let expected = lhs.add_row_vector(rhs);

        assert_eq!(actual, expected);
    }
}
