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
    row_partitioner: Option<Partitioner>,
    column_partitioner: Option<Partitioner>
}

impl Matrix {
    /// Returns size of underlying vector.
    pub fn len(&self) -> usize { self.values.len() }

    /// Returns number of rows this matrix has.
    pub fn row_count(&self) -> usize { self.rows }

    /// Returns number of columns this matrix has.
    pub fn column_count(&self) -> usize { self.columns }

    /// Returns a slice of the values this matrix has.
    pub fn read_values(&self) -> &[f64] { &self.values }

    /// Returns a new Matrix.
    pub fn from(rows: usize, columns: usize, values: Vec<f64>) -> Self {
        Self {
            rows, columns, values,
            all_partitioner: None,
            row_partitioner: None,
            column_partitioner: None
        }
    }

    /// Returns a contiguous slice of data representing columns in the matrix.
    pub fn row(&self, row_index: usize) -> &[f64] {
        assert!(row_index < self.rows, "Tried to get a row that was out of bounds.");

        let start = row_index * self.columns;
        let end = start + self.columns;
        &self.values[start..end]
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

    /// Returns transpose of matrix.
    /// Partitioner implementation complete.
    pub fn transpose(&self) -> Self {
        if self.rows == 1 || self.columns == 1 {
            return Self::from(self.columns, self.rows, self.values.clone());
        }

        let partition_strategy = match self.all_partitioner.as_ref() {
            Some(p) => p,
            None => {
                &Partitioner::with_partitions(self.len(), thread::available_parallelism().unwrap().get())
            }
        };

        let inner_process = move |partition: &Partition| {
            let mut partition_values = Vec::with_capacity(partition.get_size());
            for i in partition.get_range() {
                let index_to_read = self.columns * (i % self.rows) + i / self.rows;
                let value = self.values[index_to_read];                        
                partition_values.push(value);
            }

            partition_values
        };

        let values = partition_strategy.parallelized(inner_process);
        Self::from(self.columns, self.rows, values)
    }

    /// Makes use of supplied partitions to parallelize the operation.
    /// If partitions is cached, can be reused (to hopefully save even more time).
    /// Partitioner implementation complete.
    pub fn mul_element_wise(&self, rhs: &Matrix) -> Self {
        assert!(self.rows == rhs.rows && self.columns == rhs.columns, "When element-wise multiplying two matrices, they must have same order.");
        
        let partition_strategy = match self.all_partitioner.as_ref() {
            Some(p) => p,
            None => {
                &Partitioner::with_partitions(self.len(), thread::available_parallelism().unwrap().get())
            }
        };

        let inner_process = |partition: &Partition| {
            // We know in advance the exact capacity of result.
            // It will always be partition size * number of inner rows.
            let mut partition_values: Vec<f64> = Vec::with_capacity(partition.get_size());
            for i in partition.get_range() {
                partition_values.push(self.values[i] * rhs.values[i]);
            }

            partition_values
        };

        let values = partition_strategy.parallelized(inner_process);

        Self::from(self.row_count(), self.column_count(), values)
    }

    /// Computes matrix multiplication and divying up work amongst partitions.
    /// Faster multiplcation when you need to multiply the transposed matrix of rhs.
    /// Avoids calculating the transpose twice.
    /// Partitioner implementation complete.
    pub fn mul_with_transpose(&self, rhs: &Matrix) -> Matrix {
        assert_eq!(self.columns, rhs.columns, "When multiplying with transposed, columns must be equal for lhs and rhs.");

        let partition_strategy = match self.row_partitioner.as_ref() {
            Some(p) => p,
            None => {
                &Partitioner::with_partitions(self.rows, thread::available_parallelism().unwrap().get())
            }
        };

        let inner_process = move |partition: &Partition| {
            // We know in advance the exact capacity of result.
            // It will always be partition size * number of inner rows.
            let mut partition_values: Vec<f64> = Vec::with_capacity(partition.get_size() * rhs.rows);

            // This is start of the main logic of the code we care about
            for row in partition.get_range() {
                let ls = self.row(row);
                for transposed_row in 0..rhs.rows {
                    let rs = rhs.row(transposed_row);
                    let dot_product = dot_product_of_vector_slices(&ls, &rs);
                    partition_values.push(dot_product);
                }
            }
            // End of the main code we care abour

            partition_values
        };

        let values = partition_strategy.parallelized(inner_process);
        Self::from(self.rows, rhs.rows, values)
    }

    /// Useful for applying an activation function to the entire matrix.
    pub fn map(&self, func: fn(&f64) -> f64) -> Self {
        let values = self.values.iter().map(|&val| func(&val)).collect();
        
        Self::from(self.row_count(), self.column_count(), values)
    }

    /// Subtracts rhs Matrix from lhs Matrix.
    /// Partitioner implementation complete.
    pub fn sub(&self, rhs: &Matrix) -> Self {
       let partition_strategy = match self.all_partitioner.as_ref() {
            Some(p) => p,
            None => {
                &Partitioner::with_partitions(self.len(), thread::available_parallelism().unwrap().get())
            }
        };

        let inner_process = move |partition: &Partition| {
            let mut partition_values = Vec::with_capacity(partition.get_size());
            for i in partition.get_range() {
                let ls = self.values[i];
                let rs = rhs.values[i];
                partition_values.push(ls - rs);
            }

            partition_values
        };

        let values = partition_strategy.parallelized(inner_process);
        Self::from(self.row_count(), self.column_count(), values)
    }

    /// Adds a row to each row in matrix.
    /// Partitioner implementation complete.
    pub fn add_row_partitioned(&self, rhs: &Matrix) -> Self {
        assert_eq!(rhs.rows, 1, "Rhs matrix must have 1 row.");
        assert_eq!(self.columns, rhs.columns, "Lhs and rhs must have equal number of columns.");

        let partition_strategy = match self.row_partitioner.as_ref() {
            Some(p) => p,
            None => {
                &Partitioner::with_partitions(self.rows, thread::available_parallelism().unwrap().get())
            }
        };

        let inner_process = move |partition: &Partition| {
            let mut partition_values= Vec::with_capacity(partition.get_size() * self.columns);
            let rs = rhs.row(0); 
            for row in partition.get_range() {
                let ls = self.row(row);

                for column in 0..ls.len() {
                    partition_values.push(ls[column] + rs[column]);
                }
            }

            partition_values
        };

        let values = partition_strategy.parallelized(inner_process);
        Self::from(self.row_count(), self.column_count(), values)
    }

    /// Returns a 1 row matrix where each column is the sum of all values for that column.
    /// Partitioner implementation complete.
    pub fn reduce_rows_by_add(&self) -> Self {
        let partition_strategy = match self.column_partitioner.as_ref() {
            Some(p) => p,
            None => {
                &Partitioner::with_partitions(self.columns, thread::available_parallelism().unwrap().get())
            }
        };

        let inner_process =move |partition: &Partition| {
            let mut partition_values = Vec::with_capacity(partition.get_size());
            for column in partition.get_range() {
                let mut accumulator = 0.;
                for row in 0..self.rows {
                    accumulator += self.values[column + row * self.columns];
                }

                partition_values.push(accumulator);
            }

            partition_values
        };

        let values = partition_strategy.parallelized(inner_process);
        Self::from(1, self.columns, values)
    }

    /// Scales matrix by a scalar.
    /// Instead of making a division operator, please pass in reciprocal of scalar.
    /// Partitioner implementation complete.
    pub fn scale(&self, scalar: f64) -> Self {
        let partition_strategy = match self.all_partitioner.as_ref() {
            Some(p) => p,
            None => {
                &Partitioner::with_partitions(self.len(), thread::available_parallelism().unwrap().get())
            }
        };

        let inner_process = move |partition: &Partition| {
            let mut partition_values = Vec::with_capacity(partition.get_size());
            for i in partition.get_range() {
                partition_values.push(self.values[i] * scalar);
            }

            partition_values
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
            row_partitioner: None,
            column_partitioner: None};

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

        tc.row(1);
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
        
        let actual = m.transpose();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_random_matrix() {
        let m28x28 = Matrix::new_randomized_z(28, 28);

        let _r =m28x28.values.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ");
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
        let tc = Matrix::from(3, 4, vec![
                1., 1., 2., 10.,
                2., 3., 3., 10.,
                4., 4., 5., 10.
            ]);

        let expected = Matrix::from(1, 4, vec![
                7., 8., 10., 30.
            ]);

        let actual2 = tc.reduce_rows_by_add();
        assert_eq!(actual2, expected);
    }

    #[test]
    fn test_get_row_vector_slice() {
        let tc = Matrix::from(4, 3, vec![
                1., 2., 3.,
                10., 20., 30.,
                100., 200., 300.,
                1000., 2000., 3000.
            ]);

        let actual = tc.row(2);
        let expected = &[100., 200., 300.];

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

        let actual = tc.add_row_partitioned(&row_to_add);
        assert_eq!(actual, expected);        
    }
}
