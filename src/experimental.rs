use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;

use rand::distributions::Uniform;
use rand::prelude::Distribution;

use crate::geoalg::f64_math::optimized_functions::dot_product_of_vector_slices;
use crate::partitioner_cache;
use crate::partitions::{Partition, Partitioner};

#[repr(align(64))]
struct PaddedAtomicU64 {
    inner: AtomicU64
}

impl PaddedAtomicU64 {
    pub fn new() -> Self {
        Self { inner: AtomicU64::new(0) }
    }

    pub fn from_float(value: f64) -> Self {
        let inner = AtomicU64::new(value.to_bits());
        Self { inner }
    }

    pub fn come_in(&self, value: f64) {
        self.inner.store(value.to_bits(), Ordering::Relaxed);
    }

    pub fn go_out(&self) -> f64 {
        f64::from_bits(self.inner.load(Ordering::Relaxed))        
    }
}

/// Calculates dot product of two slices of AtomicU64 as if they were floats.
/// This portion not parallelized, outer portions meant to be parallelized.
/// Might be worthwhile in a work stealing paradigm? Maybe?
fn dot_product_atomic(lhs: &[PaddedAtomicU64], rhs: &[PaddedAtomicU64]) -> f64 {
    let mut sum = 0.;
    for i in 0..lhs.len() {
        sum += lhs[i].go_out() * 
            rhs[i].go_out()
    }

    sum
}

/// Helper function to create a contiguous section of memory.
/// Allocates the place where matrix operation result are stored when done in parallel. 
fn contiguous(rows: usize, columns: usize) -> Arc<Vec<PaddedAtomicU64>> {
    let size = rows * columns;
    let x = (0..size).map(|_| PaddedAtomicU64::new()).collect::<Vec<_>>();
    Arc::new(x)
}

fn to_vec(a_values: &Arc<Vec<PaddedAtomicU64>>) -> Vec<f64> {
    let size = a_values.len();
    let mut f64s = Vec::with_capacity(size);

    for i in 0..size {
        let x = a_values[i].go_out();
        f64s.push(x);
    }

    f64s
}

#[derive(Clone)]
pub struct Matrix {
    rows: usize,
    columns: usize,
    a_values: Arc<Vec<PaddedAtomicU64>>,
    values: Vec<f64>,
    row_partitioner: Option<Partitioner>,
    all_partitioner: Option<Partitioner>
}

impl Matrix {
    pub fn partition_by_rows(mut self) -> Self {
        self.row_partitioner = Some(Partitioner::with_partitions(self.row_count(), thread::available_parallelism().unwrap().get()));
        self
    }

    /// Returns number of rows.
    pub fn row_count(&self) -> usize { self.rows }

    /// Returns number of columns.
    pub fn column_count(&self) -> usize { self.columns }

    pub fn len(&self) -> usize { self.rows * self.columns }

    /// Creates a new Matrix by taking over ownership of an Arc<Vec<AtomicU64>>.
    /// Private for now, only needed internally.
    fn new(rows: usize, columns: usize, a_values: Arc<Vec<PaddedAtomicU64>>) -> Self {
        //let partitioner = Partitioner::none();
        let values = to_vec(&a_values);
        Self { rows, columns, a_values, values, 
            row_partitioner: None,
            all_partitioner: None }
    }

    /// Returns a row x column matrix filled with random values between -1.0 and 1.0 inclusive.
    pub fn new_randomized_uniform(rows: usize, columns: usize, uniform: Uniform<f64>) -> Self {
        assert!(columns > 0);
        assert!(rows > 0);

        let mut rng = rand::thread_rng();
        let element_count = columns * rows;
        let values = uniform.sample_iter(&mut rng).take(element_count).collect();

        Self::from(rows, columns, values)
    }

    /// Constructs Matrix from supplied Vec<f64>.
    pub fn from(rows: usize, columns: usize, values: Vec<f64>) -> Self {
        let a_values = values
            .clone()
            .into_iter()
            .map(|x| PaddedAtomicU64::from_float(x))
            .collect::<Vec<PaddedAtomicU64>>();
        
        let a_values = Arc::new(a_values);
        //let partitioner = Partitioner::none();

        Self { rows, columns, a_values, values, 
            row_partitioner: None,
            all_partitioner: None }
    }

    /// Returns internal values as a Vec<f64>.
    /// Private for now.
    pub fn to_vec(&self) -> Vec<f64> {
        to_vec(&self.a_values)
    }

    /// Returns a contiguous slice of data representing a vector of columns in the matrix.
    /// Note that they are AtomicU64, so need to_bits and from_bits operations when using.
    fn atomic_row(&self, row: usize) -> &[PaddedAtomicU64] {
        assert!(row < self.rows, "Tried to get a row that was out of bounds from a matrix.");

        let start = row * self.columns;
        let end = start + self.columns;
        &self.a_values[start..end]
    }

    /// Returns row as vector of f64s.
    pub fn row(&self, row: usize) -> Vec<f64> {
        let x = self.atomic_row(row);

        let res = x.iter().map(|a| a.go_out()).collect::<Vec<_>>();
        res
    }

    /// Performs a function on every element in Matrix.
    /// # Partition by self.len().
    pub fn map_atomic(&self, function: fn(&f64) -> f64, partitioner: &Partitioner) -> Self {
        let values = contiguous(self.rows, self.columns);
        thread::scope(|s| {
            for partition in &partitioner.partitions {
                let partitioned_values = Arc::clone(&values);
                let _handle = s.spawn(move || {
                    for i in partition.get_range() {
                        let cursor = i;
                        let read_value = self.a_values[cursor].go_out();
                        let value = function(&read_value);
                        partitioned_values[cursor].come_in(value);
                    }
                });
            }
        });

        Self::new(self.rows, self.columns, values)
    }

    pub fn map(&self, function: fn(&f64) -> f64) -> Self {
        let inner_process = move |partition: &Partition| {
            let mut partitioned_values = Vec::with_capacity(partition.get_size());
            for i in partition.get_range() {
                let original = self.values[i];
                let value = function(&original);
                partitioned_values.push(value);
            }

            partitioned_values
        };

        // Self::new(self.rows, self.columns, values)
        // Strategy for calculating the multiplication
        match self.all_partitioner.as_ref() {
            Some(p) => {
                let values = p.parallelized(inner_process);
                return Self::from(self.rows, self.rows, values);
            },
            None => {
                let all_partitioner =  Partitioner::with_partitions(self.len(), thread::available_parallelism().unwrap().get());
                let values = all_partitioner.parallelized(inner_process);
                let result = Self::from(self.rows, self.columns, values);
                return result;
            }
        }
    }

    /// Returns transpose of Matrix.
    /// # Partition by Matrix.len().
    /// Does not modify original Matrix.
    pub fn transpose(&self, partitioner: &Partitioner) -> Self {
        let values = contiguous(self.columns, self.rows);
        thread::scope(|s| {
            for partition in &partitioner.partitions {
                let partitioned_values = Arc::clone(&values);
                let _handle = s.spawn(move || {
                    for i in partition.get_range() {
                        let cursor = i;
                        let index_to_read = self.columns * (cursor % self.rows) + cursor / self.rows;
                        let value = self.a_values[index_to_read].go_out();                        
                        partitioned_values[cursor].come_in(value);
                    }
                });
            }
        });

        Self::new(self.columns, self.rows, values)
    }

    /// Returns element-wise product.
    /// # Partition by self.row_count().
    pub fn hadamard(&self, rhs: &Self, partitioner: &Partitioner) -> Self {
        let values = contiguous(self.rows, self.columns);
        thread::scope(|s| {
            for partition in &partitioner.partitions {
                let partitioned_values = Arc::clone(&values);
                let _handle = s.spawn(move || {
                    for row_index in partition.get_range() {
                        let cursor = row_index * self.columns;
                        let ls = self.atomic_row(row_index);
                        let rs = rhs.atomic_row(row_index);
                        for column_index in 0..self.columns {
                            let x = ls[column_index].go_out();
                            let y = rs[column_index].go_out();
                            let value = x * y;
                            partitioned_values[cursor + column_index].come_in(value);
                        }
                    }
                });
            }
        });

        Self::new(self.rows, self.columns, values)
    }

    pub fn mul_by_transpose(&self, rhs: &Self) -> Self {
        let inner_process = move |partition: &Partition| {
            let mut partition_values: Vec<f64> = Vec::with_capacity(partition.get_size() * rhs.rows);
            for row in partition.get_range() {
                let ls = self.row(row);
                for transposed_row in 0..rhs.rows {
                    let rs = rhs.row(transposed_row);
                    let dot_product = dot_product_of_vector_slices(&ls, &rs);
                    partition_values.push(dot_product);
                }
            }

            partition_values
        };

        // Strategy for calculating the multiplication
        match self.row_partitioner.as_ref() {
            Some(p) => {
                let values = p.parallelized(inner_process);
                return Self::from(self.rows, rhs.rows, values);
            },
            None => {
                let row_partitioner =  Partitioner::with_partitions(self.row_count(), thread::available_parallelism().unwrap().get());
                let values = row_partitioner.parallelized(inner_process);
                let result = Self::from(self.rows, rhs.rows, values);
                return result;
            }
        }
    }

    /// Subtracts right hand side from left hand side.
    /// Partition by self.row_count().
    pub fn sub(&self, rhs: &Self, partitioner: &Partitioner) -> Self {
        assert!(self.columns == rhs.columns && self.rows == rhs.rows, "Cannot subtract matrices with different orders.");

        let values = contiguous(self.rows, self.columns);
        thread::scope(|s| {
            for partition in &partitioner.partitions {
                let partitioned_values = Arc::clone(&values);
                let _handle = s.spawn(move || {
                    for row_index in partition.get_range() {
                        for column_index in 0..self.column_count() {
                            let cursor = row_index * self.column_count() + column_index; 
                            let l = self.a_values[cursor].go_out();
                            let r = rhs.a_values[cursor].go_out();
                            partitioned_values[cursor].come_in(l - r);
                        }
                    } 
                });
            }
        });

        Self::new(self.rows, self.columns, values)
    }

    /// Scales matrix elements by specified scalar.
    pub fn scale(&self, scalar: f64) -> Self {
        let values = self.to_vec().iter().map(|f| f * scalar).collect();
        Self::from(self.rows, self.columns, values)
    }

    /// Returns a 1 row matrix where the column values for each row are summed.
    /// # Partition by self.len()
    pub fn shrink_rows_by_add(&self, partitioner: &Partitioner) -> Self {
        let t = self.transpose(partitioner);

        let mut values = Vec::with_capacity(self.columns);
        for row in 0..t.row_count() {
            let x = t.row(row).iter().sum();
            values.push(x);
        }

        Self::from(1, self.column_count(), values)
    }

    /// Adds a row to each row in the matrix
    /// # Partion by self.row_count().
    pub fn add_broadcasted(&self, rhs: &Self, partitioner: &Partitioner) -> Self {
        let values = contiguous(self.rows, self.columns);
        thread::scope(|s| {
            for partition in &partitioner.partitions {
                let partitioned_values = Arc::clone(&values);
                let rs = rhs.atomic_row(0);
                let _handle = s.spawn(move || {
                    for row in partition.get_range() {
                        let cursor = row * self.columns;
                        let ls = self.atomic_row(row);
                        for column in 0..ls.len() {
                            let x = ls[column].go_out();
                            let y = rs[column].go_out();

                            partitioned_values[cursor + column].come_in(x + y);
                        }
                    }
                });
            }
        });

        Self::new(self.rows, self.columns, values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_product_atomic() {

    }

    #[test]
    fn test_add_broadcasted() {
        let p = thread::available_parallelism().unwrap().get();

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

        let partitioner = &Partitioner::with_partitions(tc.row_count(), p);

        let actual = tc.add_broadcasted(&row_to_add, partitioner);
        assert_eq!(actual.to_vec(), expected.to_vec());        
    }

    #[test]
    fn test_sub() {
        let lhs = Matrix::from(3, 3, vec![
            10., 20., 30.,
            10., 20., 30.,
            10., 20., 30.
        ]);

        let rhs = Matrix::from(3, 3, vec![
            1., 1., 1.,
            2., 2., 2.,
            3., 3., 3.
        ]);

        let expected = Matrix::from(3, 3, vec![
            9., 19., 29.,
            8., 18., 28.,
            7., 17., 27.
        ]);

        let partition_count = thread::available_parallelism().unwrap().get();
        let partitioner = Partitioner::with_partitions(lhs.rows, partition_count);
        let actual = lhs.sub(&rhs, &partitioner);

        println!("exptect: {:?}, actual: {:?}", expected.to_vec(), expected.to_vec());

        assert_eq!(actual.to_vec(), expected.to_vec());
    }

    #[test]
    fn test_shrink_rows_by_add() {
        let p = thread::available_parallelism().unwrap().get();

        let tc = Matrix::from(3, 3, vec![
            1., 1., 2.,
            2., 3., 3.,
            4., 4., 5.
        ]);

        let partitioner = &Partitioner::with_partitions(tc.len(), p);

        let actual = tc.shrink_rows_by_add(partitioner);
        let expected = Matrix::from(1, 3, vec![
            7., 8., 10.
        ]);

        assert_eq!(actual.to_vec(), expected.to_vec());
    }

    #[test]
    fn test_scale() {
        let p = thread::available_parallelism().unwrap().get();

        let tc = Matrix::from(2, 3, vec![
            1., 2., 3.,
            4., 5., 6.
        ]);

        let actual = tc.scale(3.);
        let expected = Matrix::from(2, 3, vec![
            3., 6., 9.,
            12., 15., 18.
        ]);

        assert_eq!(actual.to_vec(), expected.to_vec()); 
    }

    #[test]
    fn test_map() {
        let ls = Matrix::from(2, 3, vec![
            1., 1., 1.,
            2., 2., 2.
        ]);

        let partitioner = Partitioner::with_partitions(6, 2);
        let actual = ls.map_atomic(|&x| -x, &partitioner);

        let expected = Matrix::from(2, 3, vec![
            -1., -1., -1.,
            -2., -2., -2.
        ]);

        assert_eq!(actual.to_vec(), expected.to_vec()); 
    }

    #[test]
    fn test_transpose() {
        let tc1 = Matrix::from(3, 4, vec![
            1., 2., 3., 4.,
            5., 6., 7., 8.,
            9., 10., 11., 12.
        ]);

        let expected = Matrix::from(4, 3, vec![
            1., 5., 9.,
            2., 6., 10.,
            3., 7., 11.,
            4., 8., 12.
        ]);

        let partitioner = &Partitioner::with_partitions(12, 2);
        let actual = tc1.transpose(partitioner);
        assert_eq!(actual.to_vec(), expected.to_vec());
    }

    #[test]
    fn test_hadamard() {
        let lhs = Matrix::from(3, 3, vec![
            1., 1., 1.,
            2., 2., 2.,
            3., 3., 3.
        ]);

        let rhs = Matrix::from(3, 3, vec![
            10., 20., 30.,
            10., 20., 30.,
            10., 20., 30.
        ]);

        let expected = Matrix::from(3, 3, vec![
            10., 20., 30.,
            20., 40., 60.,
            30., 60., 90.
        ]);

        let partition_count = thread::available_parallelism().unwrap().get();
        let partitioner = Partitioner::with_partitions(lhs.rows, partition_count);
        let actual = lhs.hadamard(&rhs, &partitioner);

        assert_eq!(actual.to_vec(), expected.to_vec());
    }

    #[test]
    fn test_mul_by_transpose() {
        let lhs = Matrix::from(4, 3,  vec![
            1., 2., 3.,
            4., 5., 6.,
            7., 8., 9.,
            10., 11., 12.
        ]);

        // Assume already transposed.
        let rhs = &Matrix::from(5, 3, vec![
            1., 6., 11.,
            2., 7., 12.,
            3., 8., 13.,
            4., 9., 14.,
            5., 10., 15.
        ]);

        let expected = Matrix::from(4, 5, vec! [
            46., 52., 58., 64., 70.,
            100., 115., 130., 145., 160.,
            154., 178., 202., 226., 250.,
            208., 241., 274., 307., 340.
        ]);

        //let partitioner = &Partitioner::with_partitions(lhs.rows, 2); 
        //let rows = lhs.row_count();
        let actual = lhs.mul_by_transpose(rhs);

        assert_eq!(actual.to_vec(), expected.to_vec());
    }
}