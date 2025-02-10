use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::thread;

use rand::distributions::Uniform;
use rand::prelude::Distribution;

use crate::partitions::Partitioner;

/// Calculates dot product of two slices of AtomicU32 as if they were floats.
/// This portion not parallelized, outer portions meant to be parallelized.
/// Might be worthwhile in a work stealing paradigm? Maybe?
fn dot_product_atomic(lhs: &[AtomicU32], rhs: &[AtomicU32]) -> f32 {
    let mut sum = 0.;
    for i in 0..lhs.len() {
        sum += f32::from_bits(lhs[i].load(Ordering::Relaxed)) * 
            f32::from_bits(rhs[i].load(Ordering::Relaxed));
    }

    sum
}

/// Helper function to create a contiguous section of memory.
/// Allocates the place where matrix operation result are stored when done in parallel. 
fn contiguous(rows: usize, columns: usize) -> Arc<Vec<AtomicU32>> {
    let size = rows * columns;
    Arc::new(
        (0..size)
        .map(|_| AtomicU32::new(0))
        .collect::<Vec<_>>()
    )
}

pub struct _Matrix {
    rows: usize,
    columns: usize,
    values: Arc<Vec<AtomicU32>>
}

impl _Matrix {
    /// Returns number of rows.
    pub fn row_count(&self) -> usize { self.rows }

    /// Returns number of columns.
    pub fn column_count(&self) -> usize { self.columns }

    pub fn len(&self) -> usize { self.rows * self.columns }

    /// Creates a new Matrix by taking over ownership of an Arc<Vec<AtomicU32>>.
    /// Private for now, only needed internally.
    fn new(rows: usize, columns: usize, values: Arc<Vec<AtomicU32>>) -> Self {
        Self { rows, columns, values }
    }

    /// Returns a row x column matrix filled with random values between -1.0 and 1.0 inclusive.
    pub fn new_randomized_uniform(rows: usize, columns: usize, uniform: Uniform<f32>) -> Self {
        assert!(columns > 0);
        assert!(rows > 0);

        let mut rng = rand::thread_rng();
        let element_count = columns * rows;
        let values = uniform.sample_iter(&mut rng).take(element_count).collect();

        Self::from(rows, columns, values)
    }

    /// Constructs Matrix from supplied Vec<f32>.
    pub fn from(rows: usize, columns: usize, values: Vec<f32>) -> Self {
        let values = Arc::new(
            values.into_iter()
            .map(|x| AtomicU32::new(x.to_bits()))
            .collect::<Vec<_>>()
        );

        Self { rows, columns, values }
    }

    /// Returns internal values as a Vec<f32>.
    /// Private for now.
    pub fn to_vec(&self) -> Vec<f32> {
        let size = self.columns * self.rows;
        let mut f32s = Vec::with_capacity(size);

        for i in 0..size {
            let x = f32::from_bits(self.values[i].load(Ordering::Relaxed));
            f32s.push(x);
        }

        f32s
    }

    /// Returns a contiguous slice of data representing a vector of columns in the matrix.
    /// Note that they are AtomicU32, so need to_bits and from_bits operations when using.
    fn get_row(&self, row: usize) -> &[AtomicU32] {
        assert!(row < self.rows, "Tried to get a row that was out of bounds from a matrix.");

        let start = row * self.columns;
        let end = start + self.columns;
        &self.values[start..end]
    }

    /// Performs a function on every element in Matrix.
    pub fn map(&self, function: fn(&f32) -> f32, partitioner: &Partitioner) -> Self {
        let values = contiguous(self.rows, self.columns);
        thread::scope(|s| {
            for partition in &partitioner.partitions {
                let partitioned_values = Arc::clone(&values);
                let _handle = s.spawn(move || {
                    for i in partition.get_range() {
                        let cursor = i;
                        let read_value = f32::from_bits(self.values[cursor].load(Ordering::Relaxed));
                        let value = function(&read_value);
                        partitioned_values[cursor].store(value.to_bits(), Ordering::Relaxed);
                    }
                });
            }
        });

        Self::new(self.rows, self.columns, values)
    }

    /// Returns transpose of Matrix.
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
                        let value = self.values[index_to_read].load(Ordering::Acquire);                        
                        partitioned_values[cursor].store(value, Ordering::Release);
                    }
                });
            }
        });

        Self::new(self.columns, self.rows, values)
    }

    /// Returns element-wise product.
    pub fn hadamard(&self, rhs: &Self, partitioner: &Partitioner) -> Self {
        let values = contiguous(self.rows, self.columns);
        thread::scope(|s| {
            for partition in &partitioner.partitions {
                let partitioned_values = Arc::clone(&values);
                let _handle = s.spawn(move || {
                    for row_index in partition.get_range() {
                        let cursor = row_index * self.columns;
                        let ls = self.get_row(row_index);
                        let rs = rhs.get_row(row_index);
                        for column_index in 0..self.columns {
                            let x = f32::from_bits(ls[column_index].load(Ordering::Relaxed));
                            let y = f32::from_bits(rs[column_index].load(Ordering::Relaxed));
                            let value = x * y;
                            partitioned_values[cursor + column_index].store(value.to_bits(), Ordering::Relaxed);
                        }
                    }
                });
            }
        });

        Self::new(self.rows, self.columns, values)
    }

    /// Multiplies two matrices together.
    pub fn mul(&self, rhs: &Self, partitioner: &Partitioner) -> Self {
        self.mul_with_transposed(&rhs.transpose(partitioner), partitioner)
    }

    /// Multiples self to rhs, assuming that rhs has been transposed.
    /// This invariant allows for easy parallelization.
    pub fn mul_with_transposed(&self, rhs: &Self, partitioner: &Partitioner) -> Self {
        assert_eq!(self.columns, rhs.columns);

        let values = contiguous(self.rows, rhs.rows);        
        thread::scope(|s| {
            for partition in &partitioner.partitions {
                let partitioned_values = Arc::clone(&values);
                let _handle = s.spawn(move || {
                    for row_index in partition.get_range() {
                        let cursor = row_index * rhs.rows;
                        let ls= self.get_row(row_index);
                        for transposed_row_index in 0..rhs.rows {
                            let rs = rhs.get_row(transposed_row_index);
                            let value = dot_product_atomic(&ls, &rs);
                            partitioned_values[cursor + transposed_row_index].store(value.to_bits(), Ordering::Relaxed);
                        }
                    }
                });
            }
        });

        Self::new(self.rows, rhs.rows, values)
    }
}

#[cfg (test)]
mod tests {
    use super::*;

    #[test]
    fn dot_product_atomic() {

    }

    #[test]
    fn test_transpose() {
        let tc1 = _Matrix::from(3, 4, vec![
            1., 2., 3., 4.,
            5., 6., 7., 8.,
            9., 10., 11., 12.
        ]);

        let expected = _Matrix::from(4, 3, vec![
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
        let lhs = _Matrix::from(3, 3, vec![
            1., 1., 1.,
            2., 2., 2.,
            3., 3., 3.
        ]);

        let rhs = _Matrix::from(3, 3, vec![
            10., 20., 30.,
            10., 20., 30.,
            10., 20., 30.
        ]);

        let expected = _Matrix::from(3, 3, vec![
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
    fn test_mul_with_transposed() {
        let lhs = _Matrix::from(4, 3,  vec![
            1., 2., 3.,
            4., 5., 6.,
            7., 8., 9.,
            10., 11., 12.
        ]);

        // Assume already transposed.
        let rhs = &_Matrix::from(5, 3, vec![
            1., 6., 11.,
            2., 7., 12.,
            3., 8., 13.,
            4., 9., 14.,
            5., 10., 15.
        ]);

        let expected = _Matrix::from(4, 5, vec! [
            46., 52., 58., 64., 70.,
            100., 115., 130., 145., 160.,
            154., 178., 202., 226., 250.,
            208., 241., 274., 307., 340.
        ]);

        let partitioner = &Partitioner::with_partitions(lhs.rows, 2); 
        let actual = lhs.mul_with_transposed(rhs, partitioner);

        assert_eq!(actual.to_vec(), expected.to_vec());
    }
}