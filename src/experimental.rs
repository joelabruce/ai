use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;

use crate::partitions::{Partition, Partitioner};

/// Calculates dot product of two slices of AtomicU64 as if they were floats.
pub fn dot_product_atomic(lhs: &[AtomicU64], rhs: &[AtomicU64]) -> f64 {
    let n = lhs.len();

    let mut sum = 0.;
    for i in 0..n {
        sum += f64::from_bits(lhs[i].load(Ordering::Relaxed)) * 
            f64::from_bits(rhs[i].load(Ordering::Relaxed));
    }

    sum
}

pub struct _Matrix {
    values: Arc<Vec<AtomicU64>>,
    rows: usize,
    columns: usize
}

impl _Matrix {
    fn contiguous(rows: usize, columns: usize) -> Arc<Vec<AtomicU64>> {
        let size = rows * columns;
        Arc::new(
            (0..size)
            .map(|_| AtomicU64::new(0))
            .collect::<Vec<_>>()
        )
    }

    fn new(rows: usize, columns: usize, values: Arc<Vec<AtomicU64>>) -> Self {
        Self { rows, columns, values }
    }

    fn to_vec(&self) -> Vec<f64> {
        let size = self.columns * self.rows;
        let mut f64s = Vec::with_capacity(size);

        for i in 0..size {
            let x = f64::from_bits(self.values[i].load(Ordering::Relaxed));
            f64s.push(x);
        }

        f64s
    }

    pub fn from(rows: usize, columns: usize, values: Vec<f64>) -> Self {
        let values = Arc::new(
            values.into_iter()
            .map(|x| AtomicU64::new(x.to_bits()))
            .collect::<Vec<_>>()
        );

        Self { rows, columns, values }
    }

    /// Returns a contiguous slice of data representing columns in the matrix.
    /// Note that they are AtomicU64, so need to_bits and from_bits operations when using.
    fn get_row(&self, row: usize) -> &[AtomicU64] {
        assert!(row < self.rows, "Tried to get a row that was out of bounds.");

        let start = row * self.columns;
        let end = start + self.columns;
        &self.values[start..end]
    }    

    pub fn tryout(&self, partitioner: &Partitioner) {
        thread::scope(|s| {
            for partition in &partitioner.partitions[..] {
                let data = Arc::clone(&self.values);
                let _handle = s.spawn(move || {
                    for cursor in partition.get_range() {
                        // Actual operation goes here
                        let value = cursor as f64 * 1.1; // Example computation
                        
                        
                        data[cursor].store(value.to_bits(), Ordering::Relaxed);
                        // End of operation
                    }
                });
            }
        });

        // Read results
        let size = self.rows * self.columns;
        for i in 0..size {
            let value = f64::from_bits(self.values[i].load(Ordering::Relaxed));
            println!("data[{}] = {}", i, value);
        }
    }

    pub fn mul_with_transposed(&self, rhs: &_Matrix, partitioner: &Partitioner) -> _Matrix {
        let values = _Matrix::contiguous(self.rows, rhs.rows);
        
        thread::scope(|s| {
            for partition in &partitioner.partitions {
                let partitioned_values = Arc::clone(&values);
                let _handle = s.spawn(move || {
                    for row in partition.get_range() {
                        let cursor = row * rhs.rows;
                        let ls= self.get_row(row);
                        for transposed_row in 0..rhs.rows {
                            let rs = rhs.get_row(transposed_row);
                            let dot_product = dot_product_atomic(&ls, &rs);

                            partitioned_values[cursor + transposed_row].store(dot_product.to_bits(), Ordering::Relaxed);
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
    fn test_experimental() {
        let rows = 300;
        let columns = 200; 
        let partition_count = thread::available_parallelism().unwrap().get();
        let partitioner = &Partitioner::with_partitions(rows * columns, partition_count);

        //let test = _Matrix::new(rows, columns);
        //test.tryout(partitioner);
    }

    #[test]
    fn dot_product_atomic() {

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
        let rhs = _Matrix::from(5, 3, vec![
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
        let actual = lhs.mul_with_transposed(&rhs, partitioner);

        //for i in actual.values
        assert_eq!(actual.to_vec(), expected.to_vec());
    }
}