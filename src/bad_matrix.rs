struct BadMatrix {

}

impl BadMatrix {
    /// Multiplies self to rhs, assuming that rhs has been transposed.
    /// # Partition by self.row_count().
    pub fn mul_with_transposed(&self, rhs: &Self, partitioner: &Partitioner) -> Self {
        assert_eq!(self.columns, rhs.columns);

        let values = contiguous(self.rows, rhs.rows);        
        thread::scope(|s| {
            for partition in &partitioner.partitions {
                let partitioned_values = Arc::clone(&values);
                let _handle = s.spawn(move || {
                    for row_index in partition.get_range() {
                        let cursor = row_index * rhs.rows;
                        let ls= self.atomic_row(row_index);
                        for transposed_row_index in 0..rhs.rows {
                            let rs = rhs.atomic_row(transposed_row_index);
                            let value = dot_product_atomic(&ls, &rs);
                            partitioned_values[cursor + transposed_row_index].come_in(value);
                        }
                    }
                });
            }
        });

        Self::new(self.rows, rhs.rows, values)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_mul_with_transposed() {
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

        let partitioner = &Partitioner::with_partitions(lhs.rows, 2); 
        let actual = lhs.mul_with_transposed(rhs, partitioner);

        assert_eq!(actual.to_vec(), expected.to_vec());
    }
    
}