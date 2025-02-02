struct BadMatrix {

}

impl BadMatrix {
    /// Returns element-wise product.
    /// # Partition by self.row_count().
    pub fn hadamard_atomic(&self, rhs: &Self, partitioner: &Partitioner) -> Self {
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

    /// Multiplies self to rhs, assuming that rhs has been transposed.
    /// # Partition by self.row_count().
    pub fn mul_with_transposed_atomic(&self, rhs: &Self, partitioner: &Partitioner) -> Self {
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
}