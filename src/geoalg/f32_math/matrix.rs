use std::{ops::{Index, IndexMut}, thread};
use rand_distr::{Distribution, Normal, Uniform};
use crate::{geoalg::f32_math::simd_extensions::dot_product_simd3, nn::layers::convolution2d::Dimensions, partition::Partition, partitioner::Partitioner};

use super::simd_extensions::{im2col_transposed, SliceExt};

/// Matrix is implemented as a single dimensional vector of f32s.
/// This implementation of Matrix is row-major. 
/// Row-major is specified so certain optimizations and parallelization can be performed.
/// Column-major is not implemented. Unless it helps with optimizations, may never be implemented.
#[derive(PartialEq, Debug, Clone, Default)]
pub struct Matrix {
    rows: usize,
    columns: usize,
    values: Vec<f32>
}

impl Index<usize> for Matrix {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.values[index]
    }
}

impl IndexMut<usize> for Matrix {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.values.index_mut(index)
    }
}

impl Matrix {
    /// Returns size of underlying vector.
    /// Tensor.shape.size
    pub fn len(&self) -> usize { self.values.len() }

    /// Returns number of rows this matrix has.
    /// Tensor.shape.axis_len
    pub fn row_count(&self) -> usize { self.rows }

    /// Returns number of columns this matrix has.
    /// Tensor.shape.axis_len
    pub fn column_count(&self) -> usize { self.columns }

    pub fn shape(&self) -> (usize, usize) { (self.rows, self.columns) }

    /// Returns a slice of the values this matrix has.
    /// Tensor.stream
    pub fn read_values(&self) -> &[f32] { &self.values }

    /// Returns a new Matrix.
    pub fn new(rows: usize, columns: usize, values: Vec<f32>) -> Self {
        Self {
            rows, columns, values
        }
    }

    /// Returns a contiguous slice of data representing columns in the matrix.
    /// Tensor.Slice
    pub fn row(&self, row_index: usize) -> &[f32] {
        assert!(row_index < self.rows, "Tried to get a row that was out of bounds.");

        let start = row_index * self.columns;
        let end = start + self.columns;
        &self.values[start..end]
    }

    /// Returns a row x column matrix filled with random values between -1.0 and 1.0 inclusive.
    /// Deprecated
    pub fn new_randomized_z(rows: usize, columns: usize) -> Self {
        assert!(columns > 0);
        assert!(rows > 0);

        let mut rng = rand::thread_rng();
        let step = Uniform::new_inclusive(-1.0, 1.0);
        let element_count = columns * rows;
        let values = step.sample_iter(&mut rng).take(element_count).collect();

        Self::new(rows, columns, values)
    }

    /// Returns an rows x column matrix filled with random values specified by uniform distribution.
    /// In Tensor
    pub fn new_randomized_uniform(rows: usize, columns: usize, uniform: Uniform<f32>) -> Self {
        assert!(columns > 0);
        assert!(rows > 0);

        let mut rng = rand::thread_rng();
        let element_count = columns * rows;
        let values = uniform.sample_iter(&mut rng).take(element_count).collect();
        
        Self::new(rows, columns, values)
    }

    /// In Tensor
    pub fn new_randomized_normal(rows: usize, columns: usize, normal: Normal<f32>) -> Self {
        let mut rng = rand::thread_rng();
        let element_count = columns * rows;        
        let values = normal.sample_iter(&mut rng).take(element_count).collect();

        Self::new(rows, columns, values)
    }

    /// Returns transpose of matrix.
    /// Partitioner implementation complete.
    /// Now in Tensor
    pub fn transpose(&self) -> Self {
        if self.rows == 1 || self.columns == 1 {
            return Self::new(self.columns, self.rows, self.values.clone());
        }

        let partition_strategy = &Partitioner::with_partitions(
            self.len(),
            thread::available_parallelism().unwrap().get());
        
        let inner_process = move |partition: &Partition| {
            let mut partition_values = Vec::with_capacity(partition.size());
            for i in partition.range() {
                let index_to_read = self.columns * (i % self.rows) + i / self.rows;
                partition_values.push(self.values[index_to_read]);
            }

            partition_values
        };

        let values = partition_strategy.parallelized(inner_process);
        Self::new(self.columns, self.rows, values)
    }

    /// Makes use of supplied partitions to parallelize the operation.
    /// If partitions is cached, can be reused (to hopefully save even more time).
    /// Partitioner implementation complete.
    /// In Tensor.
    pub fn mul_element_wise(&self, rhs: &Matrix) -> Self {
        assert!(self.rows == rhs.rows && self.columns == rhs.columns, "When element-wise multiplying two matrices, they must have same order.");
        
        let partition_strategy = &Partitioner::with_partitions(
            self.len(),
            thread::available_parallelism().unwrap().get());

        let inner_process = |partition: &Partition| {
            let mut partition_values: Vec<f32> = Vec::with_capacity(partition.size());
            for i in partition.range() {
                partition_values.push(self.values[i] * rhs.values[i]);
            }

            partition_values
        };

        let values = partition_strategy.parallelized(inner_process);

        Self::new(self.row_count(), self.column_count(), values)
    }

    /// Computes matrix multiplication and divying up work amongst partitions.
    /// Faster multiplcation when you need to multiply the transposed matrix of rhs.
    /// Avoids calculating the transpose twice.
    /// Partitioner implementation complete.
    /// Now in Tensor.
    pub fn mul_with_transpose(&self, rhs: &Matrix) -> Matrix {
        assert_eq!(self.columns, rhs.columns, "When multiplying with transposed, columns must be equal for lhs and rhs.");

        let partition_strategy = Partitioner::with_partitions(self.rows, thread::available_parallelism().unwrap().get());

        let inner_process = move |partition: &Partition| {
            let mut partition_values: Vec<f32> = Vec::with_capacity(partition.size() * rhs.rows);
            for row in partition.range() {
                let ls = self.row(row);
                for transposed_row in 0..rhs.rows {
                    let rs = rhs.row(transposed_row);
                    //let dot_product = dot_product_of_vector_slices(&ls, &rs);
                    let dot_product = dot_product_simd3(&ls, &rs);
                    partition_values.push(dot_product);
                }
            }
            partition_values
        };

        let values = partition_strategy.parallelized(inner_process);
        Self::new(self.rows, rhs.rows, values)
    }

    pub fn mul_transpose_simd(&self, rhs: &Matrix) -> Matrix {
        let values = self.values.par_mm_transpose(
            &rhs.values, 
            self.row_count(),
            self.column_count(),
            rhs.row_count());

        Matrix::new(self.row_count(), rhs.row_count(), values)
    }

    /// Useful for applying an activation function to the entire matrix.
    /// Partitioner implementation complete.
    /// Rethink this, as it doesn't optimize well.
    /// Deprecated.
    pub fn map(&self, func: fn(&f32) -> f32) -> Self {
        let partition_strategy = &Partitioner::with_partitions(self.len(), thread::available_parallelism().unwrap().get());

        let inner_process = move |partition: &Partition| {
            let mut partition_values = Vec::with_capacity(partition.size());
            for i in partition.range() {
                partition_values.push(func(&self.values[i]));
            }

            partition_values
        };

        let values = partition_strategy.parallelized(inner_process);
        Self::new(self.row_count(), self.column_count(), values)
    }

    /// Subtracts rhs Matrix from lhs Matrix.
    /// Partitioner implementation complete.
    /// In Tensor
    pub fn sub(&self, rhs: &Matrix) -> Self {
        assert!(self.rows == rhs.rows && self.columns == rhs.columns, "When subtracting two matrices, they must have same order.");
 
        let partition_strategy = &Partitioner::with_partitions(self.len(), thread::available_parallelism().unwrap().get());

        let inner_process = move |partition: &Partition| {
            let mut partition_values = Vec::with_capacity(partition.size());
            for i in partition.range() {
                partition_values.push(self.values[i] - rhs.values[i]);
            }

            partition_values
        };

        let values = partition_strategy.parallelized(inner_process);
        Self::new(self.row_count(), self.column_count(), values)
    }

    /// In tensor.
    pub fn add(&self, rhs: &Matrix) -> Self {
        assert!(self.rows == rhs.rows && self.columns == rhs.columns, "When subtracting two matrices, they must have same order.");
    
        let partition_strategy = &Partitioner::with_partitions(self.len(), thread::available_parallelism().unwrap().get());

        let inner_process = move |partition: &Partition| {
            let mut partition_values = Vec::with_capacity(partition.size());
            for i in partition.range() {
                partition_values.push(self.values[i] + rhs.values[i]);
            }

            partition_values
        };

        let values = partition_strategy.parallelized(inner_process);
        Self::new(self.row_count(), self.column_count(), values)
    }

    /// Adds a row to each row in matrix.
    /// Partitioner implementation complete.
    /// In Tensor
    pub fn add_row_partitioned(&self, rhs: &Matrix) -> Self {
        assert_eq!(rhs.rows, 1, "Rhs matrix must have 1 row.");
        assert_eq!(self.columns, rhs.columns, "Lhs and rhs must have equal number of columns.");

        let partition_strategy = &Partitioner::with_partitions(self.rows, thread::available_parallelism().unwrap().get());

        let inner_process = move |partition: &Partition| {
            let mut partition_values= Vec::with_capacity(partition.size() * self.columns);
            let rs = rhs.row(0); 
            for row in partition.range() {
                let ls = self.row(row);

                for column in 0..ls.len() {
                    partition_values.push(ls[column] + rs[column]);
                }
            }

            partition_values
        };

        let values = partition_strategy.parallelized(inner_process);
        Self::new(self.row_count(), self.column_count(), values)
    }

    /// Returns a 1 row matrix where each column is the sum of all values for that column.
    /// Partitioner implementation complete.
    /// In Tensor
    pub fn reduce_rows_by_add(&self) -> Self {
        let partition_strategy = &Partitioner::with_partitions(self.columns, thread::available_parallelism().unwrap().get());

        let inner_process =move |partition: &Partition| {
            let mut partition_values = Vec::with_capacity(partition.size());
            for column in partition.range() {
                let mut accumulator = 0.;
                for row in 0..self.rows {
                    accumulator += self.values[column + row * self.columns];
                }

                partition_values.push(accumulator);
            }

            partition_values
        };

        let values = partition_strategy.parallelized(inner_process);
        Self::new(1, self.columns, values)
    }

    /// Scales matrix by a scalar.
    /// Instead of making a division operator, please pass in reciprocal of scalar.
    /// Partitioner implementation complete.
    /// Now in Tensor
    pub fn scale(&self, scalar: f32) -> Self {
        let partition_strategy = &Partitioner::with_partitions(self.len(), thread::available_parallelism().unwrap().get());

        let inner_process = move |partition: &Partition| {
            let mut partition_values = Vec::with_capacity(partition.size());
            for i in partition.range() {
                partition_values.push(self.values[i] * scalar);
            }

            partition_values
        };

        let values = partition_strategy.parallelized(inner_process);
        Self::new(self.row_count(), self.column_count(), values)
    }

    /// Parallelized cross correlation with im2col (non-batched version)
    pub fn par_cc_im2col(&self, kernels: &Matrix, k_d: &Dimensions, i_d: &Dimensions) -> Self {
        let batches = self.row_count();

        let partitioner = &Partitioner::with_partitions(
            batches, 
            thread::available_parallelism().unwrap().get());

        // Adjust for valid convolution (no padding)
        let feature_rows = i_d.height - k_d.height + 1;
        let feature_columns = i_d.width - k_d.width + 1;
        let filters_size = kernels.row_count() * feature_rows * feature_columns;
        let kernel_size = k_d.height * k_d.width;

        let inner_process = move |partition: &Partition| {
            let mut partition_values = Vec::with_capacity(partition.size() * filters_size);
            for batch_index in partition.range() {
                let input = self.row(batch_index);

                let image = &*im2col_transposed(
                    input, 
                    i_d.height,i_d.width,
                    k_d.height, k_d.width);

                let x = &*kernels.read_values().mm_transpose(
                    image, 
                    kernels.row_count(), 
                    kernel_size, feature_rows * feature_columns);

                partition_values.extend_from_slice(x);

            }

            partition_values
        };

        let values = partitioner.parallelized(inner_process);
        Matrix::new(batches, filters_size, values)
    }

    /// Single-threaded convolution
    pub fn cc_im2col(&self, kernels: &Matrix, k_d: &Dimensions, i_d: &Dimensions) -> Self {
        let batches = self.row_count();

        // Adjust for valid convolution (no padding)
        let feature_rows = i_d.height - k_d.height + 1;
        let feature_columns = i_d.width - k_d.width + 1;
        let filters_size = kernels.row_count() * feature_rows * feature_columns;
        let kernel_size = k_d.height * k_d.width;

        let mut partition_values = Vec::with_capacity(batches * filters_size);
        for batch_index in 0..batches {
            let input = self.row(batch_index);

            let image = &*im2col_transposed(
                input, 
                i_d.height,i_d.width,
                k_d.height, k_d.width);

            let x = &*kernels.read_values().mm_transpose(
                image, 
                kernels.row_count(), 
                kernel_size, feature_rows * feature_columns);

            partition_values.extend_from_slice(x);

        }

        Matrix::new(batches, filters_size, partition_values)
    }

    /// Used for convolutional layers.
    /// Might consider creating outside of matrix.
    /// Now in Tensor
    pub fn valid_cross_correlation(&self, kernels: &Matrix, k_d: &Dimensions, i_d: &Dimensions) -> Self {
        let batches = self.row_count();

        let partitioner = &Partitioner::with_partitions(
            batches, 
            thread::available_parallelism().unwrap().get());

        // Adjust for valid convolution (no padding)
        let feature_rows = i_d.height - k_d.height + 1;
        let feature_columns = i_d.width - k_d.width + 1;
        let filters_size = kernels.row_count() * feature_rows * feature_columns;

        let inner_process = move |partition: &Partition| {
            let mut partition_values = Vec::with_capacity(partition.size() * filters_size);
            for batch_index in partition.range() {
                let input = self.row(batch_index);

                // Convolve with each kernel in order
                for filter_index in 0..kernels.row_count() {
                    let filter = kernels.row(filter_index);

                    // Slides the kernel from top to bottom
                    for feature_row in 0..feature_rows {
                        // Slides the kernel from left to right
                        for feature_column in 0..feature_columns {
                            let mut c_accum = 0.;

                            // Calculates the contribution of kernel_rows dot product with image row.
                            // Uses slices instead of having to iterate over every column manually.
                            for kernel_row in 0..k_d.height {
                                // Get the row of the input, offset by kernel row, and start the row at the column.
                                let input_row_start_index = (feature_row + kernel_row) * i_d.width + feature_column;
                                // Only get as many columns as are in the kernel for the convolution.
                                let input_row_end_index = input_row_start_index + k_d.width;

                                let kernel_row_start_index = kernel_row * k_d.width;
                                let kernel_row_end_index = kernel_row_start_index + k_d.width;

                                let x = &input[input_row_start_index..input_row_end_index];
                                let y = &filter[kernel_row_start_index..kernel_row_end_index];

                                c_accum += dot_product_simd3(x, y)
                            }

                            partition_values.push(c_accum);
                        }
                    }
                }
            }

            partition_values
        };

        let values = partitioner.parallelized(inner_process);
        Matrix::new(batches, filters_size, values)
    }

    /// Todo: Needs to be optimized next.
    pub fn full_outer_convolution(&self, filters: &Matrix, k_d: &Dimensions, i_d: &Dimensions) -> Self {
        let batches = self.row_count();

        let partitioner = &Partitioner::with_partitions(
            batches, 
            thread::available_parallelism().unwrap().get());

        // Adjust for full outer convolution (don't padd, just do bounds checking)
        let i_rows = (i_d.height - k_d.height + 1) as isize;
        let i_columns = (i_d.width - k_d.width + 1) as isize;

        let o_rows = i_d.height;
        let o_columns =i_d.width;
        let filters_size = filters.row_count() * o_rows * o_columns;

        let inner_process = move |partition: &Partition| {
            let mut partition_values = vec![0.; partition.size() * filters_size];
            for batch_index in partition.range() {
                //let input = self.row(batch_index);

                let batch_offset = batch_index * o_rows * o_columns;
                for filter_index in 0..filters.row_count() {
                    //let filter = kernels.row(filter_index);
                    let filter_offset = filter_index * k_d.height * k_d.width; 

                    for row in 0..o_rows {
                        for column in 0..o_columns {
                            let mut c_accum = 0.;
                            for kernel_row in 0..k_d.height {
                                for kernel_column in 0..k_d.width {
                                    let input_row = row as isize - (k_d.height - kernel_row - 1) as isize;
                                    let input_column = column as isize - (k_d.width - kernel_column - 1) as isize;

                                    if input_row >= 0 && input_row < i_rows &&
                                       input_column > 0 && input_column < i_columns {
                                        let input_offset = (batch_index as isize) * i_rows * i_columns;

                                        c_accum += self.values[(input_offset + input_row * i_columns + input_column) as usize] 
                                            * filters.values[filter_offset + kernel_row * k_d.width + kernel_column];
                                    }

                                    //c_accum += dot_product_simd3(x, y)
                                }
                            }

                            //print!("{c_accum}, ");
                            partition_values[batch_offset + row * o_columns + column] = c_accum;
                            //partition_values.push(c_accum);
                        }
                    }
                }
            }

            partition_values
        };

        let values = partitioner.parallelized(inner_process);
        Matrix::new(batches, filters_size, values)
    }

    /// Pads a matrix with rows and columns specified by p_d
    pub fn pad(&self, p_d: Dimensions) -> Self {
        let padded_rows = self.row_count() + p_d.height * 2;
        let padded_columns = self.column_count() + p_d.width * 2;

        let mut values = vec![0.; padded_rows * padded_columns];
        let row_offset = p_d.height;
        for row in 0..self.rows {
            let row_to_copy = self.row(row);
            for column in 0..self.column_count() {
                values[(row_offset + row) * padded_columns + p_d.width + column] = row_to_copy[column];
            }
        }

        Self::new(padded_rows, padded_columns, values)
    }

    /// Might consider putting outside of matrix.
    /// i_d: input dimensions
    /// p_d: pooling dimensions
    /// o_d: output_dimensions
    /// Todo: Put in tensor
    pub fn maxpool(&self, filters: usize, stride: usize, i_d: &Dimensions, p_d: &Dimensions, o_d: &Dimensions) -> (Self, Vec<usize>) {
        let batches = self.row_count();

        let partitioner = &Partitioner::with_partitions(
            batches,
            thread::available_parallelism().unwrap().get());

        let (rows, columns) = o_d.shape();
        let rows_x_columns = rows * columns;

        let inner_process = move |partition: &Partition| {
            let mut partition_values = Vec::with_capacity(partition.size() * filters * rows_x_columns);
            for batch in partition.range() {
                let input_row = self.row(batch);
                for filter in 0..filters {
                    let filter_offset = filter * i_d.height * i_d.width;
                    for row in 0..rows {
                        let w_row = row * stride;
                        for column in 0..columns {
                            let w_column = column * stride;

                            let mut max = f32::MIN;
                            let mut max_index = 0;
                            for k_row in 0..p_d.height {
                                for k_column in 0..p_d.width {
                                    let index = filter_offset + (w_column + k_column) + (w_row + k_row) * i_d.width;
                                    let index_value = input_row[index];
                                    if index_value > max { 
                                        max = index_value;
                                        max_index = index;
                                    } 
                                }
                            }

                            partition_values.push((max, max_index));
                        }
                    }
                }
            }

            partition_values
        };

        let (values, max_indices) = partitioner.parallelized(inner_process).into_iter().unzip();

        //let msg = format!("{:?}", max_indices).bright_purple();
        //println!("Max indices: {msg}");

        (Matrix::new(batches, filters * rows_x_columns, values), max_indices)
    }
}

#[cfg(test)]
mod tests {
    use crate::prettify::*;

    use super::*;

    #[test]
    fn test_from_vec() {
        let actual = Matrix {
            values: vec![
                1., 2., 3.,
                4., 5., 6.
                ], 
            rows: 2,
            columns: 3};

        let expected = Matrix::new(
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
        let tc = Matrix::new(1, 3, vec![]);

        tc.row(1);
    }

    #[test]
    fn transpose_test() {
        let m = Matrix::new(5, 4, vec![
                0f32, 1f32, 2f32, 3f32,
                4f32, 5f32, 6f32, 7f32,
                8f32, 9f32, 10f32, 11f32,
                12f32, 13f32, 14f32, 15f32,
                16f32, 17f32, 18f32, 19f32
            ]);

        let expected = Matrix::new(4, 5, vec![
                0f32, 4f32, 8f32, 12f32, 16f32,
                1f32, 5f32, 9f32, 13f32, 17f32,
                2f32, 6f32, 10f32, 14f32, 18f32,
                3f32, 7f32, 11f32, 15f32, 19f32
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
        let tc = Matrix::new(2, 3, vec![
                1., 2., 3.,
                4., 5., 6.
            ]);

        let actual = tc.scale(3.);
        let expected = Matrix::new(2, 3, vec![
                3., 6., 9.,
                12., 15., 18.
            ]);

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_shrink_rows_by_add() {
        let tc = Matrix::new(3, 4, vec![
                1., 1., 2., 10.,
                2., 3., 3., 10.,
                4., 4., 5., 10.
            ]);

        let expected = Matrix::new(1, 4, vec![
                7., 8., 10., 30.
            ]);

        let actual2 = tc.reduce_rows_by_add();
        assert_eq!(actual2, expected);
    }

    #[test]
    fn test_get_row_vector_slice() {
        let tc = Matrix::new(4, 3, vec![
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
        let tc = Matrix::new(3, 4, vec![
                0., 0., 0., 0.,
                1., 1., 1., 1.,
                2., 2., 2., 2.
            ]);

        let row_to_add = Matrix::new(1, 4, vec![10., 20., 30., 40.]);

        let expected = Matrix::new(3, 4, vec![
                10., 20., 30., 40.,
                11., 21., 31., 41.,
                12., 22., 32., 42.
            ]);

        let actual = tc.add_row_partitioned(&row_to_add);
        assert_eq!(actual, expected);        
    }

    #[test]
    fn test_mul_with_transpose() {
        let lhs = Matrix::new(4, 3,  vec![
            1., 2., 3.,
            4., 5., 6.,
            7., 8., 9.,
            10., 11., 12.
        ]);

        // Assume already transposed.
        let rhs = &Matrix::new(5, 3, vec![
            1., 6., 11.,
            2., 7., 12.,
            3., 8., 13.,
            4., 9., 14.,
            5., 10., 15.
        ]);

        let expected = Matrix::new(4, 5, vec! [
            46., 52., 58., 64., 70.,
            100., 115., 130., 145., 160.,
            154., 178., 202., 226., 250.,
            208., 241., 274., 307., 340.
        ]);

        let actual = lhs.mul_with_transpose(rhs);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_pad() {
        let tc = Matrix::new(2, 2, vec![
            1., 2.,
            3., 4.
        ]);

        let actual = tc.pad(Dimensions { width: 2, height: 2});
        let expected = Matrix::new(6, 6, vec![
            0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0.,
            0., 0., 1., 2., 0., 0.,
            0., 0., 3., 4., 0., 0.,
            0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0.,
        ]);

        //let msg = format!("{:?}", actual).bright_purple();
        //println!("{msg}");
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_full_outer_convolution() {
        let input = Matrix::new(1, 3 * 3, vec![
            1., 2., 3.,
            4., 5., 6.,
            7., 8., 9.
        ]);
        
        let kernels = Matrix::new(1, 3 * 3, vec![
            0., 1., 0.,
            1., -4., 1.,
            0., 1., 0.
        ]);

        let actual = input.full_outer_convolution(
            &kernels, 
            &Dimensions { width: 3, height: 3 },
            &Dimensions { width: 5, height: 5 }
        );

        println!("{BRIGHT_MAGENTA}{:?}{RESET}", actual);
    }
}
