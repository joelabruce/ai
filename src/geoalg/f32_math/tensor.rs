use std::{ops::{Index, IndexMut}, simd::{cmp::SimdPartialOrd, num::SimdFloat, Simd}, thread};
use rand_distr::{Distribution, Normal, Uniform};

use crate::{nn::layers::convolution2d::Dimensions, partition::Partition, partitioner::Partitioner};

use super::{shape::Shape, simd_extensions::{dot_product_simd3, ALL_SIMD_LANES}};

#[derive(Debug, PartialEq, Clone)]
pub struct Tensor {
    pub shape: Shape,
    values: Vec<f32>
}

impl Index<usize> for Tensor {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output { &self.values[index] }
}

impl IndexMut<usize> for Tensor {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.values[index]
    }
}

impl Tensor {
    /// Creates a vector tensor specified by values.
    pub fn vector(values: Vec<f32>) -> Self { Tensor::new(Shape::d1(values.len()), values) }

    /// Creates a row-major matrix tensor.
    pub fn matrix(rows: usize, columns: usize, values: Vec<f32>) -> Self { Tensor::new(Shape::d2(rows, columns), values) }

    /// Creates a tensor with uniform distribution of random values.
    pub fn new_randomized_uniform(shape: Shape, uniform: Uniform<f32>) -> Self {
        let mut rng = rand::thread_rng();
        let element_count = shape.size();
        let values = uniform.sample_iter(&mut rng).take(element_count).collect();

        Tensor::new(shape, values)
    }

    /// Creates a tensor with normal distribution of random values.
    pub fn new_randomized_normal(shape: Shape, normal: Normal<f32>) -> Self {
        let mut rng = rand::thread_rng();
        let element_count = shape.size();        
        let values = normal.sample_iter(&mut rng).take(element_count).collect();

        Tensor::new(shape, values)
    }

    /// Creates a new tensor with specified shape.
    pub fn new(shape: Shape, values: Vec<f32>) -> Self { Tensor { shape, values } }

    /// Moves values into a new Tensor with a different shape.
    pub fn reshape(self, new_shape: Shape) -> Self {
        let old_size = self.shape.size();
        let new_size = new_shape.size();
        assert_eq!(old_size, new_size, "Cannot reshape tensor into a shape that doesn't have same size.");

        Self::new(new_shape, self.values)
    }

    /// Get a read-only stream to underlying values.
    /// Needed for simd to be super fast! :D
    pub fn stream(&self) -> &[f32] { &self.values }

    /// Gets a contiguous subset of values.
    /// Experimental.
    pub fn slice(&self, start: Shape, end: Shape) -> &[f32] {
        let start = self.shape.index_at(&start);
        let end = self.shape.index_at(&end);

        &self.values[start..end]
    }

    /// Get a slice of contiguous data from the Tensor. Only grabs 1 slice from dimension.
    /// * `axis`: Axis to pull data from.
    /// * `index`: Index in dimension to start.
    /// **Unstable**
    pub fn dim_slice(&self, axis: usize, index: usize) -> &[f32] {
        let size = self.shape.stride_for(axis);
        &self.values[size * index..size * (index + 1)]
    }

    /// Performs relu on each value in Tensor.
    pub fn relu_simd(&self) -> Self {
        let partitioner = Partitioner::with_partitions_simd(
            self.shape.size(), 
            thread::available_parallelism().unwrap().get());

        let y_simd = Simd::<f32, ALL_SIMD_LANES>::splat(0.);
        let inner_process = |partition: &Partition| {
            let mut partition_values: Vec<f32> = Vec::with_capacity(partition.size());
            partition.unary_simd(
                &mut partition_values, 
                &self.stream(),
                |x_simd| x_simd.simd_max(y_simd),
                |x| x.max(0.)
            );

            partition_values
        };

        let values = partitioner.parallelized(inner_process);
        Self::new(self.shape.clone(), values)
    }

    /// Derivative of relu on each element in Tensor.
    pub fn d_relu_simd(&self) -> Self {
        let partitioner = Partitioner::with_partitions_simd(
            self.shape.size(), 
            thread::available_parallelism().unwrap().get());

        let y_simd = Simd::<f32, ALL_SIMD_LANES>::splat(0.);
        let true_mask = Simd::<f32, ALL_SIMD_LANES>::splat(1.);
        let inner_process = |partition: &Partition| {
            let mut partition_values: Vec<f32> = Vec::with_capacity(partition.size());
            partition.unary_simd(
                &mut partition_values, 
                &self.stream(),
                |x_simd| {
                    let mask = x_simd.simd_gt(y_simd);
                    mask.select(true_mask, y_simd)
                },
                |x| if x > 0. { 1. } else { 0. }
            );

            partition_values
        };

        let values = partitioner.parallelized(inner_process);
        Self::new(self.shape.clone(), values)
    }

    /// Transposes the last 2 dimensions for only the first batch.
    /// Might consider doing a batch transpose, but not implemented yet.
    /// **Unstable**
    pub fn transpose(&self) -> Self {
        let partition_strategy = &Partitioner::with_partitions(
            self.shape.size(), 
            thread::available_parallelism().unwrap().get());

        let row_dimension = self.shape.len() - 2;
        let rows = self.shape[row_dimension];
        let columns = self.shape[self.shape.len() - 1];

        let mut dimensions = self.shape.dims()[0..row_dimension].to_vec();
        dimensions.push(columns);
        dimensions.push(rows);
        let shape = Shape::new(dimensions);

        let inner_process = move |partition: &Partition| {
            let mut partition_values = Vec::with_capacity(partition.size());
            for i in partition.range() {
                let index_to_read = columns * (i % rows) + i / rows;
                partition_values.push(self[index_to_read]);
            }

            partition_values
        };

        let values = partition_strategy.parallelized(inner_process);
        Self::new(shape, values)
    }

    /// Will matrix multiply two tensors, with the assumption that rhs has already been transposed.
    /// Will use the last two dimensions, does not yet support batching or broadcasting.
    /// * For now assumes only 1 matrix per tensor, will update when needed.
    pub fn mul_transpose_simd(&self, rhs: &Self) -> Self {
        assert!(self.shape.len() >= 2 && rhs.shape.len() >= 2, "Both tensors must have a minimum of 2 dimensions each.");

        let lhs_row_dimension = self.shape.len() - 2;
        let lhs_rows = self.shape[lhs_row_dimension];
        let lhs_stride = self.shape.stride_for(lhs_row_dimension);

        let rhs_row_dimension = rhs.shape.len() - 2;
        let rhs_rows = rhs.shape[rhs_row_dimension];
        let rhs_stride = rhs.shape.stride_for(rhs_row_dimension);
        
        // Do not partition by simd here because we are parallelizing off of lhs rows.
        let partition_strategy = Partitioner::with_partitions(
            lhs_rows,
            thread::available_parallelism().unwrap().get());

        let inner_process = move |partition: &Partition| {
            let mut partition_values: Vec<f32> = Vec::with_capacity(partition.size() * rhs_rows);
            
            let mut lhs_start = partition.get_start() * lhs_stride;
            let mut lhs_end = lhs_start + lhs_stride;
            for _row in partition.range() {
                // Grab the row from self
                let l_slice = &self.stream()[lhs_start..lhs_end];

                let mut rhs_start = 0;
                let mut rhs_end = rhs_stride;
                for _transposed_row in 0..rhs_rows {
                    // Grab the row from rhs
                    let r_slice = &rhs.stream()[rhs_start..rhs_end];

                    let dot_product = dot_product_simd3(&l_slice, &r_slice);
                    partition_values.push(dot_product);

                    rhs_start = rhs_end;
                    rhs_end += rhs_stride;
                }

                lhs_start = lhs_end;
                lhs_end += lhs_stride;
            }
            partition_values
        };

        let values = partition_strategy.parallelized(inner_process);
        Self::new(Shape::d2(lhs_rows, rhs_rows), values)
    }

    /// Multiplies each element in lhs with the corresponding rhs element.
    /// Does not require the same shape, only that the size of both tensors are the same.
    ///  * May lead to unexpected behavior, will change if needed.
    pub fn mul_element_wise_simd(&self, rhs: &Tensor) -> Self {
        let partition_strategy = &Partitioner::with_partitions_simd(
            self.shape.size(), 
            thread::available_parallelism().unwrap().get());

        let inner_process = |partition: &Partition| {
            let mut partition_values: Vec<f32> = Vec::with_capacity(partition.size());
            partition.binary_simd(
                &mut partition_values, 
                &self.stream(),
                &rhs.stream(),
                |x_simd, y_simd| x_simd * y_simd,
                |x, y| x * y
            );

            partition_values
        };

        let values = partition_strategy.parallelized(inner_process);
        Self::new(self.shape.clone(), values)
    }

    /// Subtracts two tensors.
    pub fn sub_element_wise_simd(&self, rhs: &Tensor) -> Self {
        let partition_strategy = &Partitioner::with_partitions_simd(
            self.shape.size(), 
            thread::available_parallelism().unwrap().get());

        let inner_process = |partition: &Partition| {
            let mut partition_values: Vec<f32> = Vec::with_capacity(partition.size());
            partition.binary_simd(
                &mut partition_values, 
                &self.stream(),
                &rhs.stream(),
                |x_simd, y_simd| x_simd - y_simd,
                |x, y| x - y
            );

            partition_values
        };

        let values = partition_strategy.parallelized(inner_process);
        Self::new(self.shape.clone(), values)
    }

    /// Adds two tensors.
    pub fn add_element_wise_simd(&self, rhs: &Tensor) -> Self {
        let partition_strategy = &Partitioner::with_partitions_simd(
            self.shape.size(), 
            thread::available_parallelism().unwrap().get());

        let inner_process = |partition: &Partition| {
            let mut partition_values: Vec<f32> = Vec::with_capacity(partition.size());
            partition.binary_simd(
                &mut partition_values, 
                &self.stream(),
                &rhs.stream(),
                |x_simd, y_simd| x_simd + y_simd,
                |x, y| x + y
            );

            partition_values
        };

        let values = partition_strategy.parallelized(inner_process);
        Self::new(self.shape.clone(), values)
    }

    /// Multiplies every element in tensor by scalar and returns the result.
    /// Optimized via multi-threading and SIMD.
    pub fn scale_simd(&self, scalar: f32) -> Self {
        let partitioner = Partitioner::with_partitions_simd(
            self.shape.size(), 
            thread::available_parallelism().unwrap().get());

        let y_simd = Simd::<f32, ALL_SIMD_LANES>::splat(scalar);
        let inner_process = |partition: &Partition| {
            let mut partition_values: Vec<f32> = Vec::with_capacity(partition.size());        
            partition.unary_simd(
                &mut partition_values, 
                &self.stream(),
                |x_simd| x_simd * y_simd,
                |x| x * scalar
            );

            partition_values
        };

        let values = partitioner.parallelized(inner_process);
        Self::new(self.shape.clone(), values)
    }

    /// **Unstable**
    /// Assumes `self` is a matrix tensor.
    /// Assumes rhs is a vector that has same number of elements as columns in tensor.
    pub fn broadcast_vector_add(&self, rhs: &Tensor) -> Self {
        let row_axis = self.shape.len() - 2;
        let rows = self.shape.dims()[row_axis];
        let &columns = self.shape.dims().last().unwrap();
        //assert_eq!(rhs.rows, 1, "Rhs tensor must have 1 row.");
        //assert_eq!(self.columns, rhs.columns, "Lhs and rhs must have equal number of columns.");

        let partition_strategy = &Partitioner::with_partitions(
            rows, 
            thread::available_parallelism().unwrap().get());

        let rs = rhs.stream(); 
        let inner_process = move |partition: &Partition| {
            let mut partition_values= Vec::with_capacity(partition.size() * columns);

            for row in partition.range() {
                let ls = self.dim_slice(row_axis, row);

                for column in 0..ls.len() {
                    partition_values.push(ls[column] + rs[column]);
                }
            }

            partition_values
        };

        let values = partition_strategy.parallelized(inner_process);
        //Self::from(self.row_count(), self.column_count(), values)
        Self::new(Shape::d2(rows, columns), values)
    }

    /// Reduces tensor to 1 row by adding each column's element. \
    /// **Unstable**
    pub fn reduce_vector_add(&self) -> Self {
        let &columns = self.shape.dims().last().unwrap();
        let rows = self.shape.dims()[self.shape.len() - 2];

        let partition_strategy = &Partitioner::with_partitions(
            columns, 
            thread::available_parallelism().unwrap().get());

        let inner_process =move |partition: &Partition| {
            let mut partition_values = Vec::with_capacity(partition.size());
            for column in partition.range() {
                let mut accumulator = 0.;
                for row in 0..rows {
                    accumulator += self.values[column + row * columns];
                }

                partition_values.push(accumulator);
            }

            partition_values
        };

        let values = partition_strategy.parallelized(inner_process);
        Self::new(Shape::d2(1, columns), values)
    }

    /// A Batch valid cross-correlation ensure that the output is smaller than the input.
    /// * `self`: Assumed shape is (batches, image height, image width, input channels) where inpput channels is 1 for grey scale, 3 for rgb, 4 for rgba.
    /// * `filters`: Assumed shape is (output channels, kernel height, kernel width, input channels).
    /// * `Output`: Shape is (batches, output channels, output height, output width). \
    /// **Using more than 1 channel and/or filter sizes larger than 3x3, takes better advantage of SIMD.**
    pub fn batch_valid_cross_correlation_simd(&self, filters: &Tensor) -> Self {
        let o_rows = self.shape[1] - filters.shape[1] + 1;
        let o_columns = self.shape[2] - filters.shape[2] + 1;

        let partitioner = &Partitioner::with_partitions_simd(
            self.shape[0],
            thread::available_parallelism().unwrap().get());

        let inner_process = move |parition: &Partition| {
            let mut partition_values = Vec::with_capacity(parition.size() * filters.shape.size());
            for batch_index in parition.range() {
                for filter_index in 0..filters.shape[0] {
                    for row in 0..o_rows {
                        for column in 0..o_columns {
                            let mut c_accum = 0.;

                            // Since we are doing optimized dot_products, only need to move down the rows
                            for kernel_row in 0..filters.shape[1] {
                                let start = self.shape.index_at(&Shape::new(vec![batch_index, row + kernel_row, column, 0]));
                                let end = start + filters.shape[2];
                                let input = &self.values[start..end];

                                let start = filters.shape.index_at(&Shape::new(vec![filter_index, kernel_row, 0, 0]));
                                let end = start + filters.shape[2];
                                let filter = &filters.values[start..end];

                                c_accum += dot_product_simd3(input, filter);
                            }

                            partition_values.push(c_accum);
                        }
                    }
                }
            }

            partition_values
        };

        let values = partitioner.parallelized(inner_process);
        Self::new(
            Shape::new(vec![self.shape[0], filters.shape[0], o_rows, o_columns]),
            values
        )
    }

    pub fn full_outer_convolution(&self, filters: &Tensor) -> Self {
        filters.shape.len();
        todo!()
    }

    /// Maxpool of tensor.
    /// **In Progress, not finished**
    /// `self`: Assume shape of (batches, filters * rows * columns)
    /// `result`: Assumes shape of (batch, filters * rows * columns)
    pub fn maxpool2d(&self,
        filters: usize,
        stride: usize,
        i_d: &Dimensions,
        p_d: &Dimensions,
        o_d: &Dimensions
    ) -> (Self, Vec<usize>) {
        let batches = self.shape[0];

        let partitioner = &Partitioner::with_partitions(
            batches,
            thread::available_parallelism().unwrap().get());

        let (rows, columns) = o_d.shape();
        let rows_x_columns = rows * columns;

        let inner_process = move |partition: &Partition| {
            let mut partition_values = Vec::with_capacity(partition.size() * filters * rows_x_columns);
            for batch in partition.range() {
                let input_row = self.dim_slice(0, batch);
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

        (Tensor::new(Shape::d4(batches, filters, rows, columns), values), max_indices)
    }
}

#[cfg(test)]
mod tests {
    use crate::prettify::*;

    use super::*;

    #[test]
    fn test_mutating_tensor_index() {
        let mut tc = Tensor::vector(vec![1., 2., 3.]);
        tc[0] = 5.;

        let expected = vec![5., 2., 3.];
        assert_eq!(tc.values, expected);

        tc[1] += 5.;
        assert_eq!(tc.values, vec![5., 7., 3.]);
    }

    #[test]
    fn test_contiguousness_by_index() {
        let x = Tensor::new(
            Shape::d3(4, 3, 2), vec![
            1., 2.,
            3., 4.,
            5., 6.,

            10., 11.,
            12., 13.,
            14., 15.,

            100., 200.,
            300., 400.,
            500., 600.,

            -1., 0.4,
            32., 6.,
            23., 34.
        ]);

        assert_eq!(x[1], 2.);
        assert_eq!(x[2], 3.);
        assert_eq!(x[6], 10.);
        assert_eq!(x[13], 200.);
    }

    #[test]
    fn test_reshape() {
        let t = Tensor::vector(vec![0., 1., 2., 3., 4., 5.]);
        let actual = t.reshape(Shape::d2(3, 2));
        let _ = actual.reshape(Shape::d3(1, 2, 3));
    }

    #[test]
    #[should_panic]
    fn test_reshape_invalid() {
        let t = Tensor::vector(vec![0., 1., 2., 3., 4.,]);
        let _ = t.reshape(Shape::d2(3, 2));
    }

    #[test]
    fn test_slice() {
        let tc = Tensor::new(
            Shape::new(vec![10, 3, 4]),
            (0..120).map(|x| x as f32).collect::<Vec<f32>>()
        );

        let actual = tc.slice(
            Shape::new(
                vec![1, 0, 1]),
            Shape::new(vec![2, 0, 0]));
        println!("{:?}", actual);
    }

    #[test]
    fn test_relu_and_d_relu() {
        let tc = Tensor::matrix(2, 16, vec![
            -3., 5., -9., 10., -8., 4., 7., -10., 3., 1., -5., 8., 2., -4., 6., 9.,
            -2., 6., -6., 0., 10., -1., 7., -7., -3., 3., 4., -10., -9., 1., 2., 5.
        ]);

        let expected = Tensor::matrix(2, 16, vec![
            0., 5., 0., 10., 0., 4., 7., 0., 3., 1., 0., 8., 2., 0., 6., 9.,
            0., 6., 0., 0., 10., 0., 7., 0., 0., 3., 4., 0., 0., 1., 2., 5.
        ]);

        let actual = tc.relu_simd();
        assert_eq!(actual, expected);

        let actual = tc.d_relu_simd();
        let expected = Tensor::matrix(2, 16, vec![
            0., 1., 0., 1., 0., 1., 1., 0., 1., 1., 0., 1., 1., 0., 1., 1.,
            0., 1., 0., 0., 1., 0., 1., 0., 0., 1., 1., 0., 0., 1., 1., 1.
        ]);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_transpose() {
        let m = Tensor::matrix(5, 4, vec![
            0f32, 1f32, 2f32, 3f32,
            4f32, 5f32, 6f32, 7f32,
            8f32, 9f32, 10f32, 11f32,
            12f32, 13f32, 14f32, 15f32,
            16f32, 17f32, 18f32, 19f32
        ]);

        let expected = Tensor::matrix(4, 5, vec![
            0f32, 4f32, 8f32, 12f32, 16f32,
            1f32, 5f32, 9f32, 13f32, 17f32,
            2f32, 6f32, 10f32, 14f32, 18f32,
            3f32, 7f32, 11f32, 15f32, 19f32
        ]);
        
        let actual = m.transpose();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_mul_transpose_simd() {
        // Test for if parallelism is between 2 and 4
        let lhs = Tensor::new(
            Shape::new(vec![1, 4, 3]),
            vec![
                1., 2., 3.,
                4., 5., 6.,
                7., 8., 9.,
                10., 11., 12.
            ]
        );

        // Assume already transposed.
        let rhs = Tensor::new(
            Shape::new(vec![1, 1, 5, 3]),
            vec![
                1., 6., 11.,
                2., 7., 12.,
                3., 8., 13.,
                4., 9., 14.,
                5., 10., 15.
            ]
        );

        let expected = Tensor::new(
            Shape::d2(4, 5),
            vec! [
                46., 52., 58., 64., 70.,
                100., 115., 130., 145., 160.,
                154., 178., 202., 226., 250.,
                208., 241., 274., 307., 340.
            ]
        );

        let actual = lhs.mul_transpose_simd(&rhs);
        assert_eq!(actual, expected);

        // Test for 16 threads
        let lhs = Tensor::new(
            Shape::new(vec![1, 16, 2]),
            vec![
                1., 2.,
                3., 4.,
                5., 6., 
                7., 8.,
                9., 10.,
                11., 12.,
                13., 14.,
                15., 16.,
                1., 2.,
                3., 4.,
                5., 6., 
                7., 8.,
                9., 10.,
                11., 12.,
                13., 14.,
                15., 16.,
            ]
        );

        // Assume already transposed.
        let rhs = Tensor::new(
            Shape::new(vec![1, 1, 3, 2]),
            vec![
                1., 4.,
                2., 5.,
                3., 6.
            ]
        );

        let expected = Tensor::new(
            Shape::d2(16, 3),
            vec! [
                9., 12., 15.,
                19., 26., 33.,
                29., 40., 51.,
                39., 54., 69.,
                49., 68., 87.,
                59., 82., 105.,
                69., 96., 123.,
                79., 110., 141.,
                9., 12., 15.,
                19., 26., 33.,
                29., 40., 51.,
                39., 54., 69.,
                49., 68., 87.,
                59., 82., 105.,
                69., 96., 123.,
                79., 110., 141.            ]
        );

        let actual = lhs.mul_transpose_simd(&rhs);
        println!("{BRIGHT_GREEN}{:?}{RESET}", actual);
        assert_eq!(actual, expected);

    }

    #[test]
    fn test_batch_valid_cross_correlation() {
        // Test two different images
        let inputs = Tensor::new(
            Shape::new(vec![2, 4, 4, 1]), 
            vec![
                1., 2., 3., 4., 
                5., 6., 7., 8.,
                9., 10., 11., 12.,
                13., 14., 15., 16.,

                10., 20., 30., 40.,
                50., 60., 70., 80.,
                90., 100., 110., 120.,
                130., 140., 150., 160.
        ]);

        let filters = Tensor::new(
            Shape::new(vec![3, 3, 3, 1]),
            vec![
                0., 0.15, 0.,
                0.15, 0.4, 0.15,
                0., 0.15, 0., 

                0., 0.10, 0.,
                0.10, 0.6, 0.10,
                0., 0.10, 0.,

                0., 0., 0.,
                0., 2., 0.,
                0., 0., 0.
            ]);


        let output = inputs.batch_valid_cross_correlation_simd(&filters);
        println!("{BRIGHT_CYAN}{:?}{RESET}", output);
    }

    #[test]
    fn test_scale_simd() {
        // Test for single thread, and not enough data to fill ALL_SIMD_LANES
        let tc = Tensor::vector(vec![10., 100., 0., 20.]);
        let actual = tc.scale_simd(10.);
        let expected = Tensor::vector(vec![100., 1000., 0., 200.]);
        assert_eq!(actual, expected);

        // Test for checking overflow works on 16 ALL_SIMD_LANES
        let tc = Tensor::matrix(2, 17, vec![
            2., 6., 10., 5., 7., 1., 3., 1., 7., 4., 3., 9., 7., 4., 6., 1., 10.,
            4., 9., 5., 4., 2., 4., 4., 5., 5., 2., 4., 4., 4., 2., 2., 2., 4.
        ]);
        let actual = tc.scale_simd(20.);
        let expected = Tensor::matrix(2, 17, vec![
            40., 120., 200., 100., 140., 20., 60., 20., 140., 80., 60., 180., 140., 80., 120., 20., 200.,
            80., 180., 100., 80., 40., 80., 80., 100., 100., 40., 80., 80., 80., 40., 40., 40., 80.
        ]);

        assert_eq!(actual, expected);

        // Great test for testing against 16 parallel threads on 16 ALL_SIMD_LANES
        let tc = Tensor::matrix(16, 16, vec![
            2., 6., 10., 5., 7., 1., 3., 1., 7., 4., 3., 9., 7., 4., 6., 1.,
            10., 7., 4., 10., 3., 2., 10., 7., 4., 8., 10., 6., 10., 8., 1., 4.,
            4., 1., 9., 1., 9., 8., 10., 9., 1., 1., 8., 8., 1., 7., 10., 6.,
            1., 6., 8., 1., 8., 9., 3., 10., 1., 4., 5., 4., 4., 7., 10., 5.,
            5., 1., 10., 5., 1., 9., 5., 10., 10., 5., 1., 1., 2., 1., 4., 8.,
            6., 4., 9., 5., 4., 2., 4., 4., 5., 5., 2., 4., 4., 4., 2., 2.,
            2., 4., 8., 4., 1., 2., 6., 1., 2., 3., 3., 8., 5., 1., 5., 1.,
            1., 7., 2., 7., 5., 3., 5., 1., 9., 5., 8., 7., 4., 1., 4., 1.,
            10., 5., 2., 2., 6., 10., 8., 3., 10., 5., 10., 9., 7., 5., 10., 7.,
            4., 9., 6., 3., 5., 10., 4., 1., 3., 4., 3., 1., 10., 7., 8., 1.,
            6., 9., 10., 8., 6., 4., 7., 4., 10., 7., 2., 9., 8., 8., 6., 9.,
            2., 10., 8., 8., 7., 6., 7., 1., 1., 4., 2., 5., 1., 3., 10., 2.,
            9., 9., 9., 6., 4., 5., 1., 4., 10., 6., 9., 2., 6., 9., 7., 3.,
            3., 5., 4., 8., 4., 9., 8., 6., 2., 8., 4., 6., 8., 2., 9., 1.,
            9., 5., 5., 10., 10., 8., 10., 3., 10., 2., 6., 7., 6., 5., 7., 6.,
            5., 3., 3., 10., 3., 4., 1., 7., 3., 3., 9., 1., 1., 5., 1., 7.
        ]);
        
        let actual = tc.scale_simd(10.);
        let expected = Tensor::matrix(16, 16, vec![
            20., 60., 100., 50., 70., 10., 30., 10., 70., 40., 30., 90., 70., 40., 60., 10.,
            100., 70., 40., 100., 30., 20., 100., 70., 40., 80., 100., 60., 100., 80., 10., 40.,
            40., 10., 90., 10., 90., 80., 100., 90., 10., 10., 80., 80., 10., 70., 100., 60.,
            10., 60., 80., 10., 80., 90., 30., 100., 10., 40., 50., 40., 40., 70., 100., 50.,
            50., 10., 100., 50., 10., 90., 50., 100., 100., 50., 10., 10., 20., 10., 40., 80.,
            60., 40., 90., 50., 40., 20., 40., 40., 50., 50., 20., 40., 40., 40., 20., 20.,
            20., 40., 80., 40., 10., 20., 60., 10., 20., 30., 30., 80., 50., 10., 50., 10.,
            10., 70., 20., 70., 50., 30., 50., 10., 90., 50., 80., 70., 40., 10., 40., 10.,
            100., 50., 20., 20., 60., 100., 80., 30., 100., 50., 100., 90., 70., 50., 100., 70.,
            40., 90., 60., 30., 50., 100., 40., 10., 30., 40., 30., 10., 100., 70., 80., 10.,
            60., 90., 100., 80., 60., 40., 70., 40., 100., 70., 20., 90., 80., 80., 60., 90.,
            20., 100., 80., 80., 70., 60., 70., 10., 10., 40., 20., 50., 10., 30., 100., 20.,
            90., 90., 90., 60., 40., 50., 10., 40., 100., 60., 90., 20., 60., 90., 70., 30.,
            30., 50., 40., 80., 40., 90., 80., 60., 20., 80., 40., 60., 80., 20., 90., 10.,
            90., 50., 50., 100., 100., 80., 100., 30., 100., 20., 60., 70., 60., 50., 70., 60.,
            50., 30., 30., 100., 30., 40., 10., 70., 30., 30., 90., 10., 10., 50., 10., 70.
        ]);

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_add_element_wise_simd() {
        let lhs = Tensor::new(
            Shape::d2(4, 4),
            vec![
                0., 1., 2., 3.,
                4., 5., 6., 7.,
                8., 9., 10., 11.,
                12., 13., 14., 15.
            ]
        );

        let rhs = Tensor::new(
            Shape::d2(4, 4), vec![
                0., 1., 1., 1.,
                2., 3., 2., 2.,
                3., 3., 1., 4.,
                5., 4., 3., 2.
            ]
        );

        let actual = lhs.add_element_wise_simd(&rhs);
        let expected = Tensor::new(
            Shape::d2(4, 4),
            vec![
                0., 2., 3., 4.,
                6., 8., 8., 9.,
                11., 12., 11., 15.,
                17., 17., 17., 17.
            ]
        );

        assert_eq!(actual, expected);
    }
    
    #[test]
    fn test_sub_element_wise_simd() {
        let lhs = Tensor::new(
            Shape::d2(4, 4),
            vec![
                0., 1., 2., 3.,
                4., 5., 6., 7.,
                8., 9., 10., 11.,
                12., 13., 14., 15.
            ]
        );

        let rhs = Tensor::new(
            Shape::d2(4, 4), vec![
                0., 1., 1., 1.,
                2., 3., 2., 2.,
                3., 3., 1., 4.,
                5., 4., 3., 2.
            ]
        );

        let actual = lhs.sub_element_wise_simd(&rhs);
        let expected = Tensor::new(
            Shape::d2(4, 4),
            vec![
                0., 0., 1., 2.,
                2., 2., 4., 5.,
                5., 6., 9., 7.,
                7., 9., 11., 13.
            ]
        );

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_element_wise_mul_simd() {
        let lhs = Tensor::new(
            Shape::d2(4, 4),
            vec![
                0., 1., 2., 3.,
                4., 5., 6., 7.,
                8., 9., 10., 11.,
                12., 13., 14., 15.
            ]
        );

        let rhs = Tensor::new(
            Shape::d2(4, 4), vec![
                0., 1., 1., 1.,
                2., 3., 2., 2.,
                3., 3., 1., 4.,
                5., 4., 3., 2.
            ]
        );

        let actual = lhs.mul_element_wise_simd(&rhs);
        let expected = Tensor::new(
            Shape::d2(4, 4),
            vec![
                0., 1., 2., 3.,
                8., 15., 12., 14.,
                24., 27., 10., 44.,
                60., 52., 42., 30.
            ]
        );

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_reduce_vector_add() {
        let tc = Tensor::matrix(3, 4, vec![
            1., 1., 2., 10.,
            2., 3., 3., 10.,
            4., 4., 5., 10.
        ]);

        let expected = Tensor::matrix(1, 4, vec![
                7., 8., 10., 30.
            ]);

        let actual2 = tc.reduce_vector_add();
        assert_eq!(actual2, expected);
    }
    
    #[test]
    fn test_broadcast_add_vector() {
        let tc = Tensor::matrix(3, 4, vec![
                0., 0., 0., 0.,
                1., 1., 1., 1.,
                2., 2., 2., 2.
            ]);

        let row_to_add = Tensor::vector(vec![10., 20., 30., 40.]);

        let expected = Tensor::new(Shape::d2(3, 4), vec![
                10., 20., 30., 40.,
                11., 21., 31., 41.,
                12., 22., 32., 42.
            ]);

        let actual = tc.broadcast_vector_add(&row_to_add);
        assert_eq!(actual, expected);        
    }

    #[test]
    fn test_maxpool2d(){
        let tc = Tensor::new(Shape::d4(1, 2, 4, 4), vec![
            1., 3., 2., 1.,  
            4., 2., 1., 5.,
            3., 1., 4., 2.,
            8., 6., 7., 9.,

            6., 4., 3., 8.,
            2., 4., 3., 7.,
            1., 5., 4., 3.,
            4., 7., 6., 4.
        ]);

        let o_d = Dimensions {
            height: (4 - 2) / 2 + 1,
            width: (4 - 2) / 2 + 1
        };

        let (pooled, max_indices) = tc.maxpool2d(
            2,
            2,
            &Dimensions { width: 4, height: 4 },
            &Dimensions { width: 2, height: 2 },
            &o_d);

        let expected = Tensor::new(Shape::d4(1, 2, 2, 2), vec![
            4.0, 5.0,
            8.0, 9.0,
            
            6.0, 8.0,
            7.0, 6.0]);

        let expected_indices = vec![4, 7, 12, 15, 16, 19, 29, 30];

        println!("{BRIGHT_GREEN}Tensor maxpool {:?}{RESET}", pooled);
        assert_eq!(pooled, expected);
        assert_eq!(max_indices, expected_indices);

        // // In use for backprop
        // let dvalues = Tensor::matrix(1, 2 * 4, vec![
        //     0.2, -0.5,
        //     0.3, 0.1,

        //     -1., 4.,
        //     3., 8.
        // ]);

        // assert_eq!(pooled, dvalues);
        
    }
}