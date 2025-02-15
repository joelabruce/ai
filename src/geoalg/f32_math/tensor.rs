use std::{ops::Index, simd::{num::SimdFloat, Simd}, thread};
use rand_distr::{Distribution, Normal, Uniform};

use crate::partitions::{Partition, Partitioner};

use super::{shape::Shape, simd_extensions::{dot_product_simd3, SIMD_LANES}};

#[derive(Debug, PartialEq)]
pub struct Tensor {
    shape: Shape,
    values: Vec<f32>
}

impl Index<usize> for Tensor {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.values[index]
    }
}

impl Tensor {
    /// Creates a new tensor with specified shape.
    pub fn new(shape: Shape, values: Vec<f32>) -> Self { 
        Tensor { shape, values } 
    }

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

    /// Creates a vector tensor specified by values.
    pub fn vector(values: Vec<f32>) -> Self {
        Tensor::new(Shape::d1(values.len()), values)
    }

    /// Creates a row-major matrix tensor.
    pub fn matrix(rows: usize, columns: usize, values: Vec<f32>) -> Self {
        Tensor::new(Shape::d2(rows, columns), values)
    }

    /// Get a stream to underlying values.
    pub fn stream(&self) -> &[f32] { &self.values }

    /// Gets a contiguous subset of values.
    /// Experimental.
    pub fn slice(&self, start: Shape, end: Shape) -> &[f32] {
        let start = self.shape.index_at(&start);
        let end = self.shape.index_at(&end);

        &self.values[start..end]
    }

    /// Performs function on value for each value in Tensor.
    pub fn relu_simd(&self) -> Self {
        let partitioner = Partitioner::with_partitions_simd(
            self.shape.size(), 
            thread::available_parallelism().unwrap().get());

        let y_simd = Simd::<f32, SIMD_LANES>::splat(0.);
        let inner_process = |partition: &Partition| {
            let mut partition_values: Vec<f32> = Vec::with_capacity(partition.get_size());
        
            // Avoids doing division and unnecessary multiplications
            let return_slice: &mut Vec<f32> = &mut vec![0.; SIMD_LANES];
            let mut cursor_start = partition.get_start();
            let mut cursor_end = cursor_start + SIMD_LANES;
            while cursor_end <= partition.get_end() {
                let x_simd = Simd::<f32, SIMD_LANES>::from_slice(&self.stream()[cursor_start..cursor_end]);

                let r_simd = x_simd.simd_max(y_simd);

                r_simd.copy_to_slice(return_slice);
                partition_values.extend_from_slice(return_slice);

                cursor_start = cursor_end;
                cursor_end += SIMD_LANES;
            }

            // Checks to see if there are any remainder chunks to deal with
            if cursor_end > partition.get_end() { cursor_end -= SIMD_LANES; }

            // Does normal multiplication for remaining elements that cannot fit into simd.
            // If using Partitioner::with_partitions_simd, this should only execute at most 1 time for the last thread.
            for i in cursor_end..=partition.get_end() {
                partition_values.push(self[i].max(0.));
            }

            partition_values
        };

        let values = partitioner.parallelized(inner_process);
        Self::new(self.shape.clone(), values)
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
            let mut partition_values: Vec<f32> = Vec::with_capacity(partition.get_size() * rhs_rows);
            
            let mut lhs_start = partition.get_start() * lhs_stride;
            let mut lhs_end = lhs_start + lhs_stride;
            for _row in partition.get_range() {
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

    /// Multiplies each element in lhs with the rhs element.
    /// Does not require the same shape, only that the size of both tensors are the same.
    ///  * May lead to unexpected behavior, will change if needed.
    pub fn mul_element_wise_simd(&self, rhs: &Tensor) -> Self {
        let partition_strategy = &Partitioner::with_partitions_simd(
            self.shape.size(), 
            thread::available_parallelism().unwrap().get());

            let inner_process = |partition: &Partition| {
                let mut partition_values: Vec<f32> = Vec::with_capacity(partition.get_size());
    
                // Avoids doing division and unnecessary multiplications
                let return_slice: &mut Vec<f32> = &mut vec![0.; SIMD_LANES];            
                let mut cursor_start = partition.get_start();
                let mut cursor_end = cursor_start + SIMD_LANES;
                while cursor_end <= partition.get_end() {
                    let range = cursor_start..cursor_end;
                    let x_simd = Simd::<f32, SIMD_LANES>::from_slice(&self.stream()[range.clone()]);
                    let y_simd = Simd::<f32, SIMD_LANES>::from_slice(&rhs.stream()[range.clone()]);
    
                    let r_simd = x_simd * y_simd;
    
                    r_simd.copy_to_slice(return_slice);
                    partition_values.extend_from_slice(return_slice);

                    cursor_start = cursor_end;
                    cursor_end += SIMD_LANES;
                }
    
                // Checks to see if there are any remainder chunks to deal with
                if cursor_end > partition.get_end() { cursor_end -= SIMD_LANES; }
    
                // Does normal multiplication for remaining elements that cannot fit into simd.
                // If using Partitioner::with_partitions_simd, this should only execute at most 1 time for the last thread.
                for i in cursor_end..=partition.get_end() {
                    partition_values.push(self[i] * rhs[i]);
                }
    
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

        let y_simd = Simd::<f32, SIMD_LANES>::splat(scalar);
        let inner_process = |partition: &Partition| {
            let mut partition_values: Vec<f32> = Vec::with_capacity(partition.get_size());
        
            // Avoids doing division and unnecessary multiplications
            let return_slice: &mut Vec<f32> = &mut vec![0.; SIMD_LANES];
            let mut cursor_start = partition.get_start();
            let mut cursor_end = cursor_start + SIMD_LANES;
            while cursor_end <= partition.get_end() {
                let x_simd = Simd::<f32, SIMD_LANES>::from_slice(&self.stream()[cursor_start..cursor_end]);

                let r_simd = x_simd * y_simd;

                r_simd.copy_to_slice(return_slice);
                partition_values.extend_from_slice(return_slice);

                cursor_start = cursor_end;
                cursor_end += SIMD_LANES;
            }

            // Checks to see if there are any remainder chunks to deal with
            if cursor_end > partition.get_end() { cursor_end -= SIMD_LANES; }

            // Does normal multiplication for remaining elements that cannot fit into simd.
            // If using Partitioner::with_partitions_simd, this should only execute at most 1 time for the last thread.
            for i in cursor_end..=partition.get_end() {
                partition_values.push(self[i] * scalar);
            }

            partition_values
        };

        let values = partitioner.parallelized(inner_process);
        Self::new(self.shape.clone(), values)
    }

    /// Lhs: Assumed shape is (batches, image height, image width, input channels) where inpput channels is 1 for grey scale, 3 for rgb, 4 for rgba
    /// filters: Assumed shape is (output channels, kernel height, kernel width, input channels)
    /// Output: Shape is (batches, output channels, output height, output width)
    /// When using more than 1 channel and larger filter sizes, takes better advantage of SIMD
    pub fn batch_valid_cross_correlation_simd(&self, filters: &Tensor) -> Self {
        let o_rows = self.shape[1] - filters.shape[1] + 1;
        let o_columns = self.shape[2] - filters.shape[2] + 1;

        let partitioner = &Partitioner::with_partitions_simd(
            self.shape[0],
            thread::available_parallelism().unwrap().get());

        let inner_process = move |parition: &Partition| {
            let mut partition_values = Vec::with_capacity(parition.get_size() * filters.shape.size());
            for batch_index in parition.get_range() {
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
}

#[cfg(test)]
mod tests {
    use crate::prettify::*;

    use super::*;

    #[test]
    fn test_shape_contiguousness() {
        let _x = Tensor::new(
            Shape::new(vec![4, 3, 2]), vec![
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
    fn test_max() {
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
    }

    #[test]
    fn test_scale_simd() {
        // Test for single thread, and not enough data to fill SIMD_LANES
        let tc = Tensor::vector(vec![10., 100., 0., 20.]);
        let actual = tc.scale_simd(10.);
        let expected = Tensor::vector(vec![100., 1000., 0., 200.]);
        assert_eq!(actual, expected);

        // Test for checking overflow works on 16 SIMD_LANES
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

        // Great test for testing against 16 parallel threads on 16 SIMD_LANES
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
        println!("{BRIGHT_RED}{:?}{RESET}", actual);
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
}