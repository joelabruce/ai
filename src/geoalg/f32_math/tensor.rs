use std::{ops::Index, simd::Simd, thread};
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

    pub fn new_randomized_uniform(shape: Shape, uniform: Uniform<f32>) -> Self {
        let mut rng = rand::thread_rng();
        let element_count = shape.size();
        let values = uniform.sample_iter(&mut rng).take(element_count).collect();

        Tensor::new(shape, values)
    }

    pub fn new_randomized_normal(shape: Shape, normal: Normal<f32>) -> Self {
        let mut rng = rand::thread_rng();
        let element_count = shape.size();        
        let values = normal.sample_iter(&mut rng).take(element_count).collect();

        Tensor::new(shape, values)
    }

    /// Creates a tensor specified by values, the shapes is 1-d with size set to values.len().
    pub fn vector(values: Vec<f32>) -> Self {
        Tensor::new(Shape::d1(values.len()), values)
    }

    /// Creates a tensor that is a row-major matrix with supplied values.
    pub fn matrix(rows: usize, columns: usize, values: Vec<f32>) -> Self {
        Tensor::new(Shape::d2(rows, columns), values)
    }

    /// Get a stream to underlying values.
    pub fn stream(&self) -> &[f32] {
        &self.values
    }

    /// Gets a contiguous subset of values.
    /// Experimental.
    pub fn slice(&self, start: Shape, end: Shape) -> &[f32] {
        let start = self.shape.index_at(&start);
        let end = self.shape.index_at(&end);

        &self.values[start..end]
    }

    pub fn mul_element_wise_simd(&self, rhs: &Tensor) -> Self {
        let partition_strategy = &Partitioner::with_partitions_simd(
            self.shape.size(), 
            thread::available_parallelism().unwrap().get());

            let inner_process = |partition: &Partition| {
                let mut partition_values: Vec<f32> = Vec::with_capacity(partition.get_size());
    
                // Avoids doing division and unnecessary multiplications
                let return_slice: &mut Vec<f32> = &mut vec![0.; SIMD_LANES];            
                let mut cursor = partition.get_start();
                while cursor + SIMD_LANES <= partition.get_end() {
                    let range = cursor..cursor + SIMD_LANES;
                    let x_simd = Simd::<f32, SIMD_LANES>::from_slice(&self.stream()[range.clone()]);
                    let y_simd = Simd::<f32, SIMD_LANES>::from_slice(&rhs.stream()[range.clone()]);
    
                    let r_simd = x_simd * y_simd;
    
                    r_simd.copy_to_slice(return_slice);
                    partition_values.extend_from_slice(return_slice);
                    cursor += SIMD_LANES;
                }
    
                // Checks to see if there are any remainder chunks to deal with
                if cursor > partition.get_end() { cursor -= SIMD_LANES; }
    
                // Does normal multiplication for remaining elements that cannot fit into simd.
                // If using Partitioner::with_partitions_simd, this should only execute at most 1 time for the last thread.
                for i in cursor..=partition.get_end() {
                    partition_values.push(self[i] * rhs[i]);
                }
    
                partition_values
            };

        let values = partition_strategy.parallelized(inner_process);

        Self::new(self.shape.clone(), values)
    }

    /// Lhs: Assumed shape is (batches, image height, image width, input channels) where inpput channels is 1 for grey scale, 3 for rgb, 4 for rgba
    /// filters: Assumed shape is (output channels, kernel height, kernel width, input channels)
    /// Output: Shape is (batches, output channels, output height, output width)
    /// When using more than 1 channel and larger filter sizes, takes better advantage of SIMD
    pub fn batch_valid_cross_correlation_simd(&self, filters: &Tensor) -> Self {
        let o_rows = self.shape.axis_len(1) - filters.shape.axis_len(1) + 1;
        let o_columns = self.shape.axis_len(2) - filters.shape.axis_len(2) + 1;

        let partitioner = &Partitioner::with_partitions_simd(
            self.shape.axis_len(0),
            thread::available_parallelism().unwrap().get());

        let inner_process = move |parition: &Partition| {
            let mut partition_values = Vec::with_capacity(parition.get_size() * filters.shape.size());
            for batch_index in parition.get_range() {
                for filter_index in 0..filters.shape.axis_len(0) {
                    for row in 0..o_rows {
                        for column in 0..o_columns {
                            let mut c_accum = 0.;

                            // Since we are doing optimized dot_products, only need to move down the rows
                            for kernel_row in 0..filters.shape.axis_len(1) {
                                let start = self.shape.index_at(&Shape::new(vec![batch_index, row + kernel_row, column, 0]));
                                let end = start + filters.shape.axis_len(2);
                                let input = &self.values[start..end];

                                let start = filters.shape.index_at(&Shape::new(vec![filter_index, kernel_row, 0, 0]));
                                let end = start + filters.shape.axis_len(2);
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
            Shape::new(vec![self.shape.axis_len(0), filters.shape.axis_len(0), o_rows, o_columns]),
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