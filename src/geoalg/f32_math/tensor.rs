use std::thread;
use crate::partitions::{Partition, Partitioner};

use super::{shape::Shape, simd_extensions::dot_product_simd3};

#[derive(Debug)]
pub struct Tensor {
    shape: Shape,
    values: Vec<f32>
}

impl Tensor {
    /// Creates a new tensor with specified shape.
    pub fn new(shape: Shape, values: Vec<f32>) -> Self { 
        Tensor { shape, values } 
    }

    /// Creates a tensor specified by values, the shapes is 1-d with size set to values.len().
    pub fn vector(values: Vec<f32>) -> Self {
        Tensor::new(Shape::d1(values.len()), values)
    }

    /// Creates a tensor that is a row-major matrix with supplied values.
    pub fn matrix(rows: usize, columns: usize, values: Vec<f32>) -> Self {
        Tensor::new(Shape::d2(rows, columns), values)
    }

    /// Gets a contiguous subset of values.
    /// Experimental.
    pub fn slice(&self, start: Shape, end: Shape) -> &[f32] {
        let start = self.shape.index_at(&start);
        let end = self.shape.index_at(&end);

        &self.values[start..end]
    }

    /// Lhs: Assumed shape is (batches, image height, image width, input channels) where inpput channels is 1 for grey scale, 3 for rgb, 4 for rgba
    /// filters: Assumed shape is (output channels, kernel height, kernel width, input channels)
    /// Output: Shape is (batches, output channels, output height, output width)
    /// When using more than 1 channel and larger filter sizes, takes better advantage of SIMD
    pub fn batch_valid_cross_correlation(&self, filters: &Tensor) -> Self {
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

                            // Slide kernel window horizontally and then vertically
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
    use colored::Colorize;

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


        let output = inputs.batch_valid_cross_correlation(&filters);
        let conv_result = format!("{:?}", output).bright_cyan();
        println!("{conv_result}");
    }
}