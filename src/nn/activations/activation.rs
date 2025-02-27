use std::{simd::{cmp::SimdPartialOrd, num::SimdFloat, Simd}, thread};

use crate::{geoalg::f32_math::{matrix::Matrix, simd_extensions::ALL_SIMD_LANES}, partition::Partition, partitioner::Partitioner};

pub struct Activation {
    f: fn(&Matrix) -> Matrix,
    b: fn(&Matrix, &Matrix) -> Matrix
}

impl Activation {
    pub fn forward(&self, inputs: &Matrix) -> Matrix {
        (self.f)(inputs)
    }

    pub fn backward(&self, dvalues: &Matrix, inputs: &Matrix) -> Matrix {
        (self.b)(dvalues, inputs)
    }
}

pub const RELU: Activation = Activation {
    f: |inputs| -> Matrix {
        relu_simd(inputs)
    },
    b: |dvalues, inputs| -> Matrix {
        d_relu_simd(inputs).mul_element_wise_simd(&dvalues)
    }
};

/// Performs relu on each value in Tensor.
pub fn relu_simd(inputs: &Matrix) -> Matrix {
    let partitioner = Partitioner::with_partitions_simd(
        inputs.len(), 
        thread::available_parallelism().unwrap().get());

    let y_simd = Simd::<f32, ALL_SIMD_LANES>::splat(0.);
    let inner_process = |partition: &Partition| {
        let mut partition_values: Vec<f32> = Vec::with_capacity(partition.size());
        partition.unary_simd(
            &mut partition_values, 
            &inputs.read_values(),
            |x_simd| x_simd.simd_max(y_simd),
            |x| x.max(0.)
        );

        partition_values
    };

    let values = partitioner.parallelized(inner_process);
    Matrix::new(inputs.row_count(), inputs.column_count(), values)
}

/// Derivative of relu on each element in Tensor.
pub fn d_relu_simd(inputs: &Matrix) -> Matrix {
    let partitioner = Partitioner::with_partitions_simd(
        inputs.len(), 
        thread::available_parallelism().unwrap().get());

    let y_simd = Simd::<f32, ALL_SIMD_LANES>::splat(0.);
    let true_mask = Simd::<f32, ALL_SIMD_LANES>::splat(1.);
    let inner_process = |partition: &Partition| {
        let mut partition_values: Vec<f32> = Vec::with_capacity(partition.size());
        partition.unary_simd(
            &mut partition_values, 
            &inputs.read_values(),
            |x_simd| {
                x_simd
                    .simd_gt(y_simd)
                    .select(true_mask, y_simd)
            },
            |x| if x > 0. { 1. } else { 0. }
        );

        partition_values
    };

    let values = partitioner.parallelized(inner_process);
    Matrix::new(inputs.row_count(), inputs.column_count(), values)
}
