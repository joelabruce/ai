use crate::partitions::Partition;
use std::simd::{num::SimdFloat, *};
use super::matrix::*;

pub const SIMD_LANES: usize = 16;

/// Taken from github exmples.
pub fn dot_product_simd3(lhs: &[f32], rhs: &[f32]) -> f32 {
    assert_eq!(lhs.len(), rhs.len());

    let (a_extra, a_chunks) = lhs.as_rchunks();
    let (b_extra, b_chunks) = rhs.as_rchunks();

    // These are always true, but for emphasis:
    assert_eq!(a_chunks.len(), b_chunks.len());
    assert_eq!(a_extra.len(), b_extra.len());

    let mut sums = [0.0; SIMD_LANES];
    for ((x, y), d) in std::iter::zip(a_extra, b_extra).zip(&mut sums) {
        *d = x * y;
    }

    let mut sums = Simd::<f32, SIMD_LANES>::from_array(sums);
    std::iter::zip(a_chunks, b_chunks).for_each(|(x, y)| {
        sums += Simd::<f32, SIMD_LANES>::from_array(*x) * Simd::<f32, SIMD_LANES>::from_array(*y);
    });

    sums.reduce_sum()
}

/// Take from github examples.
/// This version is slower than not doing simd, why?
/// Too much reliance on abstractions??
pub fn dot_product_simd5(a: &[f32], b: &[f32]) -> f32 {
    a.array_chunks::<SIMD_LANES>()
        .map(|&a| Simd::<f32, SIMD_LANES>::from_array(a))
        .zip(b.array_chunks::<SIMD_LANES>().map(|&b| Simd::<f32, SIMD_LANES>::from_array(b)))
        .fold(Simd::<f32, SIMD_LANES>::splat(0.), |acc, (a, b)| a.mul_add(b, acc))
        .reduce_sum()
}

impl Matrix {
    pub fn mul_element_wise_simd(&self, rhs: &Matrix) -> Self {
        assert!(self.row_count() == rhs.row_count() && self.column_count() == rhs.column_count(), "When element-wise multiplying two matrices, they must have same order.");
        
        let partition = Partition::new(0, self.len());
        let inner_process = |partition: &Partition| {
            let return_slice: &mut Vec<f32> = &mut vec![0.; SIMD_LANES];

            let mut partition_values: Vec<f32> = Vec::with_capacity(partition.get_size());
            
            let iterations = partition.get_size() / SIMD_LANES;
            for i in 0..iterations {
                let cursor = i * SIMD_LANES;
                let range = cursor..cursor + SIMD_LANES;
                let x_simd = Simd::<f32, SIMD_LANES>::from_slice(&self.read_values()[range.clone()]);
                let y_simd = Simd::<f32, SIMD_LANES>::from_slice(&rhs.read_values()[range.clone()]);

                let r_simd = x_simd * y_simd;

               r_simd.copy_to_slice(return_slice);
               partition_values.extend_from_slice(&return_slice);
            }

            if iterations * SIMD_LANES < self.len() {
                for i in iterations * SIMD_LANES..self.len() {
                    partition_values.push(self.read_at(i) * rhs.read_at(i));
                }                
            }

            partition_values
        };

        let values = inner_process(&partition);

        Self::from(self.row_count(), self.column_count(), values)
    }
}

#[cfg(test)]
mod tests {
    use crate::geoalg::f32_math::{matrix::Matrix, optimized_functions::dot_product_of_vector_slices};

    use super::dot_product_simd3;

    #[test]
    fn test_element_wise_mul_simd() {
        let lhs = Matrix::from(4, 4, vec![
            0., 1., 2., 3.,
            4., 5., 6., 7.,
            8., 9., 10., 11.,
            12., 13., 14., 15.
        ]);

        let rhs = Matrix::from(4, 4, vec![
            0., 1., 1., 1.,
            2., 3., 2., 2.,
            3., 3., 1., 4.,
            5., 4., 3., 2.
        ]);

        let actual = lhs.mul_element_wise_simd(&rhs);
        let expected = Matrix::from(4, 4, vec![
            0., 1., 2., 3.,
            8., 15., 12., 14.,
            24., 27., 10., 44.,
            60., 52., 42., 30.
        ]);

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_dot_product_simd_long_vectors() {
        let lhs = vec![
            1., 2., 3., 4., 5., 6., 7., 8.,
            1., 2., 3., 4., 5., 6., 7., 8.,
            1., 1., 3., 4., 5., 6., 7., 8.,
        ];

        let rhs = vec![
            1., 2., 3., 4.,
            1., 2., 3., 4.,
            1., 2., 3., 4.,
            10., 11., 12., 13.,
            10., 11., 12., 13.,
            10., 11., 12., 13. 
        ];

        let expected = dot_product_of_vector_slices(&lhs, &rhs);
        let actual = dot_product_simd3(&lhs, &rhs);
    }
}