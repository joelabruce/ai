use crate::{partition::Partition, partitioner::Partitioner};
use std::simd::{num::SimdFloat, *};
use super::matrix::*;

pub const ALL_SIMD_LANES: usize = 16;
//pub const HALF_SIMD_LANES: usize = 4;

/// Taken from github exmples.
pub fn dot_product_simd3(lhs: &[f32], rhs: &[f32]) -> f32 {
    //assert_eq!(lhs.len(), rhs.len());
    let (l_extra, l_chunks) = lhs.as_rchunks();
    let (r_extra, r_chunks) = rhs.as_rchunks();

    let mut sums = [0.0; ALL_SIMD_LANES];
    for ((x, y), d) in std::iter::zip(l_extra, r_extra).zip(&mut sums) {
        *d = x * y;
    }

    let mut sums = Simd::<f32, ALL_SIMD_LANES>::from_array(sums);
    std::iter::zip(l_chunks, r_chunks).for_each(|(x, y)| {
        sums += Simd::<f32, ALL_SIMD_LANES>::from_array(*x) * Simd::<f32, ALL_SIMD_LANES>::from_array(*y);
    });

    sums.reduce_sum()
}

/// Will overwrite contents of out.
pub fn mul_simd<'a>(lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
    let (l_extra, l_chunks) = lhs.as_rchunks();
    let (r_extra, r_chunks) = rhs.as_rchunks();

    let mut out = Vec::with_capacity(lhs.len());
    for i in 0..l_extra.len() {
        out.push(l_extra[i] * r_extra[i]);
    }

    std::iter::zip(l_chunks, r_chunks).for_each(|(x, y)| {
        out.extend(
            (Simd::<f32, ALL_SIMD_LANES>::from_array(*x) * Simd::<f32, ALL_SIMD_LANES>::from_array(*y)).as_array()
        );
    });

    out
}

// Taken from github examples.
// This version is slower than not doing simd, why?
// Too much reliance on abstractions??
// pub fn dot_product_simd5(a: &[f32], b: &[f32]) -> f32 {
//     a.array_chunks::<SIMD_LANES>()
//         .map(|&a| Simd::<f32, SIMD_LANES>::from_array(a))
//         .zip(b.array_chunks::<SIMD_LANES>().map(|&b| Simd::<f32, SIMD_LANES>::from_array(b)))
//         .fold(Simd::<f32, SIMD_LANES>::splat(0.), |acc, (a, b)| a.mul_add(b, acc))
//         .reduce_sum()
// }

impl Matrix {
    /// Now in Tensor
    pub fn scale_simd(&self, scalar: f32) -> Self {
        let partitioner = Partitioner::with_partitions_simd(self.len(), 16);

        let inner_process = |partition: &Partition| {
            let mut partition_values: Vec<f32> = Vec::with_capacity(partition.size());

            // Avoids doing division and unnecessary multiplications
            let return_slice: &mut Vec<f32> = &mut vec![0.; ALL_SIMD_LANES];            
            let mut cursor = partition.get_start();
            while cursor + ALL_SIMD_LANES <= partition.get_end() {
                let range = cursor..cursor + ALL_SIMD_LANES;
                let x_simd = Simd::<f32, ALL_SIMD_LANES>::from_slice(&self.read_values()[range.clone()]);
                let y_simd = Simd::<f32, ALL_SIMD_LANES>::splat(scalar);

                let r_simd = x_simd * y_simd;

                r_simd.copy_to_slice(return_slice);
                partition_values.extend_from_slice(return_slice);
                cursor += ALL_SIMD_LANES;
            }

            // Checks to see if there are any remainder chunks to deal with
            if cursor > partition.get_end() { cursor -= ALL_SIMD_LANES; }

            // Does normal multiplication for remaining elements that cannot fit into simd.
            // If using Partitioner::with_partitions_simd, this should only execute at most 1 time for the last thread.
            for i in cursor..=partition.get_end() {
                partition_values.push(self[i] * scalar);
            }

            partition_values
        };

        let values = partitioner.parallelized(inner_process);
        Self::new(self.row_count(), self.column_count(), values)
    }

    /// Now in Tensor.
    pub fn mul_element_wise_simd(&self, rhs: &Matrix) -> Self {
        assert!(self.row_count() == rhs.row_count() && self.column_count() == rhs.column_count(), "When element-wise multiplying two matrices, they must have same order.");
        
        let partitioner = Partitioner::with_partitions_simd(self.len(), 16);

        let inner_process = |partition: &Partition| {
            let mut partition_values: Vec<f32> = Vec::with_capacity(partition.size());

            // Avoids doing division and unnecessary multiplications
            let return_slice: &mut Vec<f32> = &mut vec![0.; ALL_SIMD_LANES];            
            let mut cursor = partition.get_start();
            while cursor + ALL_SIMD_LANES <= partition.get_end() {
                let range = cursor..cursor + ALL_SIMD_LANES;
                let x_simd = Simd::<f32, ALL_SIMD_LANES>::from_slice(&self.read_values()[range.clone()]);
                let y_simd = Simd::<f32, ALL_SIMD_LANES>::from_slice(&rhs.read_values()[range.clone()]);

                let r_simd = x_simd * y_simd;

                r_simd.copy_to_slice(return_slice);
                partition_values.extend_from_slice(return_slice);
                cursor += ALL_SIMD_LANES;
            }

            // Checks to see if there are any remainder chunks to deal with
            if cursor > partition.get_end() { cursor -= ALL_SIMD_LANES; }

            // Does normal multiplication for remaining elements that cannot fit into simd.
            // If using Partitioner::with_partitions_simd, this should only execute at most 1 time for the last thread.
            for i in cursor..=partition.get_end() {
                partition_values.push(self[i] * rhs[i]);
            }

            partition_values
        };

        let values = partitioner.parallelized(inner_process);
        Self::new(self.row_count(), self.column_count(), values)
    }
}

impl Partitioner {
    /// Optimizes partitioner to fill SIMD lanes as full as possible across specified partition_count.
    /// Will only multi-thread if count >= partition_count * SIMD_LANES.
    /// Front-loads as many SIMDs as possible, puts remainder in last thread since theoretically it will be the slowest to process.
    pub fn with_partitions_simd(count: usize, partition_count: usize) -> Self {
        let mut partitions: Vec<Partition>;
        
        // How many total simds can we accomodate?
        let simd_per_lane = count / ALL_SIMD_LANES;
        if simd_per_lane < 1 {
        // Not enough data for a full SIMD operation, let alone multi-threading.
            partitions = vec![Partition::new(0, count - 1)];
            return Partitioner::new(partitions);
        }

        //How many simds per thread can we accommodate?
        let simd_per_lane_per_partition = simd_per_lane / partition_count;
        if simd_per_lane_per_partition < 1 {
        // Not enough simd_chunks to make multi-threaded worth it.
        // Consideration: choosing actual parallelization based on modulo.
            partitions = vec![Partition::new(0, count - 1)];
            return Partitioner::new(partitions);
        } else {
            partitions = Vec::with_capacity(count / partition_count);
        }

        let partition_size = ALL_SIMD_LANES * simd_per_lane_per_partition;
        let simd_spread = simd_per_lane % partition_count;

        let mut start;
        let mut adjusted_partition_size;
        let mut cursor = 0;
        let mut end;
        for i in 0..partition_count - 1 {
            start = cursor;
            adjusted_partition_size = partition_size + if i < simd_spread { ALL_SIMD_LANES } else { 0 };
            cursor = start + adjusted_partition_size;
            end = cursor - 1;

            partitions.push(Partition::new(start, end));
        }

        partitions.push(Partition::new(cursor, count - 1));

        Partitioner::new(partitions)
    }
}

impl Partition {
    pub fn unary_simd(&self,
        partition_values: &mut Vec<f32>, 
        lhs_slice: &[f32], 
        simd_op: impl Fn(Simd<f32, ALL_SIMD_LANES>) -> Simd<f32, ALL_SIMD_LANES>,
        remainder_op: impl Fn(f32) -> f32
    ) {
        let return_slice: &mut Vec<f32> = &mut vec![0.; ALL_SIMD_LANES];

        let mut cursor_start = self.get_start();
        let mut cursor_end = cursor_start + ALL_SIMD_LANES;
        while cursor_end <= self.get_end() + 1 {
            let x_simd = Simd::<f32, ALL_SIMD_LANES>::from_slice(&lhs_slice[cursor_start..cursor_end]);

            let r_simd = simd_op(x_simd);
            r_simd.copy_to_slice(return_slice);
            partition_values.extend_from_slice(&return_slice);

            cursor_start = cursor_end;
            cursor_end += ALL_SIMD_LANES;
        }

        if cursor_end > self.get_end() { cursor_end -= ALL_SIMD_LANES; }

        for i in cursor_end..=self.get_end() {
            partition_values.push(remainder_op(lhs_slice[i]));
        }
    }

    pub fn binary_simd(&self,
        partition_values: &mut Vec<f32>,
        lhs_slice: &[f32],
        rhs_slice: &[f32],
        simd_op: impl Fn(Simd<f32, ALL_SIMD_LANES>, Simd<f32, ALL_SIMD_LANES>) -> Simd<f32, ALL_SIMD_LANES>,
        remainder_op: impl Fn(f32, f32) -> f32
    ) {
        let return_slice: &mut Vec<f32> = &mut vec![0.; ALL_SIMD_LANES];

        let mut cursor_start = self.get_start();
        let mut cursor_end = cursor_start + ALL_SIMD_LANES;
        while cursor_end <= self.get_end() + 1 {
            let x_simd = Simd::<f32, ALL_SIMD_LANES>::from_slice(&lhs_slice[cursor_start..cursor_end]);
            let y_simd = Simd::<f32, ALL_SIMD_LANES>::from_slice(&rhs_slice[cursor_start..cursor_end]);

            let r_simd = simd_op(x_simd, y_simd);
            r_simd.copy_to_slice(return_slice);
            partition_values.extend_from_slice(&return_slice);

            cursor_start = cursor_end;
            cursor_end += ALL_SIMD_LANES;
        }

        if cursor_end > self.get_end() { cursor_end -= ALL_SIMD_LANES; }

        for i in cursor_end..=self.get_end() {
            partition_values.push(remainder_op(lhs_slice[i], rhs_slice[i]));
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::partitioner::PARALLELISM;
    use crate::{prettify::*, timed};
    use crate::{geoalg::f32_math::{matrix::Matrix, optimized_functions::dot_product_of_vector_slices}, partitioner::Partitioner};

    use super::dot_product_simd3;

    use super::*;

    #[test]
    fn test_element_wise_mul_simd() {
        let lhs = Matrix::new_randomized_z(1000, 10000);
        let rhs = Matrix::new_randomized_z(1000, 10000);
        
        let mut expected = Matrix::new(1, 1, vec![0.; 1]);

        let elapsed = timed::timed(|| {
            expected = lhs
                .mul_element_wise(&rhs)
                .mul_element_wise(&rhs)
                .mul_element_wise(&lhs);
        });
        println!("Elapsed no simd: {elapsed}");

        let time_scale = 1.;
        let elapsed = time_scale * timed::timed(|| {
            let _actual = lhs
                .mul_element_wise_simd(&rhs)
                .mul_element_wise_simd(&rhs)
                .mul_element_wise_simd(&lhs);
        });
        println!("Elapsed old school simd: {elapsed:0.9}");

        let elapsed = time_scale * timed::timed(|| {
            let actual = mul_simd(lhs.read_values(), rhs.read_values());
            let actual = mul_simd(&actual[..], rhs.read_values());
            let _actual = mul_simd(&actual[..], lhs.read_values());
        });
        println!("Elapsed new school simd: {elapsed:0.9} *Seems to be fastest without multi-threading wowza");

        let partitioner = Partitioner::with_partitions_simd(
            lhs.len(),
            PARALLELISM);
        let elapsed = time_scale *timed::timed(|| {
            let _actual = partitioner.parallelized(|&partition| {
                let x = mul_simd(&lhs.read_values()[partition.range()], &rhs.read_values()[partition.range()]);
                let x = mul_simd(&x[..], &rhs.read_values()[partition.range()]);
                mul_simd(&x[..], &lhs.read_values()[partition.range()])
            });
        });
        println!("Elapsed thd school simd: {elapsed:0.9}");

        let elapsed = time_scale * timed::timed(||{
            let _actual = partitioner.parallelized(|&partition| {
                let mut partition_values = Vec::with_capacity(lhs.len());
                partition.binary_simd(
                    &mut partition_values, 
                    &lhs.read_values(), 
                    &rhs.read_values(), 
                    |x_simd, y_simd| { x_simd * y_simd }, 
                    |x, y| x * y);

                partition_values
            });
        });
        println!("Elapsed bin school simd: {elapsed:0.9} *Quickly becoming slowest");
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

        assert_eq!(actual, expected);

        let lhs = Matrix::new_randomized_z(1, 100000).read_values().to_vec();
        let rhs = Matrix::new_randomized_z(1, 100000).read_values().to_vec();
        let expected = dot_product_of_vector_slices(&lhs, &rhs);
        let actual = dot_product_simd3(&lhs, &rhs);
        
        let error = (actual - expected).abs().log10();
        println!("{GREEN}Dot product error: {error}{RESET}");
    }

    #[test]
    fn test_with_partitions_simd() {
        let t1 = Partitioner::with_partitions_simd(4, 1);
        let t3 = Partitioner::with_partitions_simd(32, 2);
        let t2 = Partitioner::with_partitions_simd(16, 1);
        let t4 = Partitioner::with_partitions_simd(48, 2);
        let t5 = Partitioner::with_partitions_simd(49, 2);
        let t6 = Partitioner::with_partitions_simd(69, 2);
        let t7 = Partitioner::with_partitions_simd(72, 3);
        let t8 = Partitioner::with_partitions_simd(256, 16);
        let t9 = Partitioner::with_partitions_simd(272, 16);
        let t10 = Partitioner::with_partitions_simd(312, 16);
        let t11 = Partitioner::with_partitions_simd(10019, 8);

        println!("{:?}", t1);
        println!("{:?}", t2);
        println!("{:?}", t3);
        println!("{:?}", t4);
        println!("{:?}", t5);
        println!("{:?}", t6);
        println!("{:?}", t7);
        println!("{:?}", t8);
        println!("{:?}", t9);
        println!("{:?}", t10);
        println!("{:?}", t11);

        // let msg = format!("Partitions: {:?}", x).bright_red();
        // println!("{msg}");
    }

    #[test]
    pub fn test_partition_unary_simd_with_remainder() {
        let tc = Partition::new(0, 16);

        let lhs_slice = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17.];

        let y = 3.;
        let y_simd = Simd::<f32, ALL_SIMD_LANES>::splat(y);
        let mut partition_values = Vec::with_capacity(17);
        tc.unary_simd(&mut partition_values, &lhs_slice, 
            |x_simd| x_simd * y_simd, 
            |x| x + y);

        let expected = vec![3., 6., 9., 12., 15., 18., 21., 24., 27., 30., 33., 36., 39., 42., 45., 48., 20.];
        assert_eq!(partition_values, expected);
    }

    #[test]
    pub fn test_partition_binary_simd_with_remainder() {
        let tc = Partition::new(0, 16);
        let lhs_slice = vec![3., 6., 9., 12., 15., 18., 21., 24., 27., 30., 33., 36., 39., 42., 45., 48., 20.];
        let rhs_slice = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17.];
        let expected = vec![2., 4., 6., 8., 10., 12., 14., 16., 18., 20., 22., 24., 26., 28., 30., 32., 340.0];

        let mut partition_values = Vec::with_capacity(17);
        tc.binary_simd(
            &mut partition_values, 
            &lhs_slice, 
            &rhs_slice, 
            |x_simd, y_simd| x_simd - y_simd,
            |x, y| x * y);

        assert_eq!(partition_values, expected);
    }

    #[test]
    fn test_scale_simd() {
        let tc = Matrix::new_randomized_z(1000, 1000);

        let expected = tc.scale(1.2);
        let actual = tc.scale_simd(1.2);

        assert_eq!(actual, expected);
    }
}