use crate::{partition::Partition, partitioner::Partitioner};
use std::{simd::{cmp::SimdPartialOrd, num::SimdFloat, *}, thread};
use super::matrix::*;

pub const ALL_SIMD_LANES: usize = 16;
pub const HALF_SIMD_LANES: usize = 8;
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

///
pub fn mul_simd(lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
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

///
pub fn scale_simd(lhs: &[f32], scalar: f32) -> Vec<f32> {
    let (l_extra, l_chunks) = lhs.as_rchunks();
    
    let mut out = Vec::with_capacity(lhs.len());
    for i in 0..l_extra.len() {
        out.push(l_extra[i] * scalar);
    }

    let y = Simd::<f32, ALL_SIMD_LANES>::splat(scalar);
    l_chunks.iter().for_each(|x| {
        out.extend(
            (Simd::<f32, ALL_SIMD_LANES>::from_array(*x) * y).as_array()
        );
    });

    out
}

pub fn relu_simd(lhs: &[f32]) -> Vec<f32> {
    let (l_extra, l_chunks) = lhs.as_rchunks();
    
    let mut out = Vec::with_capacity(lhs.len());
    for i in 0..l_extra.len() {
        out.push(l_extra[i].max(0.));
    }

    let y = Simd::<f32, ALL_SIMD_LANES>::splat(0.);
    l_chunks.iter().for_each(|x| {
        out.extend(
            (Simd::<f32, ALL_SIMD_LANES>::from_array(*x).simd_max(y)).as_array()
        );
    });

    out
}

pub fn d_relu_simd(lhs: &[f32]) -> Vec<f32> {
    let (l_extra, l_chunks) = lhs.as_rchunks();
    
    let mut out = Vec::with_capacity(lhs.len());
    for i in 0..l_extra.len() {
        out.push(l_extra[i].max(0.));
    }

    let y = Simd::<f32, ALL_SIMD_LANES>::splat(0.);
    let true_values = Simd::<f32, ALL_SIMD_LANES>::splat(1.);
    l_chunks.iter().for_each(|x| {
        out.extend(
            (Simd::<f32, ALL_SIMD_LANES>::from_array(*x)
                .simd_gt(y)
                .select(true_values, y)).as_array()
        );
    });

    out
}

pub fn im2col_transposed(
    images: &[f32],
    image_count: usize,
    image_height: usize,
    image_width: usize,
    kernel_height: usize,
    kernel_width: usize
) -> Vec<f32> {
    let image_size = image_height * image_width;
    let kernel_size = kernel_width * kernel_height;
    let feature_height = image_height - kernel_height + 1;
    let feature_width = image_width - kernel_width + 1;
    let feature_size = feature_height * feature_width;

    //let offsets = Vec::with_capacity(kernel_width);

    let mut im2col_transposed = Vec::with_capacity(feature_size * kernel_size);
    let mut image_offset = 0;
    for _n in 0..image_count {
        for feature_row in 0..feature_height {
            for feature_column in 0..feature_width {
                for kernel_row in 0..kernel_height {
                    let offset = image_offset + (feature_row + kernel_row) * image_width + feature_column;
                    im2col_transposed.extend_from_slice(&images[offset..offset + kernel_width]);
                }
            }
        }

        image_offset += image_size;
    }

    im2col_transposed
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

    pub fn scale_simd_new(&self, scalar: f32) -> Matrix {
        let values = self.read_values().scale_simd(scalar);
        Matrix::new(self.row_count(), self.column_count(), values)
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

/// Creates a specialized row that makes calculating convolution super duper fast.
pub fn toeplitz_bruce_matrix(kernel: &[f32],
    kernel_height: usize, kernel_width: usize, 
    image_width: usize) -> Vec<f32>
{
    //let o_height = image_height - kernel_height;
    let o_width = image_width - kernel_width;
    let size = image_width * (kernel_height + 1);

    let mut row = Vec::with_capacity(size);

    let offset = image_width - 1;
    row.resize(offset, 0.);
    for i in 0..kernel_height {
        let start = i * kernel_width;
        let end = start + kernel_width;
        row.extend_from_slice(&kernel[start..end]);
        row.resize(row.len() + o_width, 0.);
    }

    //let remaining_zeros = size - offset;
    row.resize(size, 0.);

    row
}

pub trait SliceExt {
    fn mul_simd(&self, rhs: &Self) -> Vec<f32>;
    fn scale_simd(&self, scalar: f32) -> Vec<f32>;
    fn mm_transpose(&self, rhs: &Self, l_rows: usize, columns: usize, r_rows: usize) -> Vec<f32>;
    fn par_mm_transpose(&self, rhs: &Self, l_rows: usize, columns: usize, r_rows: usize) -> Vec<f32>;
    fn cross_correlate(&self, 
        kernel: &Self, 
        kernel_count: usize,
        kernel_height: usize, kernel_width: usize,
        batches: usize,
        image_height: usize, image_width: usize) -> Vec<f32>;
}

impl SliceExt for [f32] {
    fn mul_simd(&self, rhs: &Self) -> Vec<f32> { mul_simd(self, rhs) }
    
    fn scale_simd(&self, scalar: f32) -> Vec<f32> { scale_simd(self, scalar) }

    fn mm_transpose(&self, rhs: &Self, l_rows: usize, columns: usize, r_rows: usize) -> Vec<f32> {
        let size = l_rows * r_rows;
        let mut values = Vec::with_capacity(size);

        let mut start = 0;
        let mut end = columns;
        for _ in 0..l_rows {
            let mut r_start = 0;
            let mut r_end = columns;
            let lhs = &self[start..end];
            for _ in 0..r_rows {
                values.push(dot_product_simd3(lhs, &rhs[r_start..r_end]));

                r_start = r_end;
                r_end += columns;
            } 

            start = end;
            end += columns;
        }

        values
    }

    fn par_mm_transpose(&self, rhs: &Self, l_rows: usize, columns: usize, r_rows: usize) -> Vec<f32> {
        let partitioner = Partitioner::with_partitions(
            l_rows,
             thread::available_parallelism().unwrap().get());

        let inner = |partition: &Partition| {
            let chunk = partition.size() * columns;
            let l_start = partition.get_start() * columns;
            let l_end = l_start + chunk;
            let lhs_slice = &self[l_start..l_end];
            let l_rows = partition.size();

            lhs_slice.mm_transpose(rhs, l_rows, columns, r_rows)
        };
        partitioner.parallelized(inner)
    }

    fn cross_correlate(&self, 
        kernels: &Self, 
        kernel_count: usize,
        kernel_height: usize, kernel_width: usize, 
        batches: usize,
        image_height: usize, image_width: usize) -> Vec<f32>
    {
        let o_rows = image_height - kernel_height + 1;
        let o_columns = image_width - kernel_width + 1;
        let o_size = o_rows * o_columns;
        let toeplitz_width = kernel_height * image_width;
        let kernel_size = kernel_height * kernel_width;

        let image_size = image_height * image_width;

        let partitioner = Partitioner::with_partitions(batches, 16);

        let mut toeplitzes = Vec::with_capacity(kernel_count);
        for kernel in 0..kernel_count {
            let kernel_start = kernel * kernel_size;
            let kernel_end = kernel_start + kernel_size;

            let toeplitz = toeplitz_bruce_matrix(
                &kernels[kernel_start..kernel_end], 
                kernel_height, kernel_width, 
                image_width);

            toeplitzes.push(toeplitz);
        }

        let inner_process = |partition: &Partition| {
            let mut values = Vec::with_capacity(partition.size() * kernel_count * o_size);

            for batch in partition.range() {
                let image_start = batch * image_size;
                let image_end = image_start + image_size;
                let image = &self[image_start..image_end];

                for kernel in 0..kernel_count {
                    let mut start = 0;
                    let mut end = image_width * kernel_width;

                    for _image_row in 0..o_rows {                        
                        let mut toeplitz_start = image_width - 1;
                        let mut toeplitz_end = toeplitz_start + toeplitz_width;
                        for _image_column in 0..o_columns {
                            let dp = dot_product_simd3(
                                &image[start..end],
                                &toeplitzes[kernel][toeplitz_start..toeplitz_end]);

                            values.push(dp);

                            toeplitz_start -= 1;
                            toeplitz_end -= 1;
                        }

                        start += image_width;
                        end = start + toeplitz_width;
                    }
                }
            }

            values
        };

        partitioner.parallelized(inner_process)
        //values
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
    use std::f32;

    use rand_distr::Uniform;

    use crate::nn::layers::convolution2d::Dimensions;
    use crate::partitioner::PARALLELISM;
    use crate::timed::timed_with_context;
    use crate::prettify::*;
    use crate::{geoalg::f32_math::{matrix::Matrix, optimized_functions::dot_product_of_vector_slices}, partitioner::Partitioner};

    use super::dot_product_simd3;

    use super::*;

    #[test]
    fn test_element_wise_mul_simd() {
        timed_with_context(|timed_context| {
            let lhs = Matrix::new_randomized_z(1000, 10000);
            let rhs = Matrix::new_randomized_z(1000, 10000);
            
            let _expected = lhs
                .mul_element_wise(&rhs)
                .mul_element_wise(&rhs)
                .mul_element_wise(&lhs);
            let elapsed = timed_context.checkpoint();
            println!("Mul Elapsed: {elapsed} multi-threading non-simd");

            let actual1 = lhs
                .mul_element_wise_simd(&rhs)
                .mul_element_wise_simd(&rhs)
                .mul_element_wise_simd(&lhs);
            let elapsed = timed_context.checkpoint();
            println!("Mul Elapsed: {elapsed:0.9} mul_element_wise_simd each call multi-threaded");

            let actual2 = mul_simd(lhs.read_values(), rhs.read_values());
            let actual2 = mul_simd(&actual2[..], rhs.read_values());
            let actual2 = mul_simd(&actual2[..], lhs.read_values());
            let elapsed = timed_context.checkpoint();
            println!("Mul Elapsed: {elapsed:0.9} ***** mul_simd single-threaded");

            let partitioner = Partitioner::with_partitions_simd(
                lhs.len(),
                PARALLELISM);
            let actual3 = partitioner.parallelized(|&partition| {
                let x = mul_simd(&lhs.read_values()[partition.range()], &rhs.read_values()[partition.range()]);
                let x = mul_simd(&x[..], &rhs.read_values()[partition.range()]);
                mul_simd(&x[..], &lhs.read_values()[partition.range()])
            });
            let elapsed = timed_context.checkpoint();
            println!("Mul Elapsed: {elapsed:0.9} mul_simd inside multi-threaded partitioner");

            assert_eq!(actual1.read_values(), actual2);
            assert_eq!(actual2, actual3);

            // let _actual = partitioner.parallelized(|&partition| {
            //     let mut partition_values = Vec::with_capacity(lhs.len());
            //     partition.binary_simd(
            //         &mut partition_values, 
            //         &lhs.read_values(), 
            //         &rhs.read_values(), 
            //         |x_simd, y_simd| { x_simd * y_simd }, 
            //         |x, y| x * y);

            //     partition_values
            // });
            // let elapsed = timed_context.checkpoint();
            // println!("Mul Elapsed: {elapsed:0.9} binary_simd *Quickly becoming slowest");
        });
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

        let lhs = Matrix::new_randomized_z(1, 10000).read_values().to_vec();
        let rhs = Matrix::new_randomized_z(1, 10000).read_values().to_vec();
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
        timed_with_context(|context| {
            let scalar1 = 12.4563;
            let scalar2 = 1.234;
            let tc = Matrix::new_randomized_z(5000, 10000);
            context.checkpoint();

            let expected = tc.scale(scalar1).scale(scalar2);
            let elapsed = context.checkpoint();
            println!("Scale Elapsed: {elapsed:0.9} multi-threaded non-simd");

            let actual2 = tc.scale_simd(scalar1).scale_simd(scalar2);
            let elapsed = context.checkpoint();
            println!("Scale Elapsed: {elapsed:0.9} multi-threaded simd old");

            let actual3 = scale_simd(tc.read_values(), scalar1);
            let actual3 = scale_simd(&actual3, scalar2);
            let elapsed = context.checkpoint();
            println!("Scale Elapsed: {elapsed:0.9} ***** single-threaded simd");

            assert_eq!(actual2, expected);
            assert_eq!(actual3, expected.read_values());
        });
    }

    #[test]
    fn test_matrix_multiply() {
        timed_with_context(|context| {
            let columns = 27;
            let l_rows = 32;
            let r_rows = 21632;
            let lhs = Matrix::new_randomized_z(l_rows, columns);
            let rhs = Matrix::new_randomized_z(r_rows, columns);
            let elapsed = context.checkpoint();
            println!("MMT RNG Elapsed: {elapsed:0.9}");

            let expected = lhs.mul_with_transpose(&rhs);
            let elapsed = context.checkpoint();
            println!("MMT Elapsed: {elapsed:0.9} multi-threaded simd");

            let actual1 = lhs.read_values().mm_transpose(rhs.read_values(), l_rows, columns, r_rows);
            let elapsed = context.checkpoint();
            println!("MMT Elapsed: {elapsed:0.9} single-threaded simd");
            assert_eq!(actual1, expected.read_values());

            let actual = lhs.read_values().par_mm_transpose(rhs.read_values(), l_rows, columns, r_rows);
            let elapsed = context.checkpoint();
            println!("MMT Elapsed: {elapsed:0.9} ***** NEW multi-threaded simd");
            
            println!("MMT NEW {:?} v {:?}", actual.len(), expected.len());
            assert_eq!(actual, expected.read_values());
        });
    }

    #[test]
    fn test_toeplitz() {
        let kernel = [
            1., 2., 3., 
            4., 5., 6.,
            7., 8., 9.
        ];

        let tr = toeplitz_bruce_matrix(&kernel, 
            3, 3,
            28);
        let len = tr.len();
        println!("Toeplitz: {len} ,, {:?}", &tr[..28 * 4]);
    }

    #[test]
    fn test_hybrid_convo() {
        timed_with_context(|context| {
            let m1 = Matrix::new(1, 5 * 5, vec![
                1., 2., 3., 4., 5.,
                6., 7., 8., 9., 10.,
                11., 12., 13., 14., 15.,
                16., 17., 18., 19., 20.,
                21., 22., 23., 25., 25.
            ]);

            let kernel_count = 2;
            let k_d = &Dimensions { height: 3, width: 3};
            let kernels = Matrix::new(kernel_count, 3 * 3, vec![
                -1., -2., -3.,
                -4., -5., -6.,
                -7., -8., -9.,

                -1., -2., -3.,
                -4., -5., -6.,
                -7., -8., -9.
            ]);
            let expected = m1.valid_cross_correlation(
                &kernels,
                k_d,
                &Dimensions { height: 5, width: 5}
            );
            let actual = m1.read_values().cross_correlate(
                kernels.read_values(), 
                kernel_count,
                3, 3,
                1,
                5, 5,
            );
            assert_eq!(actual, expected.read_values());

            // Test two different images
            let batch_size = 2;
            let s4x4 = 16;
            let m2 = Matrix::new(batch_size, s4x4, vec![
                1., 2., 3., 4., 
                5., 6., 7., 8.,
                9., 10., 11., 12.,
                13., 14., 15., 16.,

                10., 20., 30., 40.,
                50., 60., 70., 80.,
                90., 100., 110., 120.,
                130., 140., 150., 160.
            ]);

            let filters = 3;
            let s3x3 = 9;
            let kernels = Matrix::new(filters, s3x3, vec![
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

            let expected = m2.valid_cross_correlation(&kernels, 
                k_d, &Dimensions { height: 4, width: 4});
            let actual = m2.read_values().cross_correlate(
                kernels.read_values(), filters, 
                3, 3, 
                2, 
                4, 4);

            println!("{:?} \n\r{:?}", actual,expected.read_values());

            // Always put images in portrait mode???
            // Algorithm works best when images are in portrait dimensions
            let image_count = 500;
            let image_height = 28;
            let image_width = 28;
            let size = image_height * image_width;
            let kernel_count = 32;
            let kernel_height = 3;
            let kernel_width = 3;
            let kernel_size = kernel_height * kernel_width;
            let m1 = Matrix::new_randomized_z(image_count, size);
            let kernel = Matrix::new_randomized_z(kernel_count, kernel_size);

            context.checkpoint();

            let expected = m1.valid_cross_correlation(&kernel,
                &Dimensions { height: kernel_height, width: kernel_width },
                &Dimensions { height: image_height, width: image_width });
            let elapsed = context.checkpoint();
            println!("{BRIGHT_MAGENTA}Convo Elapsed: {elapsed} Old{RESET}");

            let actual = m1.read_values().cross_correlate(
                &kernel.read_values(),
                kernel_count,
                kernel_height, kernel_width,
                image_count,
                image_height, image_width);
            let elapsed = context.checkpoint();
            println!("{BRIGHT_MAGENTA}Convo Elapsed: {elapsed} New{RESET}");

            let epsilon = f32::EPSILON;
            let errors = actual
                .iter()
                .zip(expected.read_values())
                .filter(|(&x, &y)| (x - y).abs() > 50. * epsilon).count();
            //filter(|&x| x.cmp != 0.).count(); //.collect::<Vec<f32>>();//;.sum::<f32>();
            println!("{BRIGHT_MAGENTA}Convo Errors: {:?} / {:?} v {:?} {RESET}", errors, actual.len(), expected.len());
        });
    }

    #[test]
    fn test_im2col() {
        let _elapsed = timed_with_context(|context| {
            let image_count = 10;
            let image_height = 28;
            let image_width = 28;
            let image_depth = 1;
            let image_size = image_height * image_width * image_depth;
            let images = Matrix::new_randomized_z(image_count, image_size);
            // let images = Matrix::new(image_count, image_size, vec![
            //     1., 2., 3., 4.,
            //     5., 6., 7., 8., 
            //     9., 10., 11., 12.,
            //     13., 14., 15., 16.,

            //     10., 20., 30., 40.,
            //     50., 60., 70., 80., 
            //     90., 100., 110., 120.,
            //     130., 140., 150., 160.
            // ]);

            let kernel_count = 4;
            let kernel_height = 3;
            let kernel_width = 3;
            let kernel_depth = image_depth;
            let kernel_size = kernel_height * kernel_width * kernel_depth;
            let kernels = Matrix::new_randomized_z(kernel_count, kernel_size);
            context.checkpoint();

            let im2col_transposed = im2col_transposed(
                images.read_values(), 
                image_count, 
                image_height, image_width, 
                kernel_height, kernel_width);

            let feature_height = image_height - kernel_height + 1;
            let feature_width = image_width - kernel_width + 1;
            let feature_size = feature_height * feature_width;

            let expected = images.valid_cross_correlation(&kernels, &Dimensions { height: kernel_height, width: kernel_width} , &Dimensions { height: image_height, width: image_width } );
            let actual = kernels.read_values().mm_transpose(&im2col_transposed, kernel_count, kernel_size, image_count * feature_size);

            println!("{:?} v {:?}", actual.len(), expected.len());

            // println!("{:?}", &expected.read_values()[0..50]);
            // println!("{:?}", &actual[0..50]);

            // for i in 0..feature_size {
            //     let start = i * kernel_size;
            //     let end = start + kernel_size;
            //     println!("{:?}", &im2col_transposed[start..end]);
            // }

            // let mut start = 0;
            // let mut end = kernel_size;
            // for i in 0..feature_height {
            //     println!("{:?}", &im2col_transposed[start..end]);
            //     start = end;
            //     end += kernel_size;
                
            // }
        });
    }

    #[test]
    fn test_relu() {
        let _elapsed = timed_with_context(|context| {
            let uniform = Uniform::new_inclusive(-3., 2.);
            let m = Matrix::new_randomized_uniform(10000, 10000, uniform);

            context.checkpoint();
            let _x = relu_simd(m.read_values());
        });
    }
}