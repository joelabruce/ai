use std::{collections::btree_map::Values, thread};

pub struct Kernel{
    pub c: usize,
    pub h: usize,
    pub w: usize
}

impl Kernel {
    fn new() -> Self {
        Kernel { c: 3, h: 3, w: 3 }
    }
}

pub fn im2col_std_parallel(
    input: &[f32],
    shape: &Kernel,
    k_size: usize,
    stride: usize,
    padding: usize,
    num_threads: usize,
) -> Vec<f32> {
    let out_h = (shape.h + 2 * padding - k_size) / stride + 1;
    let out_w = (shape.w + 2 * padding - k_size) / stride + 1;
    let patch_count = out_h * out_w;
    let patch_size = shape.c * k_size * k_size;

    let mut cols = vec![0.0; patch_count * patch_size];
    
    // Calculate how many patches each thread should handle
    let patches_per_thread = (patch_count + num_threads - 1) / num_threads;
    // The amount of f32 elements per thread chunk
    let chunk_size = patches_per_thread * patch_size;

    thread::scope(|s| {
        // .chunks_mut returns an iterator of non-overlapping mutable slices
        for (chunk_idx, chunk_slice) in cols.chunks_mut(chunk_size).enumerate() {
            s.spawn(move || {
                for (in_chunk_idx, patch_slice) in chunk_slice.chunks_mut(patch_size).enumerate() {
                    // Global index of the patch we are currently filling
                    let patch_idx = chunk_idx * patches_per_thread + in_chunk_idx;
                    
                    let y = patch_idx / out_w;
                    let x = patch_idx % out_w;

                    for c in 0..shape.c {
                        for ky in 0..k_size {
                            for kx in 0..k_size {
                                let in_y = (y * stride) as i32 + ky as i32 - padding as i32;
                                let in_x = (x * stride) as i32 + kx as i32 - padding as i32;

                                if in_y >= 0 && in_y < shape.h as i32 && 
                                   in_x >= 0 && in_x < shape.w as i32 {
                                    let in_idx = c * (shape.h * shape.w) + 
                                                 (in_y as usize * shape.w) + 
                                                 in_x as usize;
                                    
                                    // Local indexing within the thread's slice
                                    let col_offset = (c * k_size * k_size) + (ky * k_size) + kx;
                                    patch_slice[col_offset] = input[in_idx];
                                }
                            }
                        }
                    }
                }
            });
        }
    });

    cols
}

#[cfg(test)]
mod tests {
    use crate::nn::layers::input;

    use super::*;

    #[test]
    fn test_multipart_mutate() {
        let k_size = 3;
        let stride = 1;
        let padding = 0;
        let num_threads = 4;
        let shape = Kernel::new();

        // Input should match shape: c * h * w = 3 * 3 * 3 = 27 elements
        let input = vec![0.0f32; shape.c * shape.h * shape.w];

        let _result = im2col_std_parallel(&input, &shape, k_size, stride, padding, num_threads);
    }
}