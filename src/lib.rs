#![feature(array_chunks)]
#![feature(slice_as_chunks)]
#![feature(portable_simd)]

pub mod geoalg;
pub mod partitions;
pub mod statistics;
pub mod digit_image;
pub mod input_csv_reader;
pub mod output_bin_writer;
pub mod nn;
pub mod partitioner_cache;
pub mod weight_initializers;
pub mod timed;
pub mod prettify;

// Consier taking this out altogether?
use crate::partitions::*;