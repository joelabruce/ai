//! # AI Neural Network Library
//!
//! A high-performance neural network library written in Rust with SIMD and multi-threading optimizations.
//!
//! ## Features
//!
//! - **SIMD-Accelerated Operations**: Utilizes portable SIMD (16-lane f32 vectors) for fast matrix operations
//! - **Multi-Threading**: Intelligent work partitioning for parallel computation
//! - **Optimized Convolutions**: im2col transformation for efficient GEMM operations
//! - **Adaptive Algorithms**: Automatic selection of optimal implementation based on workload size
//! - **Minimal Dependencies**: Only 2 external crates required (`rand`, `rand_distr`)
//!
//! ## Quick Start
//!
//! ```rust
//! use ai::nn::neural::{NeuralNetwork, NeuralNetworkNode};
//! use ai::nn::layers::dense::Dense;
//!
//! // Create a simple neural network
//! let mut network = NeuralNetwork::new();
//! network.add_node(NeuralNetworkNode::DenseLayer(Dense::new(784, 128)));
//! // Add more layers as needed...
//! ```
//!
//! ## Core Modules
//!
//! - [`geoalg`]: Geometric algebra and matrix operations with SIMD optimizations
//! - [`nn`]: Neural network layers, activations, and training infrastructure
//! - [`partitioner`]: Work partitioning for multi-threaded operations
//! - [`timed`]: Performance measurement utilities
//!
//! ## Performance
//!
//! This library is designed for CPU-based training and inference with focus on:
//! - Cache-efficient row-major matrix storage
//! - SIMD vectorization for parallel arithmetic operations
//! - Dynamic thread scaling based on available CPU cores
//! - Minimal memory allocations during training
//!
//! **Note:** Always run in release mode (`--release`) for realistic performance.
//! Debug mode can be 10-100x slower due to missing optimizations.

#![feature(portable_simd)]

pub mod geoalg;
pub mod partitioner;
pub mod partition;
pub mod statistics;
pub mod digit_image;
pub mod input_csv_reader;
pub mod output_bin_writer;
pub mod nn;
pub mod timed;
pub mod prettify;
