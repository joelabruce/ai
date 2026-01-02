# ai
AI library written in rust

## At a glance
* **Allows for *simple* creation of complex and fast neural networks utilizing SIMD and multi-threading capabilities built-in.**
* Easy to follow examples for showing how the library can be used.
* Unit tests of most features to show how they can be used in code.
* Minimal use of 3rd party libraries.
* Coded with simplicity in mind so even people new to AI can follow along!
* Uses an adjustable file cycling system while training for easy rollback if training performance begins to decline or to rollback when model shows signs of overfitting. Automatically saves model weights and biases after each epoch.

## Installation

### Quick Start (Recommended)

For the easiest setup experience, use the provided initialization scripts:

**Linux & macOS:**
```bash
./init.sh
```

**Windows (Command Prompt):**
```cmd
init.bat
```

**Windows (PowerShell):**
```powershell
.\init.ps1
```

These scripts will:
- Verify Rust installation
- Install nightly toolchain if needed
- Extract MNIST datasets
- Build the project and fetch dependencies

If you prefer manual setup or encounter any issues, follow the detailed instructions below.

### System Requirements
- **Rust:** Nightly build (required for portable SIMD features)
- **CPU:** Multi-core processor with SIMD support (SSE/AVX on x86, NEON on ARM)
- **OS:** Linux, macOS, or Windows
- **Git:** Required for cloning the repository

### Prerequisites

**All platforms:**
- [Rust toolchain](https://rustup.rs/) - Install rustup if not already installed
- Git - [Download for your platform](https://git-scm.com/downloads)

**Windows users:** You'll need one of the following to extract ZIP files from command line:
- PowerShell (built-in, Windows 5.0+)
- [7-Zip](https://www.7-zip.org/) or [WinRAR](https://www.win-rar.com/) (optional)
- Or use Windows Explorer to extract manually

### Dependencies
This project has minimal external dependencies:
- `rand = "0.8.5"` - Random number generation
- `rand_distr = "0.4.0"` - Statistical distributions

### Setup

#### 1. Install Rust Nightly

**All platforms:**
```bash
rustup install nightly
rustup default nightly
```

#### 2. Extract MNIST Datasets (Required for Examples)

**Linux & macOS:**
```bash
unzip archive/mnist_train.csv.zip -d training/
unzip archive/mnist_test.csv.zip -d training/
```

**Windows (PowerShell):**
```powershell
Expand-Archive -Path archive\mnist_train.csv.zip -DestinationPath training\
Expand-Archive -Path archive\mnist_test.csv.zip -DestinationPath training\
```

**Windows (Command Prompt with 7-Zip or WinRAR installed):**
```cmd
7z x archive\mnist_train.csv.zip -otraining\
7z x archive\mnist_test.csv.zip -otraining\
```

**Alternative for Windows:** You can also extract the ZIP files manually using Windows Explorer.

#### 3. Build the Project

**All platforms:**
```bash
cargo build --release
```

### Run unit test and see output for each test in debug mode
```
cargo test -- --show-output
```
### To see performance benchmarks that are more realistic, highly recommend running tests in release mode
```
cargo test --release -- --show-output
```
### Generate code coverage report
```
cargo llvm-cov --html
```

## API Documentation

Generate and view the full API documentation locally:
```bash
cargo doc --open
```

This will build the documentation for all modules, structs, and methods, then open it in your browser.

### Run examples
**Important:** Always run in release mode to see realistic performance. Debug mode can be 10-100x slower.

Fully connected Neural Network:
```bash
cargo run --release --example mnist_digits
```

Convolutional Neural Network:
```bash
cargo run --release --example mnist_conv_digits
```

## Architecture

### Core Components

#### Matrix Operations (`src/geoalg/f32_math/`)
- **Row-major storage** for cache efficiency and optimal memory access patterns
- **SIMD-accelerated operations** using portable SIMD (16-lane f32 vectors)
- **Multi-threaded operations** via intelligent work partitioning
- **Optimized convolution** via im2col transformation for efficient GEMM operations
- **Adaptive algorithm selection** automatically chooses single/multi-threaded SIMD based on workload size

#### Neural Network Layers (`src/nn/layers/`)
- **Dense**: Fully connected layer with He initialization
- **Convolution2d**: 2D convolution with configurable kernels and filters
- **MaxPooling**: Max pooling with gradient tracking for backpropagation
- **Input**: Shape specification layer

#### Training System (`src/nn/trainer.rs`)
- Batch sampling with automatic shuffling
- Model checkpointing with rolling file cycles
- CSV data loading and binary model persistence
- Configurable hyperparameters

### Performance Features
- **Portable SIMD**: 16 f32 elements per vector operation
- **Dynamic thread scaling** based on `available_parallelism()`
- **Work partitioning** optimized for SIMD alignment
- **im2col convolution** for efficient matrix multiplication
- **Adaptive methods**: `scale()` and `mul_transposed_b()` automatically select optimal implementation

## Useful features
### Partitioner and Partition
**Still Under construction since discovery of chunks_mut** \
Partitioner can create partitions that split-up work to be done when multi-threading. When using SIMD extensions, the partitioner can create partitions that favor SIMD based on the SIMD_LANES you specify. By using the SIMD Partitioner function, it will guarantee whenever possible that all threads except for the last one will be evenly split to do accommodate SIMD. The last thread's partition is guaranteed to be the smallest since it will process anything that cannot be split into SIMD_LANES in the other threads to attempt to achieve an even workload across all threads.

### Matrix
The major player in the library that has optimized implementations to work fast on modern CPUs that have multiple cores and supports SIMD via SIMD extensions.

### Sample
Allows for creating a batch that randomly draws from a sample. The sample can be reset for reuse at any-time, guaranteeing that no duplicates are ever present in the batch.

### Timed function
Allows for wrapping functions to determine runtime. Useful for seeing how performant a training session is.

## Examples
* MNIST Hand-written Digits Neural Networks
  * A simple fully connected neural network that uses Softmax and Cross Entropy Loss.
  * An implementation using convolutional neural network.

## Current Limitations

- **Convolution**: Only "valid" mode (no padding) currently supported
- **Optimizers**: Only basic SGD; no Adam/RMSprop/momentum yet
- **Learning Rate**: Fixed learning rate only (adaptive scheduling planned)
- **GPU**: CPU-only, no CUDA/GPU acceleration
- **Data Formats**: CSV and binary only, no HDF5/NPZ support
- **Batch Normalization**: Not yet implemented

## Coming Soon
* Transformers
  * Positional-encoding modules
  * Embedding layer
* Advanced Optimizers (Adam, RMSprop)
* Learning rate scheduling
* Batch Normalization and Dropout layers

## Goals of project
  * To develop a set of tools to allow simple creation of complex neural networks.
  * Only implement and optimize functions that directly impact this goal.
  * Though many features *could* be implemented for things such as matrix identities or determinants, those operations will only be implemented as deemed useful to creating sophisticated neural networks. Right now they are not implemented because currently no layers need them to function.
  * Work to keep the code base as clutter free as possible. Actively remove functions that are not providing value to the end goal of simply creating neural networks.