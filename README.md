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
Requires using nightly rust build for SIMD to work.

```
rustup upgrade -- nightly
rustup default nightly
```

Ensure you unzip the archive zip files into the training folder to be able to run the examples.

### Run unit test and see output for test not fully implemented
```
cargo test -- --nocapture
```
### Generate code coverage report
```
cargo llvm-cov --html
```

### Run examples
Make sure to run in release mode to see how fast the library can perform on your machine.
```
cargo run --release --example mnist_digits
```

## Useful features
### Partitioner and Partition
Partitioner can create partitions that split-up work to be done when multi-threading. When using SIMD extensions, the partitioner can create partitions that favor SIMD based on the SIMD_LANES you specify. By using the SIMD Partitioner function, it will guarantee whenever possible that all threads except for the last one will be evenly split to do accommodate SIMD. The last thread's partition is guaranteed to be the smallest since it will process anything that cannot be split into SIMD_LANES in the other threads to attempt to achieve an even workload across all threads.

### Tensor
The major player in the library that has optimized implementations to work fast on modern CPUs that have multiple cores and supports SIMD via SIMD extensions.

### Sample
Allows for creating a batch that randomly draws from a sample. The sample can be reset for reuse at any-time, guaranteeing that no duplicates are ever present in the batch.

### Timed function
Allows for wrapping functions to determine runtime. Useful for seeing how performant a trining session is.

## Examples
* MNIST Hand-written Digits Neural Networks
  * A simple fully connected neural network that uses Softmax and Cross Entropy Loss.
  * An implementation using convolutional neural network.

## Goals of project
  * To develop a set of tools to allow simple creation of complex neural networks.
  * Only implement and optimize functions that directly impact this goal.
  * Though many features *could* be implemented for things such as matrix identities or determinants, those operations will only be implemented as deemed useful to creating sophisticated neural networks. Right now they are not implemented because currently no layers need them to function.
  * Work to keep the code base as clutter free as possible. Actively remove functions that are not providing value to the end goal of simply creating neural networks.