use std::thread;

use rand::distributions::Uniform;
use crate::{geoalg::f64_math::matrix::Matrix, partitioner_cache::PartitionerCache};
use super::*;

/// A fully connected layer.
pub struct DenseLayer {
    pub biases: Matrix,
    pub weights: Matrix,

    partitioner_cache: PartitionerCache,
    parallelism: usize
}

impl DenseLayer {
    fn random_weight_biases(neuron_count: usize, prev_layer_input_count: usize) -> (Matrix, Matrix) {
        let uniform = Uniform::new_inclusive(-0.5, 0.5);
        let weights = Matrix::new_randomized_uniform(prev_layer_input_count, neuron_count, uniform);
        let biases = Matrix::from(1, neuron_count, vec![0.0; neuron_count]);

        (weights, biases)
    }

    pub fn new(input_size: usize, neuron_count: usize) -> DenseLayer {
        let (weights, biases) = DenseLayer::random_weight_biases(neuron_count, input_size);

        DenseLayer {
            weights,
            biases,
            partitioner_cache: PartitionerCache::new(),
            parallelism: thread::available_parallelism().unwrap().get()
        }
    }

    /// Instantiates and returns a new Hidden Layer based off self's shape.
    pub fn calulated_dense_layer(&self, neuron_count: usize) -> DenseLayer {
        assert!(self.weights.len() > 0);
        assert!(self.biases.len() > 0);

        DenseLayer::new(self.biases.len(), neuron_count)
    }
}

impl Propagates for DenseLayer {
    /// Forward propagates by performing weights dot inputs + biases.
    fn forward<'a>(&mut self, inputs: &'a Matrix) -> Matrix {
        let r = inputs
            .mul_with_transpose(&self.weights.transpose())
            .add_row_partitioned(&self.biases);
        r
    }

    /// Back propagates.
    /// Will return dvalues to be used in next back propagation layer.
    fn backward<'a>(& mut self, dvalues: &Matrix, inputs: &Matrix) -> Matrix {
        // Mutate the weights based on derivative weights
        let inputs_t = inputs.transpose();
        let dweights = inputs_t.mul_with_transpose(&dvalues.transpose());

        self.weights = self.weights.sub(&dweights.scale(learning_rate()));

        // Mutate the biases based on derivative biases
        //let dbiases = dvalues.shrink_rows_by_add();
        let dbiases = dvalues.reduce_rows_by_add();
        self.biases = self.biases.sub(&dbiases.scale(learning_rate()));

        let result = dvalues.mul_with_transpose(&self.weights);
        result
    }
}