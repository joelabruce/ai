use std::{result, thread};

use rand::distributions::Uniform;
use crate::{partitioner_cache::PartitionerCache};
use super::*;

/// A fully connected layer.
pub struct DenseLayer {
    pub biases: Matrix,
    pub weights: Matrix,

    partitioner_cache: PartitionerCache
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
            partitioner_cache: PartitionerCache::new(thread::available_parallelism().unwrap().get())
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
        let weights_t = self.weights.transpose();
        let r= inputs.mul_by_transpose(&weights_t);

        let partitioner = self.partitioner_cache.get_or_add(r.row_count());
        let x = r.add_broadcasted(&self.biases, partitioner);
        x
    }

    /// Back propagates.
    /// Will return dvalues to be used in next back propagation layer.
    fn backward<'a>(& mut self, dvalues: &Matrix, inputs: &Matrix) -> Matrix {
        // Mutate the weights based on derivative weights
        let inputs_t = inputs.transpose();

        let dvalues_t = dvalues.transpose();

        let dweights = inputs_t.mul_by_transpose(&dvalues_t);

        let scaled_dweights= dweights.scale(learning_rate());

        let mut partitioner = self.partitioner_cache.get_or_add(self.weights.row_count());
        self.weights = self.weights.sub(&scaled_dweights, partitioner);

        // Mutate the biases based on derivative biases
        partitioner = self.partitioner_cache.get_or_add(dvalues.len());
        let dbiases = dvalues.shrink_rows_by_add(partitioner);
        
        let scaled_dbiases = dbiases.scale(learning_rate());

        partitioner = self.partitioner_cache.get_or_add(self.biases.row_count());
        self.biases = self.biases.sub(&scaled_dbiases, partitioner);

        let result = dvalues.mul_by_transpose(&self.weights);
        result
    }
}