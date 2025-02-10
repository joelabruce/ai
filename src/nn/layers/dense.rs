use rand_distr::Uniform;

//use rand::distributions::Uniform;
use crate::geoalg::f32_math::matrix::Matrix;
use super::*;

/// A fully connected layer.
pub struct Dense {
    pub biases: Matrix,
    pub weights: Matrix,

    //partitioner_cache: PartitionerCache,
    //parallelism: usize
}

impl Dense {
    fn random_weight_biases(prev_layer_input_count: usize, neuron_count: usize) -> (Matrix, Matrix) {
        let uniform = Uniform::new_inclusive(-0.5, 0.5);
        let weights = Matrix::new_randomized_uniform(prev_layer_input_count, neuron_count, uniform);
        let biases = Matrix::from(1, neuron_count, vec![0.0; neuron_count]);

        (weights, biases)
    }

    pub fn new(input_size: usize, neuron_count: usize) -> Dense {
        let (weights, biases) = Dense::random_weight_biases(input_size, neuron_count);

        Dense {
            weights,
            biases,
        }
    }

    /// Instantiates and returns a new Hidden Layer based off self's shape.
    pub fn influences_dense(&self, neuron_count: usize) -> Dense {
        assert!(self.weights.len() > 0);
        assert!(self.biases.len() > 0);

        Dense::new(self.biases.len(), neuron_count)
    }
}

impl Propagates for Dense {
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

        self.weights = self.weights.sub(&dweights.scale_simd(learning_rate()));

        // Mutate the biases based on derivative biases
        let dbiases = dvalues.reduce_rows_by_add();
        self.biases = self.biases.sub(&dbiases.scale_simd(learning_rate()));

        let result = dvalues.mul_with_transpose(&self.weights);
        result
    }
}