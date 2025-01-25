use rand::distributions::Uniform;
use crate::geoalg::f64_math::matrix::Matrix;
use super::*;

/// A fully connected layer.
pub struct DenseLayer {
    pub biases: Matrix,
    pub weights: Matrix
}

impl DenseLayer {
    fn random_weight_biases(neuron_count: usize, prev_layer_input_count: usize) -> (Matrix, Matrix) {
        let uniform = Uniform::new_inclusive(-0.5, 0.5);
        let weights = Matrix::new_randomized_uniform(prev_layer_input_count, neuron_count, uniform);
        let biases = Matrix::from_vec(vec![0.0; neuron_count], 1, neuron_count);

        (weights, biases)
    }

    pub fn new(input_size: usize, neuron_count: usize) -> DenseLayer {
        let (weights, biases) = DenseLayer::random_weight_biases(neuron_count, input_size);
        DenseLayer {
            weights,
            biases
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
    /// Z
    fn forward<'a>(&self, inputs: &'a Matrix) -> Matrix {
        let r = inputs
            .mul_threaded_rowwise(&self.weights)
            .add_row_vector(&self.biases);
        r
    }

    /// In progress to support ADAM optimization
    fn backward<'a>(& mut self, dvalues: &Matrix, inputs: &Matrix) -> Matrix {
        // Mutate the weights based on derivative weights
        let dweights = inputs.get_transpose().mul_threaded_rowwise(dvalues);
        self.weights = self.weights.sub(&dweights.scale(learning_rate()));


        // Mutate the biases based on derivative biases
        let dbiases = dvalues.shrink_rows_by_add();
        self.biases = self.biases.sub(&dbiases.scale(learning_rate()));

        let x = dvalues.mul_with_transposed_threaded_rowwise(&self.weights);
        x
    }
}