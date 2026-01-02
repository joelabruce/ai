use rand_distr::Uniform;

use crate::geoalg::f32_math::matrix::Matrix;
use super::*;

/// A fully connected (dense) neural network layer.
///
/// Each neuron in this layer is connected to every input from the previous layer.
/// The layer performs the operation: `output = inputs × weights^T + biases`
///
/// # Initialization
///
/// Weights are initialized using He uniform initialization:
/// - Range: `[-√(6/n), √(6/n)]` where n is the input size
/// - Biases are initialized to zero
///
/// This initialization helps prevent vanishing/exploding gradients during training.
///
/// # Examples
///
/// ```
/// use ai::nn::layers::dense::Dense;
/// use ai::nn::layers::Propagates;
/// use ai::geoalg::f32_math::matrix::Matrix;
///
/// // Create a dense layer: 784 inputs → 128 outputs
/// let mut layer = Dense::new(784, 128);
///
/// // Forward propagate a batch of 32 samples
/// let inputs = Matrix::new_randomized_z(32, 784);
/// let outputs = layer.forward(&inputs);
/// assert_eq!(outputs.shape(), (32, 128));
/// ```
///
/// # Performance
///
/// - **Forward pass**: Matrix multiplication with SIMD and multi-threading
/// - **Backward pass**: Gradient computation parallelized for large batches
/// - **Memory**: Stores weights (input_size × neuron_count) and biases (1 × neuron_count)
pub struct Dense {
    pub biases: Matrix,
    pub weights: Matrix,
}

impl Dense {
    /// Initializes using He initialization.
    /// Might consider allowing for more flexibility here, but have to think carefuly about the cleanest way to do this.
    fn random_weight_biases(prev_layer_input_count: usize, neuron_count: usize) -> (Matrix, Matrix) {
        //let std_dev = 2. / (prev_layer_input_count as f32);
        //let normal = Normal::new(0., std_dev).unwrap();
        //let weights = Matrix::new_randomized_normal(prev_layer_input_count, neuron_count, normal);
        
        let term = (6. / (prev_layer_input_count as f32)).sqrt();
        let uniform = Uniform::new_inclusive(-term, term);
        let weights = Matrix::new_randomized_uniform(prev_layer_input_count, neuron_count, uniform);
        let biases = Matrix::new(1, neuron_count, vec![0.0; neuron_count]);

        (weights, biases)
    }

    /// Creates a new dense layer with He uniform initialization.
    ///
    /// # Arguments
    ///
    /// * `input_size` - Number of input features from the previous layer
    /// * `neuron_count` - Number of neurons (output features) in this layer
    ///
    /// # Examples
    ///
    /// ```
    /// use ai::nn::layers::dense::Dense;
    ///
    /// // Create layer that takes 784 inputs and produces 128 outputs
    /// let layer = Dense::new(784, 128);
    /// ```
    pub fn new(input_size: usize, neuron_count: usize) -> Dense {
        let (weights, biases) = Dense::random_weight_biases(input_size, neuron_count);

        Dense { weights, biases }
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
            .mul_transposed_b(&self.weights.transpose())
            .add_row_partitioned(&self.biases);
        r
    }

    /// Back propagates.
    /// Will return dvalues to be used in next back propagation layer.
    fn backward<'a>(& mut self, learning_rate: &mut LearningRate, dvalues: &Matrix, inputs: &Matrix) -> Matrix {
        // Mutate the weights based on derivative weights
        let inputs_t = inputs.transpose();
        let dweights = inputs_t.mul_transposed_b(&dvalues.transpose());

        self.weights = self.weights.sub(&dweights.scale(learning_rate.rate()));

        // Mutate the biases based on derivative biases
        let dbiases = dvalues.reduce_rows_by_add();
        self.biases = self.biases.sub(&dbiases.scale(learning_rate.rate()));

        let result = dvalues.mul_transposed_b(&self.weights);
        result
    }
}

#[cfg(test)]
mod test {
    use crate::{geoalg::f32_math::matrix::Matrix, nn::{layers::Propagates, learning_rate::LearningRate}};

    use super::Dense;
    use crate::prettify::*;

    #[test]
    fn test_propagations() {
        let features = 784;
        let neuron_count = 128;

        let mut dense = Dense::new(features, neuron_count);

        let batches = 500;
        let inputs = &Matrix::new_randomized_z(batches, features);

        let forward = &dense.forward(inputs);
        println!("{BRIGHT_YELLOW}{:?}{RESET}", forward.shape());

        let learning_rate = &mut LearningRate::new(0.01);
        let dvalues = &Matrix::new_randomized_z(batches, neuron_count);
        let _dz = dense.backward(learning_rate, dvalues, inputs);
    }

    #[test]
    fn test_influences() {
        let features = 784;
        let neuron_count_1 = 128;
        let mut dense1 = Dense::new(features, neuron_count_1);

        let neuron_count_2 = 64;
        let mut dense2 = dense1.influences_dense(neuron_count_2);

        let batches = 500;
        let inputs = &Matrix::new_randomized_z(batches, features);

        let forward1 = &dense1.forward(inputs);
        println!("{BRIGHT_YELLOW}{:?}{RESET}", forward1.shape());

        let _forward2 = &dense2.forward(forward1);

        let learning_rate = &mut LearningRate::new(0.01);
        let dvalues = &Matrix::new_randomized_z(batches, neuron_count_2);
        let dz2 = &dense2.backward(learning_rate, dvalues, forward1);

        let _dz1 = dense1.backward(learning_rate, dz2, inputs);
    }
}