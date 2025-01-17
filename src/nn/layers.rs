extern crate rand;
use rand::distributions::Uniform;
use crate::geoalg::f64_math::matrix::*;

pub fn learning_rate() -> f64 {
    0.01
}

pub trait Propagates {
    fn forward(&self, inputs: &Matrix) -> Matrix;
    fn backward<'a>(&'a mut self, dvalues: &Matrix, inputs: &Matrix) -> Matrix;
}

/// Allows for creation of succeeding layers based on initial data passed in.
pub struct InputLayer {
    pub input_matrix: Matrix
}

/// Should be created beginning with InputLayer to ensure shapes are correct.
pub struct HiddenLayer {
    pub biases: Matrix,
    pub weights: Matrix
}

pub struct ConvolutionalLayer {
    pub kernel: Matrix,
    pub biases: Matrix,
    pub stride: usize
}

impl InputLayer {
    /// Creates input layer based on inputs supplied.
    /// Allows for automatic shaping of succeeding layer generation.
    pub fn from_vec(values: Vec<Vec<f64>>) -> InputLayer {
        let rows = values.len();                                                // Number of inputs
        let values: Vec<f64> = values.into_iter().flat_map(|v| v).collect();
        let columns = values.len() / rows;                                      // Number of features.

        InputLayer {
            input_matrix: Matrix::from_vec(values, rows, columns)
        }
    }

    /// Input layer always forwards to first hidden layer and not to an activation function.
    /// # Arguments
    /// # Returns
    pub fn forward(&self) -> Matrix {
        self.input_matrix.clone()
    }

}

impl HiddenLayer {
    fn random_weight_biases(neuron_count: usize, prev_layer_input_count: usize) -> (Matrix, Matrix) {
        let uniform = Uniform::new_inclusive(-0.5, 0.5);
        let weights = Matrix::new_randomized_uniform(prev_layer_input_count, neuron_count, uniform);
        let biases = Matrix::from_vec(vec![0.0; neuron_count], 1, neuron_count);

        (weights, biases)
    }

    pub fn new(input_size: usize, neuron_count: usize) -> HiddenLayer {
        let (weights, biases) = HiddenLayer::random_weight_biases(neuron_count, input_size);
        HiddenLayer {
            weights,
            biases
        }
    }

    /// Instantiates and returns a new Hidden Layer based off self's shape.
    pub fn calulated_hidden_layer(&self, neuron_count: usize) -> HiddenLayer {
        assert!(self.weights.len() > 0);
        assert!(self.biases.len() > 0);

        HiddenLayer::new(self.biases.len(), neuron_count)
    }
}

impl ConvolutionalLayer {
    //fn 
}

impl Propagates for HiddenLayer {
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

#[cfg(test)]
mod tests {
    #[test]
    fn testflow() {

    }
}