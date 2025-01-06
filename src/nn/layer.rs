use crate::geoalg::f64_math::matrix::*;
use crate::nn::activation_functions::*;

pub struct InputLayer {
    pub values: Matrix
}

pub struct HiddenLayer {
    pub biases: Matrix,
    pub weights: Matrix,
    pub activation: Activation
}

pub struct OutputLayer {
    pub biases: Matrix,
    pub weights: Matrix,
    pub activation: OutputActivation
}

impl InputLayer {
    pub fn new_hidden_layer(&self, neuron_count: usize, activation: Activation) -> HiddenLayer {
        assert!(self.values.get_element_count() > 0);

        let weights = Matrix::new_randomized(self.values.rows, neuron_count);
        let biases = Matrix::from_vec(vec![0f64; neuron_count], neuron_count, 1);

        HiddenLayer {
            biases,
            weights,
            activation
        }
    }

    pub fn forward_to_hidden(&self, next_layer: &HiddenLayer) -> Matrix {        
        let partial = self.values
            .mul(&next_layer.weights)
            .add(&next_layer.biases)
            .map(next_layer.activation.function);

        partial
    }
}

impl HiddenLayer {
    /// Instantiates and returns a new Hidden Layer based on the provided hidden layer.
    /// # Arguments
    /// # Returns
    pub fn new_hidden_layer(&self, neuron_count: usize, activation: Activation) -> HiddenLayer {
        assert!(self.weights.get_element_count() > 0);

        let weights = Matrix::new_randomized(self.weights.rows, neuron_count);
        let biases = Matrix::from_vec(vec![0.0; neuron_count], neuron_count, 1);

        HiddenLayer {
            weights,
            biases,
            activation
        }
    }

    /// Instantiates and returns a new Output Layer.
    pub fn new_output_layer(&self, neuron_count: usize, activation: OutputActivation) -> OutputLayer {
        assert!(self.weights.get_element_count() > 0);

        let weights = Matrix::new_randomized(self.weights.rows, neuron_count);
        let biases = Matrix::from_vec(vec![0.0; neuron_count], neuron_count, 1);

        OutputLayer {
            biases,
            weights,
            activation
        }
    }

    pub fn forward_to_hidden(&self, next_layer: &HiddenLayer, previous_values: &Matrix ) -> Matrix {        
        let partial = previous_values
            .mul(&next_layer.weights)
            .add(&next_layer.biases)
            .map(next_layer.activation.function);

        partial
    }

    pub fn forward_to_output(&self, output_layer: &OutputLayer, previous_values: &Matrix) -> Matrix {
        let partial = previous_values
            .mul(&output_layer.weights)
            .add(&output_layer.biases);

        //(output_layer.activation.function)();
            //.map(next_layer.activation.function);
        
        partial
    }

    // pub fn back_from_hidden(&mut self, previous_layer: &HiddenLayer) {
    //     ();
    // }
}