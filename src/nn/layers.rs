use rand::distributions::Uniform;

use crate::geoalg::f64_math::matrix::*;
use crate::nn::activation_functions::*;

/// Expected values for training of inputs.
pub struct Targets {
    pub values: Matrix
}

/// Allows for creation of succeeding layers based on initial data passed in.
pub struct InputLayer {
    pub values: Matrix
}

/// Should be created beginning with InputLayer to ensure shapes are correct.
pub struct HiddenLayer {
    pub biases: Matrix,
    pub weights: Matrix,
    //internal_state: Matrix
}

impl InputLayer {
    /// Creates input layer based on inputs supplied.
    /// Allows for automatic shaping of succeeding layer generation.
    pub fn from_vec(values: Vec<Vec<f64>>) -> InputLayer {
        let rows = values.len();
        let flat_v: Vec<f64> = values.into_iter().flat_map(|v| v).collect();

        let columns = flat_v.len() / rows;

        let mat = Matrix {
            rows,
            columns,
            values: flat_v
        };

        let mat = mat.get_transpose();
        InputLayer {
            values: mat
        }
    } 

    /// Creates a new hidden layer based on the shape of the input layer.
    /// # Arguments
    /// # Returns
    pub fn new_hidden_layer(&self, neuron_count: usize) -> HiddenLayer {
        assert!(self.values.get_element_count() > 0);

        let uniform = Uniform::new_inclusive(-0.1, 0.1);
        let weights = Matrix::new_randomized_uniform(self.values.rows, neuron_count, uniform);
        let biases = Matrix::from_vec(vec![0f64; neuron_count], neuron_count, 1);
        
        HiddenLayer {
            biases,
            weights
        }
    }

    /// Input layer always forwards to first hidden layer and not to an activation function.
    /// Performs weights dot inputs + biases.
    /// # Arguments
    /// # Returns
    pub fn forward(&self, next_layer: &HiddenLayer) -> Matrix {
        let output = next_layer.weights
            .mul(&self.values)
            .add_column_vector(&next_layer.biases);

        output
    }
}

impl HiddenLayer {
    /// Instantiates and returns a new Hidden Layer.
    /// # Arguments
    /// # Returns
    pub fn new_hidden_layer(&self, neuron_count: usize) -> HiddenLayer {
        assert!(self.weights.get_element_count() > 0);

        let weights = Matrix::new_randomized(self.weights.rows, neuron_count);
        let biases = Matrix::from_vec(vec![0.0; neuron_count], neuron_count, 1);

        HiddenLayer {
            weights,
            biases
        }
    }

    /// Forwards inputs to non-folding activation function.
    /// # Arguments
    /// # Returns
    pub fn forward<'a>(&self, activation: &Activation, inputs: &'a Matrix) -> Matrix {
        inputs.map(activation.f)
    }

    /// Forwards inputs to folding activation function.
    /// # Arguments
    /// # Returns
    pub fn forward_fold<'a>(&self, activation: &FoldActivation, inputs: &'a Matrix) -> Matrix {
        (activation.f)(&inputs)
    }

    pub fn backward<'a>(&mut self, dvalues: &Matrix, inputs: &Matrix) {
        let t = inputs.get_transpose();
        t.mul(dvalues);
    }
}

#[cfg(test)]
mod tests {
     use super::*;

    #[test]
    fn test() {
        let input_count = 2;

        let raw_inputs = vec![
            vec![0.0; 784]
            , vec![0.0; 784]
        ];

        let targets = Matrix {
            rows: 10,
            columns: input_count,
            values: vec![
                1.0, 0.0,
                0.0, 0.0,
                0.0, 1.0,
                0.0, 0.0,
                0.0, 0.0,
                0.0, 0.0,
                0.0, 0.0,
                0.0, 0.0,
                0.0, 0.0,
                0.0, 0.0]
        };

        // Create Layers in network
        let il = InputLayer::from_vec(raw_inputs);
        assert_eq!(il.values.columns, input_count);
        assert_eq!(il.values.rows, 784);
        assert_eq!(il.values.get_element_count(), 784 * input_count);

        let dense1 = il.new_hidden_layer(128);
        assert_eq!(dense1.weights.columns, 784);
        assert_eq!(dense1.weights.rows, 128);
        assert_eq!(dense1.weights.get_element_count(), 100352);

        // This is also the output layer
        let mut dense2 = dense1.new_hidden_layer(10);
        assert_eq!(dense2.weights.columns, 128);
        assert_eq!(dense2.weights.rows, 10);
        assert_eq!(dense2.weights.get_element_count(), 1280);

        // Sanity check to make sure the output shapes are correct.
        // Forward propagation should not be able to mutate anything, only backpropagation can do that.
        let fcalc1 = il.forward(&dense1);
        let fcalc2 = dense1.forward(&RELU, &fcalc1);
        let fcalc3 = RELU.forward(&dense2, &fcalc2);
        
        let predictions = dense2.forward_fold(&SOFTMAX, &fcalc3);
        println!("Predictions: {predictions}");
        assert_eq!(predictions.columns, input_count);
        assert_eq!(predictions.rows, 10);

        let loss = forward_categorical_cross_entropy_loss(&predictions, &targets);
        println!("Loss vector: {:?}", loss);

        let gradient = backward_categorical_cross_entropy_loss(&predictions, &targets, input_count);
        println!("Gradient: {gradient}");

        dense2.backward(&gradient, &fcalc3);
        //let errors = OutputLayer::derivative_cross_entropy_loss_wrt_softmax(&predictions, &targets, input_count);
    }
}