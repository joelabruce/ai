use crate::geoalg::f64_math::matrix::*;
use crate::nn::activation_functions::*;

pub struct Targets {
    pub values: Matrix
}

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

    /// Performs weights dot inputs + biases and then the activation function.
    pub fn forward_to_hidden(&self, next_layer: &HiddenLayer, for_input_row: usize) -> Matrix {
        let output = next_layer.weights
            .mul(&self.values)
            .add_column_vector(&next_layer.biases)
            .map(next_layer.activation.function);

        output
    }
}

impl HiddenLayer {
    /// Instantiates and returns a new Hidden Layer.
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
    /// # Arguments
    /// # Returns
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

    /// Performs next_layer_weights dot previous_output + biases and then the activation function.
    pub fn forward_to_hidden(&self, next_layer: &HiddenLayer, previous_values: &Matrix ) -> Matrix {        
        let partial = next_layer.weights
            .mul(&previous_values)
            .add_column_vector(&next_layer.biases)
            .map(next_layer.activation.function);

        partial
    }

    /// The output from this function is a mtrix containing the predictions of the neural network.
    /// Performs output_layer_weights dot previous_output + biases and then the activation function.
    pub fn forward_to_output(&self, output_layer: &OutputLayer, previous_values: &Matrix) -> Matrix {
        let partial = output_layer.weights
            .mul(&previous_values)
            .add_column_vector(&output_layer.biases);

        let r = (output_layer.activation.function)(&partial);
        
        r
    }

    pub fn backprop_to() {

    }
}

impl OutputLayer {
    pub fn errors(expected: Matrix, predictions: Matrix) -> Matrix {
        expected.sub(&predictions)
    }

    // pub fn gradient(&self, errors: &Matrix) -> Matrix {

    // }

    pub fn backprop_to(&self, hidden_layer: &HiddenLayer, errors: &Matrix) -> Matrix {
        let mut r = self.weights
            .mul(&errors);
            
        r 
    }
}

#[cfg(test)]
mod tests {
     use super::*;

    #[test]
    fn test() {
        let raw_inputs = vec![
            vec![0.0; 784]
            , vec![0.0; 784]
        ];

        let input_count = 2;

        // When doing calculations, we get transposed row to make it a column for calculations
        let il = InputLayer::from_vec(raw_inputs);
        //assert_eq!(il.values.columns, 1);
        //assert_eq!(il.values.rows, 784);
        //assert_eq!(il.values.get_element_count(), 784);

        let hl1 = il.new_hidden_layer(128, RELU);
        //println!("{:?}", hl1.weights);
        assert_eq!(hl1.weights.columns, 784);
        assert_eq!(hl1.weights.rows, 128);
        assert_eq!(hl1.weights.get_element_count(), 100352);

        let hl2 = hl1.new_hidden_layer(20, RELU);
        //println!("{:?}", hl2.weights);
        assert_eq!(hl2.weights.columns, 128);
        assert_eq!(hl2.weights.rows, 20);
        assert_eq!(hl2.weights.get_element_count(), 2560);

        let ol = hl2.new_output_layer(10, SOFTMAX);
        //println!("{:?}", ol.weights);
        assert_eq!(ol.weights.columns, 20);
        assert_eq!(ol.weights.rows, 10);
        assert_eq!(ol.weights.get_element_count(), 200);

        // Sanity check to make sure the output shapes are correct.
        let mut output = il.forward_to_hidden(&hl1, 0);
        //println!("{:?}", output);
        assert_eq!(output.columns, input_count);
        assert_eq!(output.rows, 128);

        output = hl1.forward_to_hidden(&hl2, &output);
        //print!("1 forward: {:?}", output);
        assert_eq!(output.columns, input_count);
        assert_eq!(output.rows, 20);

        output = hl2.forward_to_output(&ol, &output);
        assert_eq!(output.columns, input_count);
        assert_eq!(output.rows, 10);

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

        //println!("{:?}", output);

        let errors = OutputLayer::errors(targets, output);
        assert_eq!(errors.columns, input_count);
        assert_eq!(errors.rows, 10);

        //println!("{:?}", errors);
    }
}