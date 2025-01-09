use rand::distributions::Uniform;

use crate::geoalg::f64_math::matrix::*;

pub fn learning_rate() -> f64 {
    0.5
}

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
    pub weights: Matrix
}

impl InputLayer {
    /// Creates input layer based on inputs supplied.
    /// Allows for automatic shaping of succeeding layer generation.
    pub fn from_vec(values: Vec<Vec<f64>>) -> InputLayer {
        let rows = values.len();    // Number of samples in batch
        let flat_v: Vec<f64> = values.into_iter().flat_map(|v| v).collect();
        let columns = flat_v.len() / rows;  // Number of features.

        let values = Matrix {
            rows,
            columns,
            values: flat_v
        };

        InputLayer {
            values
        }
    } 

    /// Creates a new hidden layer based on the shape of the input layer.
    /// # Arguments
    /// # Returns
    pub fn new_hidden_layer(&self, neuron_count: usize) -> HiddenLayer {
        assert!(self.values.get_element_count() > 0);

        let uniform = Uniform::new_inclusive(-0.1, 0.1);
        let weights = Matrix::new_randomized_uniform(self.values.columns, neuron_count, uniform);
        let biases = Matrix::from_vec(vec![0f64; neuron_count], 1, neuron_count);
        
        HiddenLayer {
            biases,
            weights
        }
    }

    // Input layer always forwards to first hidden layer and not to an activation function.
    // # Arguments
    // # Returns
    pub fn forward(&self) -> Matrix {
        self.values.clone()
    }

}

impl HiddenLayer {
    /// Instantiates and returns a new Hidden Layer.
    /// # Arguments
    /// # Returns
    pub fn new_hidden_layer(&self, neuron_count: usize) -> HiddenLayer {
        assert!(self.weights.get_element_count() > 0);

        let uniform = Uniform::new_inclusive(-0.1, 0.1);
        let weights = Matrix::new_randomized_uniform(self.weights.columns, neuron_count, uniform);
        let biases = Matrix::from_vec(vec![0.0; neuron_count], 1, neuron_count);

        HiddenLayer {
            weights,
            biases
        }
    }

    /// Forward propagates by performing weights dot inputs + biases.
    /// Z
    pub fn forward<'a>(&self, inputs: &'a Matrix) -> Matrix {
        let r = inputs
            .mul(&self.weights)
            .add_row_vector(&self.biases);
        r
    }

    /// In progress to support ADAM optimization
    pub fn backward<'a>(& mut self, dvalues: &Matrix, inputs: &Matrix) -> Matrix {
        let shape_inputs = inputs.shape();
        let shape_dvalues = dvalues.shape();
        println!("X: {shape_inputs}, dZ: {shape_dvalues}");

        let shape_weights = self.weights.shape();
        let dweights = inputs.get_transpose().mul(dvalues);
        let shape_dweights = dweights.shape();
        println!("W: {shape_weights}, dW: {shape_dweights}");

        // Mutate the weights
        self.weights = self.weights.sub(&dweights).scale(learning_rate());

        let dbiases = dvalues.shrink_rows_by_add();
        let db_shape = dbiases.shape();
        let biases_shape = self.biases.shape();
        println!("biases: {biases_shape}, db_shape: {db_shape}");

        // Mutate the biases
        self.biases = self.biases.sub(&dbiases).scale(learning_rate());

        let x= dvalues.mul(&self.weights.get_transpose());
        let x_shape = x.shape();
        println!("X: {x_shape}");
        println!();

        x
    }
}

#[cfg(test)]
mod tests {
     use super::*;
     use crate::{input_csv_reader::InputCsvReader, nn::activation_functions::*};

    #[ignore = "not finished"]
    #[test]
    fn test() {
        let mut reader = InputCsvReader::new("./training/mnist_train.csv");
        let _ = reader.read_and_skip_header_line();

        let mut target_ohes = vec![];
        let mut raw_inputs = vec![];
        let input_count = 5;

        for _sample in 0..input_count {
            let (v, label) = reader.read_and_parse_data_line(784);
            raw_inputs.push(v);
            target_ohes.extend(one_hot_encode(label, 10));
        }

        let targets = Matrix {
            rows: input_count,
            columns: 10,
            values: target_ohes
        };

        // Create Layers in network
        let il = InputLayer::from_vec(raw_inputs);
        let shape_il = il.values.shape();
        println!("input layer shape: {shape_il}");
        assert_eq!(il.values.rows, input_count);
        assert_eq!(il.values.columns, 784);
        assert_eq!(il.values.get_element_count(), 784 * input_count);

        let mut dense1 = il.new_hidden_layer(128);
        assert_eq!(dense1.weights.rows, 784);
        assert_eq!(dense1.weights.columns, 128);
        assert_eq!(dense1.weights.get_element_count(), 100352);

        // This is also the output layer
        let mut dense2 = dense1.new_hidden_layer(10);
        assert_eq!(dense2.weights.rows, 128);
        assert_eq!(dense2.weights.columns, 10);
        assert_eq!(dense2.weights.get_element_count(), 1280);

        // Sanity check to make sure the output shapes are correct.
        // Forward propagation should not be able to mutate anything, only backpropagation can do that.
        let inputs = il.forward();                                  // input layer -> 
        let fcalc1 = dense1.forward(&inputs);                       // dense1 ->
        let fcalc2 = RELU.forward(&fcalc1);                 // RELU ->
        let fcalc3 = dense2.forward(&fcalc2);               // dense2 -> 
        let predictions = (SOFTMAX.f)(&fcalc3);                     // softmax ->

        //println!("Predictions: {:?}", predictions);
        assert_eq!(predictions.columns, 10);
        assert_eq!(predictions.rows, input_count);

        // Used to calculate accuracy and if progress is being made.
        // Not being used yet.
        let _loss = forward_categorical_cross_entropy_loss(&predictions, &targets);
        //println!("Loss vector: {:?}", loss);

        // Start backpropagating.
        let dvalues4 = backward_categorical_cross_entropy_loss_wrt_softmax(&predictions, &targets); // loss ->
        let dvalues4 = dvalues4.div_by_scalar(input_count as f64);   // Don't forget to scale by batch size

        // Can use simple gradient descent now! Woohoo!
        // Will implement adam optimization later. Whew
        let dvalues3 = dense2.backward(&dvalues4, &fcalc2);
        let dvalues2 = RELU.backward(&dvalues3, &fcalc1);
        let _dvalues1 = dense1.backward(&dvalues2, &il.values);
    }
}