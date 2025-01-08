use rand::distributions::Uniform;

use crate::geoalg::f64_math::matrix::*;

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
        let rows = values.len();
        let flat_v: Vec<f64> = values.into_iter().flat_map(|v| v).collect();

        let columns = flat_v.len() / rows;  // Should be 784 for images

        let mat = Matrix {
            rows,
            columns,
            values: flat_v
        };

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
        let weights = Matrix::new_randomized_uniform(neuron_count, self.values.columns, uniform);
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
        let weights = Matrix::new_randomized_uniform(neuron_count, self.weights.columns, uniform);
        let biases = Matrix::from_vec(vec![0.0; neuron_count], 1, neuron_count);

        HiddenLayer {
            weights,
            biases
        }
    }

    /// Forward propagates by performing weights dot inputs + biases.
    pub fn forward<'a>(&self, inputs: &'a Matrix) -> Matrix {
        let r = inputs
            .mul(&self.weights)
            .add_row_vector(&self.biases);
        r
    }

    /// In progress to support ADAM optimization
     pub fn backward<'a>(&self, dvalues: &Matrix, _inputs: &Matrix) -> Matrix {
        //let t = inputs.get_transpose();
        
        //let dweights = t.mul(dvalues);
        //let dbiases = dvalues
        dvalues.mul(&self.weights.get_transpose())
    }
}

#[cfg(test)]
mod tests {
     use super::*;
     use crate::nn::activation_functions::*;

    #[test]
    fn test() {
        let input_count = 2;

        let raw_inputs = vec![
            vec![0.5; 784]
            , vec![0.5; 784]
        ];

        let targets = Matrix {
            rows: input_count,
            columns: 10,
            values: vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        };

        // Create Layers in network
        let il = InputLayer::from_vec(raw_inputs);
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

        println!("Predictions: {predictions}");
        assert_eq!(predictions.columns, 10);
        assert_eq!(predictions.rows, input_count);

        // Used to calculate accuracy and if progress is being made.
        let loss = forward_categorical_cross_entropy_loss(&predictions, &targets);
        println!("Loss vector: {:?}", loss);

        // Start backpropagating.
        let dvalues4 = backward_categorical_cross_entropy_loss_wrt_softmax(&predictions, &targets); // loss ->
        let scaled_dv4 = dvalues4.div_by_scalar(input_count as f64);
        println!("Dvalues for back cross_entropy wrt softmax: {scaled_dv4}");        

        // Can use simple gradient descent now! Woohoo!
        // Will implement adam optimization later. Whew
        let dvalues3 = dense2.backward(&dvalues4, &fcalc3);
        let dvalues2 = RELU.backward(&dvalues3, &fcalc2);
        let dvalues1 = dense1.backward(&dvalues2, &fcalc1);
    }
}