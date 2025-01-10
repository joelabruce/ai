use rand::distributions::Uniform;
use crate::geoalg::f64_math::matrix::*;

pub fn learning_rate() -> f64 {
    0.01
}

pub trait Propagates {
    fn forward(&self, inputs: &Matrix) -> Matrix;
    fn backward<'a>(&'a mut self, dvalues: &Matrix, inputs: &Matrix) -> Matrix;
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

        let uniform = Uniform::new_inclusive(-0.5, 0.5);
        let weights = Matrix::new_randomized_uniform(self.values.columns, neuron_count, uniform);
        let biases = Matrix::new_zeroed(1,neuron_count);
        
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

        let uniform = Uniform::new_inclusive(-0.5, 0.5);
        let weights = Matrix::new_randomized_uniform(self.weights.columns, neuron_count, uniform);
        let biases = Matrix::from_vec(vec![0.0; neuron_count], 1, neuron_count);

        HiddenLayer {
            weights,
            biases
        }
    }
}

impl Propagates for HiddenLayer {
    /// Forward propagates by performing weights dot inputs + biases.
    /// Z
    fn forward<'a>(&self, inputs: &'a Matrix) -> Matrix {
        let r = inputs
            .mul(&self.weights)
            .add_row_vector(&self.biases);
        r
    }

    /// In progress to support ADAM optimization
    fn backward<'a>(& mut self, dvalues: &Matrix, inputs: &Matrix) -> Matrix {
        // let shape_inputs = inputs.shape();
        // let shape_dvalues = dvalues.shape();
        // println!("X: {shape_inputs}, dZ: {shape_dvalues}");

        // let shape_weights = self.weights.shape();
        let dweights = inputs.get_transpose().mul(dvalues);
        // let shape_dweights = dweights.shape();
        // println!("W: {shape_weights}, dW: {shape_dweights}");

        // Mutate the weights
        self.weights = self.weights.sub(&dweights.scale(learning_rate()));

        let dbiases = dvalues.shrink_rows_by_add();
        // let db_shape = dbiases.shape();
        // let biases_shape = self.biases.shape();
        // println!("biases: {biases_shape}, db_shape: {db_shape}");

        // Mutate the biases
        self.biases = self.biases.sub(&dbiases.scale(learning_rate()));

        let x= dvalues.mul(&self.weights.get_transpose());
        // let x_shape = x.shape();
        // println!("X: {x_shape}");
        // println!();

        x
    }
}
