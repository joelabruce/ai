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
    pub weights: Matrix
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

        let weights = Matrix::new_randomized(self.values.rows, neuron_count);
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

    pub fn backward<'a>(&self) {

    }
}

    /// Calculates the cross-entropy (used with softmax) for each input sample.
    pub fn cross_entropy_loss(predictions: &Matrix, expected: &Matrix) -> Vec<f64> {
        let t = expected.get_transpose();

        let mut r = Vec::with_capacity(t.rows);
        for row in 0..t.rows {
            let (index, _) = t.get_row_vector_slice(row)
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap();

            let loss = -predictions.get(index, row).unwrap().log10();

            r.push(loss);
        }

        r
    }

    /// Error, same as softmax derivative??
    pub fn derivative_cross_entropy_loss_wrt_softmax(predictions: &Matrix, expected: &Matrix, samples: usize) -> Matrix {
        predictions.sub(&expected).div_by_scalar(samples as f64)
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

        // Create Layers in network
        // When doing calculations, we get transposed row to make it a column for calculations
        let il = InputLayer::from_vec(raw_inputs);
        assert_eq!(il.values.columns, input_count);
        assert_eq!(il.values.rows, 784);
        assert_eq!(il.values.get_element_count(), 784 * input_count);

        let hl1 = il.new_hidden_layer(128);
        //println!("{:?}", hl1.weights);
        assert_eq!(hl1.weights.columns, 784);
        assert_eq!(hl1.weights.rows, 128);
        assert_eq!(hl1.weights.get_element_count(), 100352);

        let hl2 = hl1.new_hidden_layer(20);
        //println!("{:?}", hl2.weights);
        assert_eq!(hl2.weights.columns, 128);
        assert_eq!(hl2.weights.rows, 20);
        assert_eq!(hl2.weights.get_element_count(), 2560);

        let outlayer = hl2.new_hidden_layer(10);
        //println!("{:?}", ol.weights);
        assert_eq!(outlayer.weights.columns, 20);
        assert_eq!(outlayer.weights.rows, 10);
        assert_eq!(outlayer.weights.get_element_count(), 200);

        // Sanity check to make sure the output shapes are correct.
        let inputs1 = il.forward(&hl1);
        assert_eq!(inputs1.columns, input_count);
        assert_eq!(inputs1.rows, 128);

        let inputs2 = hl1.forward(&RELU, &inputs1);
        let inputs3 = RELU.forward(&hl2, &inputs2);

        let inputs4 = hl2.forward(&RELU, &inputs3);
        let inputs5 = RELU.forward(&outlayer, &inputs4);
        
        let predictions = outlayer.forward_fold(&SOFTMAX, &inputs5);
        println!("{:?}", predictions);
        assert_eq!(predictions.columns, input_count);
        assert_eq!(predictions.rows, 10);

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

        let loss = cross_entropy_loss(&predictions, &targets);
        println!("Loss vector: {:?}", loss);

        //let errors = OutputLayer::derivative_cross_entropy_loss_wrt_softmax(&predictions, &targets, input_count);
    }
}