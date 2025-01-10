use crate::nn::layers::*;
use crate::nn::activation_functions::*;
use crate::geoalg::f64_math::matrix::*;
use crate::input_csv_reader::*;

pub struct Neural {
    pub nodes: Vec<Node>
}

pub enum Node {
    HiddenLayer(HiddenLayer),
    Activation(Activation)
}

impl Neural {
    pub fn import_inputs(file_path: &str, batch_size: usize) -> (Vec<Vec<f64>>, Matrix) {
        // Open file for reading to create training data.
        let mut reader = InputCsvReader::new(file_path);
        let _ = reader.read_and_skip_header_line();          // Discard header

        let mut target_ohes = vec![];              // One-hot encoding of the targets
        let mut normalized_inputs = vec![];   // Normalized data

        for _sample in 0..batch_size {
            let (v, label) = reader.read_and_parse_data_line(784);
            normalized_inputs.push(v);
            target_ohes.extend(one_hot_encode(label, 10));
        }

        let targets = Matrix {
            rows: batch_size,
            columns: 10,
            values: target_ohes
        };

        (normalized_inputs, targets)
    }

    pub fn get_next_batch() {

    }

    /// Forward propagates the inputs through the layers.
    /// Calculates a Vec of matrices to be used for backpropagation.
    pub fn forward(with_input: &InputLayer, to_nodes: &mut Vec<Node>) -> Vec<Matrix> {
        let mut forward_stack = Vec::with_capacity(to_nodes.len() + 1);

        forward_stack.push(with_input.forward());
        for node in to_nodes.iter() {
            match node {
                Node::Activation(n) => forward_stack.push(n.forward(forward_stack.last().unwrap())),
                Node::HiddenLayer(n) => forward_stack.push(n.forward(forward_stack.last().unwrap()))
            }
        } 

        forward_stack
    }

    /// Applies backpropagation.
    /// Pops the items off of fcalcs, but keeps nodes in the Vec so we can do the next forward pass.
    pub fn backward(from_nodes: &mut Vec<Node>, dz: &Matrix, fcalcs: &mut Vec<Matrix>) {
        let mut dvalues = dz.clone();
        
        for i in (0..from_nodes.len()).rev() {
            let node_opt = from_nodes.get_mut(i);

            if let Some(node) = node_opt {
                match node {
                    Node::Activation(n) => {
                        let fcalc = fcalcs.pop().unwrap();
                        dvalues = n.backward(&dvalues, &fcalc);
                    },
                    Node::HiddenLayer(n) => {
                        let fcalc = fcalcs.pop().unwrap();
                        dvalues = n.backward(&dvalues, &fcalc);
                    }
                };
            };
       }
    }
}

#[cfg(test)]
mod tests {
    use core::f64;

    use super::*;

    #[ignore = "Needs mnist_train.csv to train on."]
    #[test]
    fn test() {
        // Read from specified files for creating neural network
        let batch_size = 100;
        let (normalized_inputs, targets) = Neural::import_inputs("./training/mnist_train.csv", batch_size);
        let (validation_inputs, validation_tagets) = Neural::import_inputs("./training/mnist_test.csv", 200);

        // Create Layers in network
        let il = InputLayer::from_vec(normalized_inputs);
        assert_eq!(il.values.rows, batch_size);
        assert_eq!(il.values.columns, 784);
        assert_eq!(il.values.get_element_count(), 784 * batch_size);

        // Since this is an input layer also, can use instead of normalized_inputs when calculating validation loss
        let vl = InputLayer::from_vec(validation_inputs);

        // Hidden layer 1
        let mut dense1 = il.new_hidden_layer(128);
        assert_eq!(dense1.weights.rows, 784);
        assert_eq!(dense1.weights.columns, 128);
        assert_eq!(dense1.weights.get_element_count(), 100352);

        // Hidden layer 2
        let mut dense2 = dense1.new_hidden_layer(64);
        assert_eq!(dense2.weights.rows, 128);
        assert_eq!(dense2.weights.columns, 64);
        assert_eq!(dense2.weights.get_element_count(), 8192);

        // Output layer
        let mut dense3 = dense2.new_hidden_layer(10);

        // Add layers to the network for forward and backward propagation.
        let mut training_nodes: Vec<Node> = Vec::new();
        //training_nodes.push(Node::InputLayer(il));
        training_nodes.push(Node::HiddenLayer(dense1));
        training_nodes.push(Node::Activation(RELU));
        training_nodes.push(Node::HiddenLayer(dense2));
        training_nodes.push(Node::Activation(RELU));
        training_nodes.push(Node::HiddenLayer(dense3));

        // Begin training
        
        let mut lowest_loss = f64::INFINITY;
        for epoch in 0..1000 {
            // Forward pass on training data
            let mut forward_stack = Neural::forward(&il, &mut training_nodes);
            let predictions = &(SOFTMAX.f)(&forward_stack.pop().unwrap());
            let sample_losses = forward_categorical_cross_entropy_loss(&predictions, &targets);
            let data_loss = sample_losses.values.iter().copied().sum::<f64>() / sample_losses.get_element_count() as f64;            
            let dvalues6 = backward_categorical_cross_entropy_loss_wrt_softmax(&predictions, &targets).div_by_scalar(batch_size as f64);
            Neural::backward(&mut training_nodes, &dvalues6, &mut forward_stack);

            // Validate updated neural network against validation inputs it hasn't been trained on.
            forward_stack = Neural::forward(&vl, &mut training_nodes);
            let v_predictions = &(SOFTMAX.f)(&forward_stack.pop().unwrap());
            let v_sample_losses = forward_categorical_cross_entropy_loss(&v_predictions, &validation_tagets);
            let v_data_loss = v_sample_losses.values.iter().copied().sum::<f64>() / v_sample_losses.get_element_count() as f64;

            if epoch % 10 == 0 || epoch < 10 {
                println!("Epoch: {epoch} | Data Loss: {data_loss} | Validation Loss: {v_data_loss}");
                if v_data_loss < lowest_loss {
                    lowest_loss = v_data_loss
                }
                else {
                    // The validation loss has increased which indicates overfitting on training data. 
                    println!("Finished!, consider training on next batch");
                    break;
                }
            }
        }
    }
}