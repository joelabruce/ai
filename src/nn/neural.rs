use crate::digit_image::DigitImage;
use crate::nn::layers::*;
use crate::nn::activation_functions::*;
use crate::geoalg::f64_math::matrix::*;
use crate::input_csv_reader::*;
use crate::sample::Sample;

pub struct Neural {
    pub nodes: Vec<Node>
}

pub enum Node {
    HiddenLayer(HiddenLayer),
    Activation(Activation)
}

impl Neural {
    pub fn open_for_importing(file_path: &str) -> InputCsvReader {
        let reader = InputCsvReader::new(file_path);

        reader
    }

    /// Creates DigitImage Sample from CVS file
    pub fn create_sample_for_digit_images_from_file(reader: &mut InputCsvReader, total_size: usize) -> Sample<DigitImage> {
        let mut data = vec![];   // Normalized data
        for _sample in 0..total_size {
            let digit_image = reader.read_and_parse_data_line(784);
            data.push(digit_image);
        }

        Sample::create_sample(data)
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

pub fn handwritten_digits() {
    // Hidden layer 1
    let dense1 = HiddenLayer::new(784, 128);
    let dense2 = dense1.calulated_hidden_layer(64);
    let dense3 = dense2.calulated_hidden_layer(10);

    // Add layers to the network for forward and backward propagation.
    let mut training_nodes: Vec<Node> = Vec::new();
    training_nodes.push(Node::HiddenLayer(dense1));
    training_nodes.push(Node::Activation(RELU));
    training_nodes.push(Node::HiddenLayer(dense2));
    training_nodes.push(Node::Activation(RELU));
    training_nodes.push(Node::HiddenLayer(dense3));

    // Training hyper-parameters
    let total_epochs = 10;
    let training_sample = 60000;
    let batch_size = 96;
    let batches = training_sample / batch_size;
    let v_batch_size = std::cmp::min(batches * batch_size / 5, 9999);        
    let mut lowest_loss = f64::INFINITY;

    // Validtion setup
    let mut testing_reader = Neural::open_for_importing("./training/mnist_test.csv");
    let _ = testing_reader.read_and_skip_header_line();
    let mut testing_sample = Neural::create_sample_for_digit_images_from_file(&mut testing_reader, 10000);
    let (vl, validation_tagets) = InputLayer::from_sample_digit_images(&mut testing_sample, v_batch_size);

    // Training setup
    let mut training_reader = Neural::open_for_importing("./training/mnist_train.csv");
    let _ = training_reader.read_and_skip_header_line();
    let mut training_sample = Neural::create_sample_for_digit_images_from_file(&mut training_reader, training_sample);

    // Create Layers in network
    let mut forward_stack: Vec<Matrix>;

    for epoch in 1..=total_epochs {
        training_sample.reset();
        for batch in 0..batches {
            let (il, targets) = InputLayer::from_sample_digit_images(&mut training_sample, batch_size);
            forward_stack = Neural::forward(&il, &mut training_nodes);
 
            // Forward pass on training data btch
            let predictions = &(SOFTMAX.f)(&forward_stack.pop().unwrap());
            let sample_losses = forward_categorical_cross_entropy_loss(&predictions, &targets);
            let data_loss = sample_losses.values.iter().copied().sum::<f64>() / sample_losses.len() as f64;            
            let dvalues6 = backward_categorical_cross_entropy_loss_wrt_softmax(&predictions, &targets).div_by_scalar(batch_size as f64);
            
            // Backward pass on training data batch
            Neural::backward(&mut training_nodes, &dvalues6, &mut forward_stack);

            if batch % 100 == 0 {
                println!("<Checkpoint> Training on batch #{batch} | Data Loss: {data_loss}");
            }
        }

        print!("Epoch #{epoch} completed...");

        // Validate updated neural network against validation inputs it hasn't been trained on.
        forward_stack = Neural::forward(&vl, &mut training_nodes);
        let v_predictions = &(SOFTMAX.f)(&forward_stack.pop().unwrap());
        let v_sample_losses = forward_categorical_cross_entropy_loss(&v_predictions, &validation_tagets);
        let v_data_loss = v_sample_losses.values.iter().copied().sum::<f64>() / v_sample_losses.len() as f64;

        println!("| Validation Loss {v_data_loss}");
        
        if v_data_loss < lowest_loss { lowest_loss = v_data_loss } else { println!("Warning, validation loss increased! Consider stopping training here."); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[ignore = "Needs mnist_train.csv to train on."]
    #[test]
    fn nn_test() {
        handwritten_digits();
    }
}