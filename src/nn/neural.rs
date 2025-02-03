use std::fs::File;
use std::io::Read;

use dense_layer::DenseLayer;
use input_layer::InputLayer;

use crate::digit_image::DigitImage;
use crate::nn::layers::*;
use crate::nn::activation_functions::*;
use crate::geoalg::f32_math::matrix::*;
use crate::input_csv_reader::*;
use crate::output_bin_writer::OutputBinWriter;
use crate::statistics::sample::Sample;

pub struct NeuralNetwork { }

pub enum Node {
    HiddenLayer(DenseLayer),
    Activation(Activation)
}

/// Creates an input layer drawn randomly from a sample.
pub fn from_sample_digit_images(sample: &mut Sample<DigitImage>, requested_batch_size: usize) -> (InputLayer, Matrix) {
    let data_from_sample = sample.random_batch(requested_batch_size);

    let mut pixel_vector = Vec::with_capacity(data_from_sample.len() * 785);
    let mut taget_vector = Vec::with_capacity(data_from_sample.len() * 10);
    let rows = data_from_sample.len();
    for datum in data_from_sample {
        pixel_vector.extend(datum.pixels.clone());
        taget_vector.extend(datum.one_hot_encoded_label());
    }

    (InputLayer {
        input_matrix: Matrix::from(rows, 784, pixel_vector)
    }, Matrix::from(rows, 10, taget_vector))
}


impl NeuralNetwork {
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
    pub fn forward(with_input: InputLayer, to_nodes: &mut Vec<Node>) -> Vec<Matrix> {
        let mut forward_stack = Vec::with_capacity(to_nodes.len() + 1);

        forward_stack.push(with_input.input_matrix);
        for node in to_nodes.iter_mut() {
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

    pub fn save_network(from_nodes: &Vec<Node>, to_writer: &mut OutputBinWriter) {
        for node in from_nodes {
            match node {
                Node::HiddenLayer(n) => {
                    to_writer.write_slice_f32(&n.weights.read_values());
                    to_writer.write_slice_f32(&n.biases.read_values());
                }
                _ => { }
            }
        }
    }

    pub fn attempt_load_network(from_file_path: &str, to_nodes: &mut Vec<Node>) {
        let file_open_try = File::open(from_file_path);

        let chunk_size = 4; // 4 for 32-bit, 8 for 64-bit

        match file_open_try {
            Ok(mut file) => {   // File could be opened for read
                for node in to_nodes {
                    match node {
                        Node::HiddenLayer(n) => {
                            // Load weights first
                            let mut weights_buf = vec![0u8; n.weights.len() * chunk_size];
                            let mut columns = n.weights.column_count();
                            let mut rows = n.weights.row_count();

                            file.read_exact(&mut weights_buf).expect("Should not error reading in weights for a layer.");
                            let weights_floats: Vec<f32> = weights_buf
                                .chunks_exact(chunk_size)
                                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                                .collect();
                            n.weights = Matrix::from(rows, columns, weights_floats);

                            // Load biases next
                            columns = n.biases.column_count();
                            rows = n.biases.row_count();
                            let mut biases_buf = vec![0u8; n.biases.len() * chunk_size];
                            file.read_exact(&mut biases_buf).expect("Should not error reading biases for a layer.");
                            let biases_floats: Vec<f32> = biases_buf
                                .chunks_exact(chunk_size)
                                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                                .collect();
                            n.biases = Matrix::from(rows, columns, biases_floats);
                            println!("Loaded weights and biases for dense layer.")
                        }
                        _ => { }
                    }
                }

                println!("Success in loading the neural network!");
            }
            _ => {}
        }
    }
}

pub fn handwritten_digits(load_from_file: bool) {
    // Create dense layers
    let dense1 = DenseLayer::new(784, 128);
    let dense2 = dense1.calulated_dense_layer(64);
    let dense3 = dense2.calulated_dense_layer(10);

    // Add layers to the network for forward and backward propagation.
    let mut training_nodes: Vec<Node> = Vec::new();
    training_nodes.push(Node::HiddenLayer(dense1));
    training_nodes.push(Node::Activation(RELU));
    training_nodes.push(Node::HiddenLayer(dense2));
    training_nodes.push(Node::Activation(RELU));
    training_nodes.push(Node::HiddenLayer(dense3));

    let trained_model_location = "./tests/network_training.nn";
    if load_from_file {
        println!("Trying to load trained neural network...");
        NeuralNetwork::attempt_load_network(&trained_model_location, &mut training_nodes);
    }

    // Training hyper-parameters
    let total_epochs = 4;
    let training_sample = 60000;
    let batch_size = 500;
    let batches = training_sample / batch_size;
    let v_batch_size = std::cmp::min(batches * batch_size / 5, 9999);        
    let mut lowest_loss = f32::INFINITY;

    // Validtion setup
    let mut testing_reader = NeuralNetwork::open_for_importing("./training/mnist_test.csv");
    let _ = testing_reader.read_and_skip_header_line();
    let mut testing_sample = NeuralNetwork::create_sample_for_digit_images_from_file(&mut testing_reader, 10000);
    let (vl, validation_tagets) = from_sample_digit_images(&mut testing_sample, v_batch_size);

    // Training setup
    let mut training_reader = NeuralNetwork::open_for_importing("./training/mnist_train.csv");
    let _ = training_reader.read_and_skip_header_line();
    let mut training_sample = NeuralNetwork::create_sample_for_digit_images_from_file(&mut training_reader, training_sample);

    // Create Layers in network
    let mut forward_stack: Vec<Matrix>;

    for epoch in 1..=total_epochs {
        training_sample.reset();
        for batch in 0..batches {
            let (il, targets) = from_sample_digit_images(&mut training_sample, batch_size);
            forward_stack = NeuralNetwork::forward(il, &mut training_nodes);
 
            // Forward pass on training data btch
            let predictions = (SOFTMAX.f)(&forward_stack.pop().unwrap());
            let sample_losses = forward_categorical_cross_entropy_loss(&predictions, &targets);
            let data_loss = sample_losses.read_values().into_iter().sum::<f32>() / sample_losses.len() as f32;            
            
            // Backward pass on training data batch
            let dvalues6 = backward_categorical_cross_entropy_loss_wrt_softmax(&predictions, &targets).scale(1. / batch_size as f32);
            NeuralNetwork::backward(&mut training_nodes, &dvalues6, &mut forward_stack);

            if batch % 50 == 0 {
                print!("Training to batch #{batch} complete | Data Loss: {data_loss}");
                
                let accuracy = accuracy(&predictions, &targets);
                println!(" | Accuracy: {accuracy}");
            }
        }

        print!("Epoch #{epoch} completed. Saving...");
        let mut network_saver = OutputBinWriter::new(&format!("{trained_model_location}"));
        NeuralNetwork::save_network(&training_nodes, &mut network_saver);

        // Validate updated neural network against validation inputs it hasn't been trained on.
        // Clone the validation layer, so it is not consumed
        // Better to clone here than cloning for each iteration of the batches being trained on for performance.
        forward_stack = NeuralNetwork::forward(vl.clone(), &mut training_nodes);
        let v_predictions = &(SOFTMAX.f)(&forward_stack.pop().unwrap());
        let v_sample_losses = forward_categorical_cross_entropy_loss(&v_predictions, &validation_tagets);
        let v_data_loss = v_sample_losses.read_values().into_iter().sum::<f32>() / v_sample_losses.len() as f32;

        print!("| Validation Loss: {v_data_loss}");

        let accuracy = accuracy(&v_predictions, &validation_tagets);
        println!(" | Accuracy: {accuracy}");
        
        if v_data_loss < lowest_loss { lowest_loss = v_data_loss } else { println!("Warning, validation has not improved! Consider stopping training here."); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[ignore = "Needs mnist_train.csv to train on."]
    #[test]
    fn nn_test() {
        handwritten_digits(true);
    }
}