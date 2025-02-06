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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nn_test() {
    }
}
