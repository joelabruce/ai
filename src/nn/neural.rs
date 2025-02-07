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

pub enum NeuralNetworkNode {
    HiddenLayer(DenseLayer),
    Activation(Activation)
}

/// Contains structure and hyper-parameters needed for a neural network
pub struct NeuralNetwork { 
    pub epoch: usize,
    pub nodes: Vec<NeuralNetworkNode>
}

impl NeuralNetwork {
    pub fn add_node(mut self, node: NeuralNetworkNode) -> Self {
        self.nodes.push(node);
        self
    }

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
    pub fn forward(with_input: InputLayer, to_nodes: &mut Vec<NeuralNetworkNode>) -> Vec<Matrix> {
        let mut forward_stack = Vec::with_capacity(to_nodes.len() + 1);

        forward_stack.push(with_input.input_matrix);
        for node in to_nodes.iter_mut() {
            match node {
                NeuralNetworkNode::Activation(n) => forward_stack.push(n.forward(forward_stack.last().unwrap())),
                NeuralNetworkNode::HiddenLayer(n) => forward_stack.push(n.forward(forward_stack.last().unwrap()))
            }
        } 

        forward_stack
    }

    /// Applies backpropagation.
    /// Pops the items off of fcalcs, but keeps nodes in the Vec so we can do the next forward pass.
    pub fn backward(from_nodes: &mut Vec<NeuralNetworkNode>, dz: &Matrix, fcalcs: &mut Vec<Matrix>) {
        let mut dvalues = dz.clone();
        
        for i in (0..from_nodes.len()).rev() {
            let node_opt = from_nodes.get_mut(i);

            if let Some(node) = node_opt {
                match node {
                    NeuralNetworkNode::Activation(n) => {
                        let fcalc = fcalcs.pop().unwrap();
                        dvalues = n.backward(&dvalues, &fcalc);
                    },
                    NeuralNetworkNode::HiddenLayer(n) => {
                        let fcalc = fcalcs.pop().unwrap();
                        dvalues = n.backward(&dvalues, &fcalc);
                    }
                };
            };
       }
    }

    pub fn save_network(epochs: usize, from_nodes: &Vec<NeuralNetworkNode>, to_writer: &mut OutputBinWriter) {
        to_writer.write_usize(epochs);
        for node in from_nodes {
            match node {
                NeuralNetworkNode::HiddenLayer(n) => {
                    to_writer.write_slice_f32(&n.weights.read_values());
                    to_writer.write_slice_f32(&n.biases.read_values());
                }
                _ => { }
            }
        }
    }

    fn read_usize(file: &mut File) -> usize {
        let mut buf = [0u8; size_of::<usize>()];
        file.read_exact(&mut buf).expect("Couldn't read usize from file");

        usize::from_le_bytes(buf)
    }

    fn read_section(file: &mut File, size: usize, chunk_size: usize) -> Vec<f32> {
        let mut buf = vec![0u8; size * chunk_size];
        file.read_exact(&mut buf).expect("Couldn't read section of file");
        let floats: Vec<f32> = buf
            .chunks_exact(chunk_size)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        floats
    }

    pub fn attempt_load_network(from_file_path: &str, cycle: usize, to_nodes: &mut Vec<NeuralNetworkNode>) -> usize {
        let file_open_try = File::open(format!("{from_file_path}{cycle}.nn"));
 
        // 4 for 32-bit, 8 for 64-bit
        const CHUNK_SIZE: usize = 4;

        let mut epoch: usize = 0;

        match file_open_try {
            Ok(mut file) => {   // File could be opened for read
                epoch = NeuralNetwork::read_usize(&mut file);
                for node in to_nodes {
                    match node {
                        NeuralNetworkNode::HiddenLayer(n) => {
                            // Load weights first
                            let (mut rows, mut columns) = n.weights.shape();
                            let weight_floats = NeuralNetwork::read_section(&mut file, columns * rows, CHUNK_SIZE);
                            n.weights = Matrix::from(rows, columns, weight_floats);

                            // Load biases next
                            (rows, columns) = n.biases.shape();                            
                            let biases_floats = NeuralNetwork::read_section(&mut file, columns * rows, CHUNK_SIZE);
                            n.biases = Matrix::from(rows, columns, biases_floats);
                            println!("Loaded weights and biases for dense layer.")
                        }
                        _ => { }
                    }
                }

                println!("Success in loading the neural network!");
            }
            _ => {
                println!("Could not find or open file specified, starting new model.");
            }
        }

        epoch
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nn_test() {
    }
}
