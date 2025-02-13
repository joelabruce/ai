// For this example to work, you have to extract both mnist_test.csv nd mnist_train.csv from the archive into the training folder.
// Use the following command to run in release mode:
// cargo run --release --example mnist_digits

use ai::{digit_image::DigitImage, geoalg::f32_math::matrix::Matrix, nn::{activation_functions::RELU, layers::{convolution2d::{Convolution2d, Dimensions}, input::Input}, neural::NeuralNetworkNode, trainer::train_network}, statistics::sample::Sample, timed};

/// Creates an input layer drawn randomly from a sample.
pub fn from_sample_digit_images(sample: &mut Sample<DigitImage>, requested_batch_size: usize) -> (Input, Matrix) {
    let data_from_sample = sample.random_batch(requested_batch_size);

    let mut pixel_vector = Vec::with_capacity(data_from_sample.len() * 785);
    let mut target_vector = Vec::with_capacity(data_from_sample.len() * 10);
    let rows = data_from_sample.len();
    for datum in data_from_sample {
        pixel_vector.extend(datum.pixels.clone());
        target_vector.extend(datum.one_hot_encoded_label());
    }

    (
        Input::from(rows, 784, pixel_vector), 
        Matrix::from(rows, 10, target_vector)
    )
}

pub fn handwritten_digits(load_from_file: bool, include_batch_output: bool) {
    let time_to_run = timed::timed(|| {
        // Create layers
        let convo = Convolution2d::new(
            8, 
            1, 
            Dimensions { width: 3, height: 3 } ,
            Dimensions { width: 28, height: 28 });

        let maxpool = convo.influences_maxpool(
            Dimensions { width: 2, height: 2 },
            2);

        let dense1 = maxpool.influences_dense(100);
        let dense2 = dense1.influences_dense(10);

        // Add layers to the network for forward and backward propagation.
        let mut nn_nodes: Vec<NeuralNetworkNode> = Vec::new();
        nn_nodes.push(NeuralNetworkNode::Convolution2dLayer(convo));
        nn_nodes.push(NeuralNetworkNode::ActivationLayer(RELU));
        nn_nodes.push(NeuralNetworkNode::MaxPoolLayer(maxpool));
        nn_nodes.push(NeuralNetworkNode::DenseLayer(dense1));
        nn_nodes.push(NeuralNetworkNode::ActivationLayer(RELU));
        nn_nodes.push(NeuralNetworkNode::DenseLayer(dense2));

        train_network(&mut nn_nodes, load_from_file, include_batch_output)
    });

    println!("Total time to run: {time_to_run}");
}

// Runs a neural network for handwritten digit recogniton.
fn main() {
    handwritten_digits(true, false);
}