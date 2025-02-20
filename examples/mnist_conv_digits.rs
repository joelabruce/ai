// For this example to work, you have to extract both mnist_test.csv nd mnist_train.csv from the archive into the training folder.
// Use the following command to run in release mode:
// cargo run --release --example mnist_digits

use ai::{nn::{activations::activation::RELU, layers::{convolution2d::{Convolution2d, Dimensions}, input::InputTypes}, neural::NeuralNetworkNode, trainer::{train_network, TrainingHyperParameters}}, timed};


pub fn handwritten_digits(load_from_file: bool, include_batch_output: bool) {
    let time_to_run = timed::timed(|| {
        // Create hyper-parameters fine-tuned for this example.
        let tp = TrainingHyperParameters {
            backup_cycle: 1,
            total_epochs: 2,
            training_sample_size: 60000,
            validation_sample_size: 10000,
            batch_size: 200,
            trained_model_location: "convo_model".to_string(),
            batch_inform_size: 10,
            output_accuracy: true,
            output_loss: true,
            save: true
        };

        // Create layers for network to train on
        let convo1 = Convolution2d::new(
            16, 
            1, 
            Dimensions { width: 3, height: 3 } ,
            Dimensions { width: 28, height: 28 });

        let maxpool1 = convo1.feed_into_maxpool(
            Dimensions { width: 2, height: 2 },
            2);

        let reshaper = maxpool1.feed_into_reshaper();

        let dense1 = reshaper.feed_into_dense(64);
        let dense2 = dense1.feed_into_dense(10);

        // Add layers to the network for forward and backward propagation.
        let mut nn_nodes: Vec<NeuralNetworkNode> = Vec::new();
        nn_nodes.push(NeuralNetworkNode::Convolution2dLayer(convo1));
        nn_nodes.push(NeuralNetworkNode::Activation(RELU));
        nn_nodes.push(NeuralNetworkNode::MaxPoolLayer(maxpool1));
        nn_nodes.push(NeuralNetworkNode::Reshaper(reshaper));
        nn_nodes.push(NeuralNetworkNode::DenseLayer(dense1));
        nn_nodes.push(NeuralNetworkNode::Activation(RELU));
        nn_nodes.push(NeuralNetworkNode::DenseLayer(dense2));

        let input_type = &InputTypes::Image { height: 28, width: 28, channels: 1 };
        train_network(&mut nn_nodes, tp, load_from_file, include_batch_output, input_type);
    });

    println!("Total time to run: {time_to_run}");
}

// Runs a neural network for handwritten digit recogniton.
fn main() {
    // Since this example runs slower, we turn include_batch_output to true
    handwritten_digits(true, true);
}