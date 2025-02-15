// For this example to work, you have to extract both mnist_test.csv nd mnist_train.csv from the archive into the training folder.
// Use the following command to run in release mode:
// cargo run --release --example mnist_digits

use ai::{nn::{activation_functions::RELU, layers::convolution2d::{Convolution2dDeprecated, Dimensions}, neural::NeuralNetworkNode, trainer::{train_network, TrainingHyperParameters}}, timed};

pub fn handwritten_digits(load_from_file: bool, include_batch_output: bool) {
    let time_to_run = timed::timed(|| {
        // Create hyper-parameters fine-tuned for this example.
        let tp = TrainingHyperParameters {
            backup_cycle: 1,
            total_epochs: 5,
            training_sample: 60000,
            batch_size: 125,
            trained_model_location: "convo_model".to_string(),
            batch_inform_size: 50,
            output_accuracy: true,
            output_loss: true,
            save: true
        };

        // Create layers for network to train on
        let convo = Convolution2dDeprecated::new(
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

        train_network(&mut nn_nodes, tp, load_from_file, include_batch_output)
    });

    println!("Total time to run: {time_to_run}");
}

// Runs a neural network for handwritten digit recogniton.
fn main() {
    // Since this example runs slower, we turn include_batch_output to true
    handwritten_digits(true, true);
}