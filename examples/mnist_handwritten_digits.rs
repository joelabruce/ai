// For this example to work, you have to extract both mnist_test.csv nd mnist_train.csv from the archive into the training folder.
// Use the following command to run in release mode:
// cargo run --release --example mnist_digits

use ai::{nn::{activations::activation::RELU, layers::dense::Dense, neural::NeuralNetworkNode, trainer::{train_network, TrainingHyperParameters}}, timed};

pub fn handwritten_digits(load_from_file: bool, include_batch_output: bool) {
    let time_to_run = timed::timed(|| {
        // Create hyper-parameters fine-tuned for this example.
        let tp = TrainingHyperParameters {
            backup_cycle: 4,
            total_epochs: 10,
            training_sample: 60000,
            batch_size: 2000,
            trained_model_location: "dense_model".to_string(),
            batch_inform_size: 10,
            output_accuracy: true,
            output_loss: false,
            save: true
        };
        
        // Create layers
        let dense1 = Dense::new(784, 128);
        let dense2 = dense1.influences_dense(64);
        let dense3 = dense2.influences_dense(10);

        // Add layers to the network for forward and backward propagation.
        let mut nn_nodes: Vec<NeuralNetworkNode> = Vec::new();
        nn_nodes.push(NeuralNetworkNode::DenseLayer(dense1));
        nn_nodes.push(NeuralNetworkNode::ActivationFunction(RELU));
        nn_nodes.push(NeuralNetworkNode::DenseLayer(dense2));
        nn_nodes.push(NeuralNetworkNode::ActivationFunction(RELU));
        nn_nodes.push(NeuralNetworkNode::DenseLayer(dense3));

        train_network(&mut nn_nodes, tp, load_from_file, include_batch_output);
    });

    println!("Total time to run: {time_to_run}");
}

// Runs a neural network for handwritten digit recogniton.
fn main() {
    // Since this example runs so fast, we turn off include_batch_output
    handwritten_digits(true, false);
}