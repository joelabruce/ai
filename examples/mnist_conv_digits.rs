// For this example to work, you have to extract both mnist_test.csv nd mnist_train.csv from the archive into the training folder.
// Use the following command to run in release mode:
// cargo run --release --example mnist_digits

use std::io::Write;

use ai::{digit_image::DigitImage, geoalg::f32_math::matrix::Matrix, nn::{activation_functions::{accuracy, backward_categorical_cross_entropy_loss_wrt_softmax, forward_categorical_cross_entropy_loss, RELU, SOFTMAX}, layers::{convolution2d::{Convolution2d, Dimensions}, input::Input}, neural::{NeuralNetwork, NeuralNetworkNode}}, output_bin_writer::OutputBinWriter, statistics::sample::Sample, timed};

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
        // Training hyper-parameters
        let backup_cycle = 1;
        let total_epochs = 10;
        let training_sample = 60000;
        let batch_size = 125;
        let batches = training_sample / batch_size;
        let v_batch_size = std::cmp::min(batches * batch_size / 5, 9999);        
        let mut lowest_loss = f32::INFINITY;

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

        let trained_model_location = "./tests/convo_model";

        // Begin processing neural network now, this code should only be in 1 place between examples.
        // Consider refactoring to make examples easier instead of redundant everywhere.
        let mut epoch_offset = 0;
        if load_from_file {
            println!("Trying to load trained neural network...");
            epoch_offset = NeuralNetwork::attempt_load_network(&trained_model_location, 1,&mut nn_nodes);
        }

        // Validtion setup
        let mut testing_reader = NeuralNetwork::open_for_importing("./training/mnist_test.csv");
        let _ = testing_reader.read_and_skip_header_line();
        let mut testing_sample = NeuralNetwork::create_sample_for_digit_images_from_file(&mut testing_reader, 10000);
        let (vl, v_targets) = from_sample_digit_images(&mut testing_sample, v_batch_size);

        // Training setup
        let mut training_reader = NeuralNetwork::open_for_importing("./training/mnist_train.csv");
        let _ = training_reader.read_and_skip_header_line();
        let mut training_sample = NeuralNetwork::create_sample_for_digit_images_from_file(&mut training_reader, training_sample);

        // Create Layers in network
        let mut forward_stack: Vec<Matrix>;

        println!("-Beginning training-");
        for epoch in epoch_offset + 1..=epoch_offset + total_epochs {
            print!("Epoch # {epoch} ... ");
            std::io::stdout().flush().unwrap();

            if include_batch_output { println!(); }

            training_sample.reset();
            for _batch in 0..batches {
                let (il, targets) = from_sample_digit_images(&mut training_sample, batch_size);
                forward_stack = NeuralNetwork::forward(il, &mut nn_nodes);
    
                // Forward pass on training data btch
                let predictions = (SOFTMAX.f)(&forward_stack.pop().unwrap());
                
                // Backward pass on training data batch
                let dvalues6 = backward_categorical_cross_entropy_loss_wrt_softmax(&predictions, &targets).scale_simd(1. / batch_size as f32);
                NeuralNetwork::backward(&mut nn_nodes, &dvalues6, &mut forward_stack);

                // Only uncomment if network training is slow to see if accuracy and data loss is actually improving
                if include_batch_output && _batch > 0 && _batch % 100 == 0 {
                    // Only needed when outputting data loss for debugging purposes.
                    let accuracy = 100. * accuracy(&predictions, &targets);
                    let sample_losses = forward_categorical_cross_entropy_loss(&predictions, &targets);
                    let data_loss = sample_losses.read_values().into_iter().sum::<f32>() / sample_losses.len() as f32;            
                    println!("   Training through batch #{_batch} complete | Accuracy: {accuracy:7.3}% | Loss: {data_loss}");
                }
            }

            let backup_to_write = 1 + (epoch - 1) % backup_cycle;
            print!("Complete. Saving cycle # {backup_to_write} ... ");
            let mut network_saver = OutputBinWriter::new(format!("{trained_model_location}{backup_to_write}.nn").as_str());
            NeuralNetwork::save_network(epoch, &nn_nodes, &mut network_saver);
            print!("Complete");

            // Validate updated neural network against validation inputs it hasn't been trained on.
            // Clone the validation layer, so it is not consumed
            // Better to clone here than cloning for each iteration of the batches being trained on for performance.
            forward_stack = NeuralNetwork::forward(vl.clone(), &mut nn_nodes);
            let v_predictions = &(SOFTMAX.f)(&forward_stack.pop().unwrap());
            
            let accuracy = 100. * accuracy(&v_predictions, &v_targets);
            let v_sample_losses = forward_categorical_cross_entropy_loss(&v_predictions, &v_targets);
            let v_data_loss = v_sample_losses.read_values().into_iter().sum::<f32>() / v_sample_losses.len() as f32;
            print!(" | Accuracy: {accuracy:7.3}% | Loss: {v_data_loss}");

            if v_data_loss < lowest_loss { lowest_loss = v_data_loss; println!(); } else { println!(" *Warning, validation has not improved! Consider stopping training here."); }
        }
    });

    println!("Total time to run: {time_to_run}");
}

// Runs a neural network for handwritten digit recogniton.
fn main() {
    handwritten_digits(true, false);
}