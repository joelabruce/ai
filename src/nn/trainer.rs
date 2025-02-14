use std::io::Write;

use crate::{digit_image::DigitImage, geoalg::f32_math::matrix::Matrix, nn::{activation_functions::{accuracy, backward_categorical_cross_entropy_loss_wrt_softmax, forward_categorical_cross_entropy_loss, SOFTMAX}, neural::NeuralNetwork}, output_bin_writer::OutputBinWriter, statistics::sample::Sample};

use super::{layers::input::Input, neural::NeuralNetworkNode};

pub struct TrainingHyperParameters {
    pub backup_cycle: usize,
    pub total_epochs: usize,
    pub training_sample: usize,
    pub batch_size: usize,
    pub trained_model_location: String,
    pub batch_inform_size: usize,
    pub output_accuracy: bool,
    pub output_loss: bool,
    pub save: bool
}

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

/// Try to put all println output in here instead of in the other functions.
/// Unstable.
pub fn train_network(nn_nodes: &mut Vec<NeuralNetworkNode>, tp: TrainingHyperParameters, load_from_file: bool, include_batch_output: bool) {
    // Training hyper-parameters
    let batches = tp.training_sample / tp.batch_size;
    let v_batch_size = std::cmp::min(batches * tp.batch_size / 5, 9999);        
    let trained_model_location = &tp.trained_model_location;

    let mut epoch_offset = 0;
    if load_from_file {
        print!("Try to load trained neural network ... ");
        match NeuralNetwork::attempt_load_network(&tp.trained_model_location, 1, nn_nodes) {
            Ok(epochs) => {
                epoch_offset = epochs;
                println!("Successful in loading trained neural network!")
            },
            Err(msg) => println!("{msg}")
        }
    }

    // Validtion setup
    let mut testing_reader = NeuralNetwork::open_for_importing("./training/mnist_test.csv");
    let _ = testing_reader.read_and_skip_header_line();
    let mut testing_sample = NeuralNetwork::create_sample_for_digit_images_from_file(&mut testing_reader, 10000);
    let (vl, v_targets) = from_sample_digit_images(&mut testing_sample, v_batch_size);

    // Training setup
    let mut training_reader = NeuralNetwork::open_for_importing("./training/mnist_train.csv");
    let _ = training_reader.read_and_skip_header_line();
    let mut training_sample = NeuralNetwork::create_sample_for_digit_images_from_file(&mut training_reader, tp.training_sample);

    // Create Layers in network
    let mut lowest_loss = f32::INFINITY;
    let mut forward_stack: Vec<Matrix>;

    println!("-Beginning training-");
    for epoch in epoch_offset + 1..=epoch_offset + tp.total_epochs {
        print!("Epoch # {epoch} ... ");
        std::io::stdout().flush().unwrap();

        if include_batch_output { println!(); }

        training_sample.reset();
        for _batch in 0..batches {
            let (il, targets) = from_sample_digit_images(&mut training_sample, tp.batch_size);
            forward_stack = NeuralNetwork::forward(il, nn_nodes);

            // Forward pass on training data btch
            let predictions = (SOFTMAX.f)(&forward_stack.pop().unwrap());
            
            // Backward pass on training data batch
            let dvalues6 = backward_categorical_cross_entropy_loss_wrt_softmax(&predictions, &targets).scale_simd(1. / tp.batch_size as f32);
            NeuralNetwork::backward( nn_nodes, &dvalues6, &mut forward_stack);

            // Only uncomment if network training is slow to see if accuracy and data loss is actually improving
            if include_batch_output && _batch > 0 && _batch % tp.batch_inform_size == 0 {
                print!("  Training through batch #{_batch:4} complete ");
                std::io::stdout().flush().unwrap();

                // Only needed when outputting data loss for debugging purposes.
                if tp.output_accuracy {
                    let accuracy = 100. * accuracy(&predictions, &targets);
                    print!("| Accuracy: {accuracy:7.3}% ");
                }

                if tp.output_loss {
                    let sample_losses = forward_categorical_cross_entropy_loss(&predictions, &targets);
                    let data_loss = sample_losses.read_values().into_iter().sum::<f32>() / sample_losses.len() as f32;            
                    println!("| Loss: {data_loss:.5}");
                }
            }
        }

        if tp.save {
            let backup_to_write = 1 + (epoch - 1) % tp.backup_cycle;
            print!("Complete. Saving cycle # {backup_to_write} ... ");
            std::io::stdout().flush().unwrap();

            let mut network_saver = OutputBinWriter::new(format!("{trained_model_location}{backup_to_write}.nn").as_str());
            NeuralNetwork::save_network(epoch, &nn_nodes, &mut network_saver);
        }

        print!("Complete ");
        std::io::stdout().flush().unwrap();

        // Validate updated neural network against validation inputs it hasn't been trained on.
        // Clone the validation layer, so it is not consumed
        // Better to clone here than cloning for each iteration of the batches being trained on for performance.
        forward_stack = NeuralNetwork::forward(vl.clone(),  nn_nodes);
        let v_predictions = &(SOFTMAX.f)(&forward_stack.pop().unwrap());
        
        if tp.output_accuracy {
            let accuracy = 100. * accuracy(&v_predictions, &v_targets);
            print!("| Accuracy: {accuracy:7.3}% ");
        }

        let v_sample_losses = forward_categorical_cross_entropy_loss(&v_predictions, &v_targets);
        let v_data_loss = v_sample_losses.read_values().into_iter().sum::<f32>() / v_sample_losses.len() as f32;
    
        if tp.output_loss {
            print!("| Loss: {v_data_loss:.5}");
        }

        if v_data_loss < lowest_loss { lowest_loss = v_data_loss; } else { print!(" *Warning, validation has not improved! Consider stopping training here."); }
        println!();
    }
}