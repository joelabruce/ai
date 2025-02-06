use ai::{digit_image::DigitImage, geoalg::f32_math::matrix::Matrix, nn::{activation_functions::{accuracy, backward_categorical_cross_entropy_loss_wrt_softmax, forward_categorical_cross_entropy_loss, RELU, SOFTMAX}, layers::{dense_layer::DenseLayer, input_layer::InputLayer}, neural::{NeuralNetwork, Node}}, output_bin_writer::OutputBinWriter, statistics::sample::Sample};

/// Creates an input layer drawn randomly from a sample.
pub fn from_sample_digit_images(sample: &mut Sample<DigitImage>, requested_batch_size: usize) -> (InputLayer, Matrix) {
    let data_from_sample = sample.random_batch(requested_batch_size);

    let mut pixel_vector = Vec::with_capacity(data_from_sample.len() * 785);
    let mut target_vector = Vec::with_capacity(data_from_sample.len() * 10);
    let rows = data_from_sample.len();
    for datum in data_from_sample {
        pixel_vector.extend(datum.pixels.clone());
        target_vector.extend(datum.one_hot_encoded_label());
    }

    (InputLayer {
        input_matrix: Matrix::from(rows, 784, pixel_vector)
    }, Matrix::from(rows, 10, target_vector))
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
            let dvalues6 = backward_categorical_cross_entropy_loss_wrt_softmax(&predictions, &targets).scale_simd(1. / batch_size as f32);
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

// Runs a neural network for handwritten digit recogniton.
fn main() {
    handwritten_digits(true);
}