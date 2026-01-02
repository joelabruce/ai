use std::io::Write;

use crate::{digit_image::DigitImage, geoalg::f32_math::matrix::Matrix, input_csv_reader::InputCsvReader, nn::{activation_functions::{accuracy, backward_categorical_cross_entropy_loss_wrt_softmax, forward_categorical_cross_entropy_loss, SOFTMAX}, error::{NeuralNetworkError, Result}, learning_rate::LearningRate, neural::NeuralNetwork}, output_bin_writer::OutputBinWriter, statistics::sample::Sample};

use super::layers::input::Input;

/// Hyperparameters for training a neural network.
///
/// Controls all aspects of the training process including batch sizes, epochs,
/// checkpointing, and logging.
///
/// # Examples
///
/// ```
/// use ai::nn::trainer::TrainingHyperParameters;
///
/// let hyperparams = TrainingHyperParameters {
///     backup_cycle: 10,
///     total_epochs: 50,
///     training_sample: 60000,
///     batch_size: 128,
///     trained_model_location: String::from("./trained/model"),
///     batch_inform_size: 100,
///     output_accuracy: true,
///     output_loss: true,
///     save_per_epoch: true,
/// };
/// ```
pub struct TrainingHyperParameters {
    /// Number of epochs between full model backups (rolling file system)
    pub backup_cycle: usize,
    /// Total number of training epochs to run
    pub total_epochs: usize,
    /// Total number of training samples in the dataset
    pub training_sample: usize,
    /// Number of samples per training batch
    pub batch_size: usize,
    /// Directory path where trained models are saved
    pub trained_model_location: String,
    /// Print progress every N batches (0 to disable)
    pub batch_inform_size: usize,
    /// Whether to print accuracy metrics during training
    pub output_accuracy: bool,
    /// Whether to print loss values during training
    pub output_loss: bool,
    /// Whether to save model weights after each epoch
    pub save_per_epoch: bool
}

impl TrainingHyperParameters {
    /// Validates the hyperparameters and returns an error if any are invalid.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `batch_size` is 0
    /// - `training_sample` is less than `batch_size`
    /// - `total_epochs` is 0
    /// - `backup_cycle` is 0
    pub fn validate(&self) -> Result<()> {
        if self.batch_size == 0 {
            return Err(NeuralNetworkError::InvalidHyperparameters {
                parameter: "batch_size".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }

        if self.training_sample < self.batch_size {
            return Err(NeuralNetworkError::InvalidHyperparameters {
                parameter: "training_sample".to_string(),
                reason: format!(
                    "must be at least batch_size ({}), got {}",
                    self.batch_size, self.training_sample
                ),
            });
        }

        if self.total_epochs == 0 {
            return Err(NeuralNetworkError::InvalidHyperparameters {
                parameter: "total_epochs".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }

        if self.backup_cycle == 0 {
            return Err(NeuralNetworkError::InvalidHyperparameters {
                parameter: "backup_cycle".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }

        Ok(())
    }
}

/// Creates an InputCsvReader
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
        Matrix::new(rows, 10, target_vector)
    )
}

/// Trains a neural network using mini-batch gradient descent with categorical cross-entropy loss.
///
/// This function handles the complete training loop including:
/// - Loading checkpointed models (if requested)
/// - Batch sampling with automatic shuffling
/// - Forward and backward propagation
/// - Model checkpointing with rolling file cycles
/// - Validation metrics (accuracy and loss)
/// - Progress logging
///
/// # Arguments
///
/// * `nn` - The neural network to train (mutable)
/// * `tp` - Training hyperparameters configuration
/// * `load_from_file` - Whether to attempt loading a previously trained model
/// * `include_batch_output` - Whether to print per-batch progress
///
/// # Training Process
///
/// 1. **Initialization**: Load existing model or start fresh
/// 2. **Each Epoch**:
///    - Shuffle training data via random batch sampling
///    - Forward pass: compute predictions
///    - Compute loss: categorical cross-entropy with softmax
///    - Backward pass: compute gradients and update weights
///    - Validation: evaluate on test set
///    - Save checkpoint: rolling file system with configurable backup cycle
///
/// # File Organization
///
/// Models are saved with rolling file cycles to prevent overfitting:
/// - Files: `model.1.bin`, `model.2.bin`, ..., `model.{backup_cycle}.bin`
/// - Automatically rotates through files, allowing easy rollback
///
/// # Examples
///
/// See `examples/mnist_handwritten_digits.rs` for complete usage.
///
/// # Performance
///
/// Training speed depends on:
/// - Network architecture (layers, neurons, filters)
/// - Batch size (larger = more throughput, less frequent updates)
/// - Matrix operations use SIMD and multi-threading automatically
pub fn train_network(nn: &mut NeuralNetwork, tp: TrainingHyperParameters, load_from_file: bool, include_batch_output: bool) {
        // Training hyper-parameters
    let batches = tp.training_sample / tp.batch_size;
    let v_batch_size = std::cmp::min(batches * tp.batch_size / 5, 9999);        
    let trained_model_location = &tp.trained_model_location;
    let learning_rate = &mut LearningRate::new(0.01);

    let mut epoch_offset = 0;
    if load_from_file {
        print!("Try to load trained neural network ... ");
        match nn.attempt_load_network(&tp.trained_model_location, 1) {
            Ok(epochs) => {
                epoch_offset = epochs;
                println!("Successful in loading trained neural network!")
            },
            Err(msg) => println!("{msg}")
        }
    }

    // Validtion setup
    let mut testing_reader = open_for_importing("./training/mnist_test.csv");
    let _ = testing_reader.read_and_skip_header_line();
    let mut testing_sample = create_sample_for_digit_images_from_file(&mut testing_reader, 10000);
    let (vl, v_targets) = from_sample_digit_images(&mut testing_sample, v_batch_size);

    // Training setup
    let mut training_reader = open_for_importing("./training/mnist_train.csv");
    let _ = training_reader.read_and_skip_header_line();
    let mut training_sample = create_sample_for_digit_images_from_file(&mut training_reader, tp.training_sample);

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
            forward_stack = nn.forward(il);// NeuralNetwork::forward(il, nn_nodes);

            // Forward pass on training data btch
            let predictions = (SOFTMAX.f)(&forward_stack.pop().unwrap());
            
            // Backward pass on training data batch
            let dvalues6 = backward_categorical_cross_entropy_loss_wrt_softmax(&predictions, &targets).scale(1. / tp.batch_size as f32);
            nn.backward(learning_rate, &dvalues6, &mut forward_stack);

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

        if tp.save_per_epoch {
            let backup_to_write = 1 + (epoch - 1) % tp.backup_cycle;
            print!("Complete. Saving cycle # {backup_to_write} ... ");
            std::io::stdout().flush().unwrap();

            let mut network_saver = OutputBinWriter::new(format!("{trained_model_location}{backup_to_write}.nn").as_str());
            nn.save_network(epoch, &mut network_saver);
        }

        print!("Complete ");
        std::io::stdout().flush().unwrap();

        // Validate updated neural network against validation inputs it hasn't been trained on.
        // Clone the validation layer, so it is not consumed
        // Better to clone here than cloning for each iteration of the batches being trained on for performance.
        forward_stack = nn.forward(vl.clone());
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

    if !tp.save_per_epoch {
        let epoch = epoch_offset + tp.total_epochs;
        let backup_to_write = 1 + (epoch - 1) % tp.backup_cycle;
        print!("Complete. Saving cycle # {backup_to_write} ... ");
        std::io::stdout().flush().unwrap();

        let mut network_saver = OutputBinWriter::new(format!("{trained_model_location}{backup_to_write}.nn").as_str());
        nn.save_network(epoch, &mut network_saver);
    }
}