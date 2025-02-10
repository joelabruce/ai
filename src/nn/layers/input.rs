use super::Matrix;

/// Allows for creation of succeeding layers based on initial data passed in.
#[derive(Clone)]
pub struct Input {
    pub input_matrix: Matrix
}

impl Input {
    // Creates input layer based on inputs supplied.
    // Allows for automatic shaping of succeeding layer generation.
    pub fn from(batch_size: usize, features: usize, raw_values: Vec<f32>) -> Self {
        Self {
            input_matrix: Matrix::from(batch_size, features, raw_values)
        }
    }
}