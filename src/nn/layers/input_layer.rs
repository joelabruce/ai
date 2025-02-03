use super::Matrix;

/// Allows for creation of succeeding layers based on initial data passed in.
#[derive(Clone)]
pub struct InputLayer {
    pub input_matrix: Matrix
}

impl InputLayer {
    /// Creates input layer based on inputs supplied.
    /// Allows for automatic shaping of succeeding layer generation.
    pub fn from_vec(values: Vec<Vec<f32>>) -> InputLayer {
        let rows = values.len();                                                // Number of inputs
        let columns = values.len() / rows;                                      // Number of features.
        let values: Vec<f32> = values.into_iter().flat_map(|v| v).collect();

        InputLayer {
            input_matrix: Matrix::from(rows, columns, values)
        }
    }
}