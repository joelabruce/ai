//! Error types for neural network operations.

use std::fmt;
use std::io;

/// Errors that can occur during neural network operations.
///
/// This enum provides context for failures during training, inference,
/// model loading/saving, and layer operations.
#[derive(Debug)]
pub enum NeuralNetworkError {
    /// Error reading or writing model files
    IoError(io::Error),

    /// Model file not found at specified path
    ModelNotFound {
        path: String,
        cycle: usize,
    },

    /// Model file is corrupted or has invalid format
    CorruptedModel {
        path: String,
        reason: String,
    },

    /// Dimension mismatch between layers
    DimensionMismatch {
        layer: String,
        expected: (usize, usize),
        got: (usize, usize),
    },

    /// Invalid layer configuration
    InvalidLayerConfig {
        layer: String,
        reason: String,
    },

    /// Invalid training parameters
    InvalidHyperparameters {
        parameter: String,
        reason: String,
    },

    /// Generic error with context
    Other(String),
}

impl fmt::Display for NeuralNetworkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NeuralNetworkError::IoError(e) => write!(f, "I/O error: {}", e),
            NeuralNetworkError::ModelNotFound { path, cycle } => {
                write!(f, "Model file not found: {}{}.nn", path, cycle)
            }
            NeuralNetworkError::CorruptedModel { path, reason } => {
                write!(f, "Corrupted model file '{}': {}", path, reason)
            }
            NeuralNetworkError::DimensionMismatch { layer, expected, got } => {
                write!(
                    f,
                    "Dimension mismatch in layer '{}': expected {:?}, got {:?}",
                    layer, expected, got
                )
            }
            NeuralNetworkError::InvalidLayerConfig { layer, reason } => {
                write!(f, "Invalid configuration for layer '{}': {}", layer, reason)
            }
            NeuralNetworkError::InvalidHyperparameters { parameter, reason } => {
                write!(f, "Invalid hyperparameter '{}': {}", parameter, reason)
            }
            NeuralNetworkError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for NeuralNetworkError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            NeuralNetworkError::IoError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for NeuralNetworkError {
    fn from(error: io::Error) -> Self {
        NeuralNetworkError::IoError(error)
    }
}

/// Convenience type alias for Results in neural network operations.
pub type Result<T> = std::result::Result<T, NeuralNetworkError>;
