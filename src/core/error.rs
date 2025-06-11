//! Error types for SVM implementation

use thiserror::Error;

#[derive(Error, Debug)]
pub enum SVMError {
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Optimization failed: {0}")]
    OptimizationError(String),

    #[error("Model not trained")]
    ModelNotTrained,

    #[error("Invalid dataset: {0}")]
    InvalidDataset(String),

    #[error("Invalid label: expected -1 or +1, got {0}")]
    InvalidLabel(f64),

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Empty dataset")]
    EmptyDataset,

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Parse error: {0}")]
    ParseError(String),
}

pub type Result<T> = std::result::Result<T, SVMError>;
