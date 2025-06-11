//! Core traits for SVM implementation

use crate::core::{Prediction, Sample};

/// Dataset abstraction for efficient data access
pub trait Dataset: Send + Sync {
    /// Number of samples in the dataset
    fn len(&self) -> usize;

    /// Number of features (dimensionality)
    fn dim(&self) -> usize;

    /// Get a single sample by index
    ///
    /// # Panics
    /// Panics if index >= len()
    fn get_sample(&self, i: usize) -> Sample;

    /// Get multiple samples efficiently (for parallel processing)
    fn get_batch(&self, indices: &[usize]) -> Vec<Sample> {
        indices.iter().map(|&i| self.get_sample(i)).collect()
    }

    /// Get all labels as a vector (for initialization)
    fn get_labels(&self) -> Vec<f64>;

    /// Check if the dataset is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Trained SVM model
pub trait SVMModel: Send + Sync {
    /// Predict a single sample
    fn predict(&self, sample: &Sample) -> Prediction;

    /// Predict multiple samples in parallel
    fn predict_batch(&self, samples: &[Sample]) -> Vec<Prediction> {
        samples.iter().map(|s| self.predict(s)).collect()
    }

    /// Get the number of support vectors
    fn n_support_vectors(&self) -> usize;

    /// Get the bias term
    fn bias(&self) -> f64;
}
