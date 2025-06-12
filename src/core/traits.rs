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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Prediction, Sample, SparseVector};

    // Mock SVM model for testing trait default implementations
    struct MockSVMModel {
        mock_bias: f64,
        mock_n_sv: usize,
    }

    impl SVMModel for MockSVMModel {
        fn predict(&self, sample: &Sample) -> Prediction {
            // Simple mock: predict based on first feature value
            let decision_value = if let Some(&val) = sample.features.values.first() {
                val * sample.label
            } else {
                0.0
            };
            Prediction::new(sample.label, decision_value)
        }

        fn n_support_vectors(&self) -> usize {
            self.mock_n_sv
        }

        fn bias(&self) -> f64 {
            self.mock_bias
        }
    }

    #[test]
    fn test_predict_batch_default_implementation() {
        let model = MockSVMModel {
            mock_bias: 0.5,
            mock_n_sv: 3,
        };

        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![1.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![2.0]), -1.0),
            Sample::new(SparseVector::new(vec![0], vec![3.0]), 1.0),
        ];

        let predictions = model.predict_batch(&samples);
        
        assert_eq!(predictions.len(), 3);
        assert_eq!(predictions[0].label, 1.0);
        assert_eq!(predictions[0].decision_value, 1.0); // 1.0 * 1.0
        assert_eq!(predictions[1].label, -1.0);
        assert_eq!(predictions[1].decision_value, -2.0); // 2.0 * -1.0
        assert_eq!(predictions[2].label, 1.0);
        assert_eq!(predictions[2].decision_value, 3.0); // 3.0 * 1.0
    }

    #[test]
    fn test_model_properties() {
        let model = MockSVMModel {
            mock_bias: 0.25,
            mock_n_sv: 5,
        };

        assert_eq!(model.n_support_vectors(), 5);
        assert_eq!(model.bias(), 0.25);
    }
}
