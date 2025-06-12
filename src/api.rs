//! High-level API for Support Vector Machine operations
//!
//! This module provides a user-friendly interface for common SVM tasks,
//! including training, prediction, and model evaluation.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use rsvm::api::SVM;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Train a model on data
//! let svm = SVM::new()
//!     .with_c(1.0)
//!     .with_epsilon(0.001)
//!     .train_from_file("data.libsvm")?;
//!
//! // Make predictions
//! let predictions = svm.predict_from_file("test.libsvm")?;
//! println!("Accuracy: {:.2}%", svm.evaluate_from_file("test.libsvm")? * 100.0);
//! # Ok(())
//! # }
//! ```

use crate::core::{Dataset, OptimizerConfig, Prediction, Result, SVMError, SVMModel, Sample};
use crate::data::{CSVDataset, LibSVMDataset};
use crate::kernel::{Kernel, LinearKernel};
use crate::optimizer::{SVMOptimizer, TrainedSVM};
use std::path::Path;

/// High-level SVM interface with builder pattern
pub struct SVM<K: Kernel = LinearKernel> {
    kernel: K,
    config: OptimizerConfig,
}

impl SVM<LinearKernel> {
    /// Create a new SVM with linear kernel and default parameters
    pub fn new() -> Self {
        Self {
            kernel: LinearKernel::new(),
            config: OptimizerConfig::default(),
        }
    }
}

impl Default for SVM<LinearKernel> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Kernel> SVM<K> {
    /// Create SVM with custom kernel
    pub fn with_kernel(kernel: K) -> Self {
        Self {
            kernel,
            config: OptimizerConfig::default(),
        }
    }

    /// Set regularization parameter C
    pub fn with_c(mut self, c: f64) -> Self {
        self.config.c = c;
        self
    }

    /// Set convergence tolerance
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.config.epsilon = epsilon;
        self
    }

    /// Set maximum number of iterations
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.config.max_iterations = max_iterations;
        self
    }

    /// Set kernel cache size in bytes
    pub fn with_cache_size(mut self, cache_size: usize) -> Self {
        self.config.cache_size = cache_size;
        self
    }

    /// Train on a dataset
    pub fn train<D: Dataset>(self, dataset: &D) -> Result<TrainedModel<K>> {
        let optimizer = SVMOptimizer::new(self.kernel, self.config);
        let model = optimizer.train(dataset)?;
        Ok(TrainedModel { model })
    }

    /// Train on samples
    pub fn train_samples(self, samples: &[Sample]) -> Result<TrainedModel<K>> {
        let optimizer = SVMOptimizer::new(self.kernel, self.config);
        let model = optimizer.train_samples(samples)?;
        Ok(TrainedModel { model })
    }

    /// Train from LibSVM format file
    pub fn train_from_file<P: AsRef<Path>>(self, path: P) -> Result<TrainedModel<K>> {
        let dataset = LibSVMDataset::from_file(path)?;
        self.train(&dataset)
    }

    /// Train from CSV file (automatically detects headers)
    pub fn train_from_csv<P: AsRef<Path>>(self, path: P) -> Result<TrainedModel<K>> {
        let dataset = CSVDataset::from_file(path)?;
        self.train(&dataset)
    }
}

/// Trained SVM model with high-level prediction interface
pub struct TrainedModel<K: Kernel> {
    model: TrainedSVM<K>,
}

impl<K: Kernel> TrainedModel<K> {
    /// Predict a single sample
    pub fn predict(&self, sample: &Sample) -> Prediction {
        self.model.predict(sample)
    }

    /// Predict multiple samples
    pub fn predict_batch(&self, samples: &[Sample]) -> Vec<Prediction> {
        self.model.predict_batch(samples)
    }

    /// Predict from dataset
    pub fn predict_dataset<D: Dataset>(&self, dataset: &D) -> Vec<Prediction> {
        let samples: Vec<Sample> = (0..dataset.len()).map(|i| dataset.get_sample(i)).collect();
        self.predict_batch(&samples)
    }

    /// Predict from LibSVM file
    pub fn predict_from_file<P: AsRef<Path>>(&self, path: P) -> Result<Vec<Prediction>> {
        let dataset = LibSVMDataset::from_file(path)?;
        Ok(self.predict_dataset(&dataset))
    }

    /// Predict from CSV file
    pub fn predict_from_csv<P: AsRef<Path>>(&self, path: P) -> Result<Vec<Prediction>> {
        let dataset = CSVDataset::from_file(path)?;
        Ok(self.predict_dataset(&dataset))
    }

    /// Evaluate accuracy on a dataset
    pub fn evaluate<D: Dataset>(&self, dataset: &D) -> f64 {
        let predictions = self.predict_dataset(dataset);
        let labels = dataset.get_labels();

        let correct = predictions
            .iter()
            .zip(labels.iter())
            .filter(|(pred, &actual)| pred.label == actual)
            .count();

        correct as f64 / labels.len() as f64
    }

    /// Evaluate accuracy from LibSVM file
    pub fn evaluate_from_file<P: AsRef<Path>>(&self, path: P) -> Result<f64> {
        let dataset = LibSVMDataset::from_file(path)?;
        Ok(self.evaluate(&dataset))
    }

    /// Evaluate accuracy from CSV file
    pub fn evaluate_from_csv<P: AsRef<Path>>(&self, path: P) -> Result<f64> {
        let dataset = CSVDataset::from_file(path)?;
        Ok(self.evaluate(&dataset))
    }

    /// Get detailed evaluation metrics
    pub fn evaluate_detailed<D: Dataset>(&self, dataset: &D) -> EvaluationMetrics {
        let predictions = self.predict_dataset(dataset);
        let labels = dataset.get_labels();

        let mut tp = 0; // True positives
        let mut tn = 0; // True negatives
        let mut fp = 0; // False positives
        let mut fn_ = 0; // False negatives

        for (pred, &actual) in predictions.iter().zip(labels.iter()) {
            match (pred.label > 0.0, actual > 0.0) {
                (true, true) => tp += 1,
                (false, false) => tn += 1,
                (true, false) => fp += 1,
                (false, true) => fn_ += 1,
            }
        }

        EvaluationMetrics::new(tp, tn, fp, fn_)
    }

    /// Get model information
    pub fn info(&self) -> ModelInfo {
        ModelInfo {
            n_support_vectors: self.model.n_support_vectors(),
            bias: self.model.bias(),
            support_vector_indices: self.model.support_vector_indices().to_vec(),
        }
    }

    /// Get the underlying trained model
    pub fn inner(&self) -> &TrainedSVM<K> {
        &self.model
    }
}

/// Detailed evaluation metrics
#[derive(Debug, Clone)]
pub struct EvaluationMetrics {
    pub true_positives: usize,
    pub true_negatives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
}

impl EvaluationMetrics {
    fn new(tp: usize, tn: usize, fp: usize, fn_: usize) -> Self {
        Self {
            true_positives: tp,
            true_negatives: tn,
            false_positives: fp,
            false_negatives: fn_,
        }
    }

    /// Calculate accuracy: (TP + TN) / (TP + TN + FP + FN)
    pub fn accuracy(&self) -> f64 {
        let total =
            self.true_positives + self.true_negatives + self.false_positives + self.false_negatives;
        if total == 0 {
            0.0
        } else {
            (self.true_positives + self.true_negatives) as f64 / total as f64
        }
    }

    /// Calculate precision: TP / (TP + FP)
    pub fn precision(&self) -> f64 {
        let denominator = self.true_positives + self.false_positives;
        if denominator == 0 {
            0.0
        } else {
            self.true_positives as f64 / denominator as f64
        }
    }

    /// Calculate recall (sensitivity): TP / (TP + FN)
    pub fn recall(&self) -> f64 {
        let denominator = self.true_positives + self.false_negatives;
        if denominator == 0 {
            0.0
        } else {
            self.true_positives as f64 / denominator as f64
        }
    }

    /// Calculate F1 score: 2 * (precision * recall) / (precision + recall)
    pub fn f1_score(&self) -> f64 {
        let p = self.precision();
        let r = self.recall();
        if p + r == 0.0 {
            0.0
        } else {
            2.0 * (p * r) / (p + r)
        }
    }

    /// Calculate specificity: TN / (TN + FP)
    pub fn specificity(&self) -> f64 {
        let denominator = self.true_negatives + self.false_positives;
        if denominator == 0 {
            0.0
        } else {
            self.true_negatives as f64 / denominator as f64
        }
    }
}

/// Model information
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub n_support_vectors: usize,
    pub bias: f64,
    pub support_vector_indices: Vec<usize>,
}

/// Convenience functions for quick operations
pub mod quick {
    use super::*;

    /// Train a linear SVM on LibSVM data with default parameters
    pub fn train_libsvm<P: AsRef<Path>>(path: P) -> Result<TrainedModel<LinearKernel>> {
        SVM::new().train_from_file(path)
    }

    /// Train a linear SVM on CSV data with default parameters
    pub fn train_csv<P: AsRef<Path>>(path: P) -> Result<TrainedModel<LinearKernel>> {
        SVM::new().train_from_csv(path)
    }

    /// Train with custom C parameter
    pub fn train_libsvm_with_c<P: AsRef<Path>>(
        path: P,
        c: f64,
    ) -> Result<TrainedModel<LinearKernel>> {
        SVM::new().with_c(c).train_from_file(path)
    }

    /// Quick evaluation: train on training file, test on test file
    pub fn evaluate_split<P1: AsRef<Path>, P2: AsRef<Path>>(
        train_path: P1,
        test_path: P2,
    ) -> Result<f64> {
        let model = train_libsvm(train_path)?;
        model.evaluate_from_file(test_path)
    }

    /// Cross-validation helper (simple random split)
    pub fn simple_validation<D: Dataset>(dataset: &D, train_ratio: f64, c: f64) -> Result<f64> {
        if train_ratio <= 0.0 || train_ratio >= 1.0 {
            return Err(SVMError::InvalidParameter(format!(
                "Train ratio must be between 0 and 1, got: {train_ratio}"
            )));
        }

        let n = dataset.len();
        let train_size = (n as f64 * train_ratio) as usize;

        // Simple sequential split (not randomized for reproducibility)
        let train_samples: Vec<Sample> = (0..train_size).map(|i| dataset.get_sample(i)).collect();
        let test_samples: Vec<Sample> = (train_size..n).map(|i| dataset.get_sample(i)).collect();

        let model = SVM::new().with_c(c).train_samples(&train_samples)?;

        let correct = test_samples
            .iter()
            .map(|sample| model.predict(sample))
            .zip(test_samples.iter())
            .filter(|(pred, sample)| pred.label == sample.label)
            .count();

        Ok(correct as f64 / test_samples.len() as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::SparseVector;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_svm_builder_pattern() {
        let svm = SVM::new()
            .with_c(2.0)
            .with_epsilon(0.01)
            .with_max_iterations(5000);

        assert_eq!(svm.config.c, 2.0);
        assert_eq!(svm.config.epsilon, 0.01);
        assert_eq!(svm.config.max_iterations, 5000);
    }

    #[test]
    fn test_quick_training() {
        // Create test data
        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![2.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-2.0]), -1.0),
            Sample::new(SparseVector::new(vec![0], vec![1.5]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-1.5]), -1.0),
        ];

        let model = SVM::new()
            .train_samples(&samples)
            .expect("Training should succeed");

        // Test predictions
        let test_sample = Sample::new(SparseVector::new(vec![0], vec![1.0]), 1.0);
        let prediction = model.predict(&test_sample);
        assert_eq!(prediction.label, 1.0);

        // Test model info
        let info = model.info();
        assert!(info.n_support_vectors > 0);
    }

    #[test]
    fn test_evaluation_metrics() {
        let metrics = EvaluationMetrics::new(10, 5, 2, 3);

        assert_eq!(metrics.accuracy(), 0.75); // (10+5)/(10+5+2+3)
        assert_eq!(metrics.precision(), 10.0 / 12.0); // 10/(10+2)
        assert_eq!(metrics.recall(), 10.0 / 13.0); // 10/(10+3)
        assert!(metrics.f1_score() > 0.0);
        assert_eq!(metrics.specificity(), 5.0 / 7.0); // 5/(5+2)
    }

    #[test]
    fn test_file_operations() {
        // Create temporary LibSVM file
        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        writeln!(temp_file, "+1 1:2.0").expect("Failed to write");
        writeln!(temp_file, "-1 1:-2.0").expect("Failed to write");
        writeln!(temp_file, "+1 1:1.5").expect("Failed to write");
        writeln!(temp_file, "-1 1:-1.5").expect("Failed to write");
        temp_file.flush().expect("Failed to flush");

        // Test training from file
        let model = SVM::new()
            .train_from_file(temp_file.path())
            .expect("Training should succeed");

        // Test evaluation
        let accuracy = model
            .evaluate_from_file(temp_file.path())
            .expect("Evaluation should succeed");
        assert!(accuracy > 0.0);

        // Test quick functions
        let model2 = quick::train_libsvm(temp_file.path()).expect("Quick training should succeed");
        assert!(model2.info().n_support_vectors > 0);
    }

    #[test]
    fn test_simple_validation() {
        // Create mock dataset
        struct MockDataset {
            samples: Vec<Sample>,
        }

        impl Dataset for MockDataset {
            fn len(&self) -> usize {
                self.samples.len()
            }
            fn dim(&self) -> usize {
                1
            }
            fn get_sample(&self, i: usize) -> Sample {
                self.samples[i].clone()
            }
            fn get_labels(&self) -> Vec<f64> {
                self.samples.iter().map(|s| s.label).collect()
            }
        }

        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![2.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-2.0]), -1.0),
            Sample::new(SparseVector::new(vec![0], vec![1.5]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-1.5]), -1.0),
            Sample::new(SparseVector::new(vec![0], vec![1.8]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-1.8]), -1.0),
        ];

        let dataset = MockDataset { samples };
        let accuracy =
            quick::simple_validation(&dataset, 0.7, 1.0).expect("Validation should succeed");
        assert!((0.0..=1.0).contains(&accuracy));
    }
}
