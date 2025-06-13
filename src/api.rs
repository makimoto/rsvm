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

use crate::core::{
    Dataset, OptimizerConfig, Prediction, Result, SVMError, SVMModel, Sample, WorkingSetStrategy,
};
use crate::data::{CSVDataset, LibSVMDataset};
use crate::kernel::{Kernel, LinearKernel};
use crate::optimizer::{SVMOptimizer, TrainedSVM};
use crate::utils::scaling::{ScalingMethod, ScalingParams};
use std::path::Path;

/// High-level SVM interface with builder pattern
pub struct SVM<K: Kernel = LinearKernel> {
    kernel: K,
    config: OptimizerConfig,
    scaling_method: Option<ScalingMethod>,
}

impl SVM<LinearKernel> {
    /// Create a new SVM with linear kernel and default parameters
    pub fn new() -> Self {
        Self {
            kernel: LinearKernel::new(),
            config: OptimizerConfig::default(),
            scaling_method: None,
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
            scaling_method: None,
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

    /// Enable or disable shrinking heuristic
    ///
    /// Shrinking can significantly improve performance on large datasets
    /// by temporarily removing variables that are likely to remain at bounds.
    /// Based on Section 4 of the SVMlight paper.
    pub fn with_shrinking(mut self, enable_shrinking: bool) -> Self {
        self.config.shrinking = enable_shrinking;
        self
    }

    /// Set shrinking frequency (number of iterations between shrinking checks)
    ///
    /// Lower values check more frequently but may add overhead.
    /// Higher values check less frequently but may miss opportunities.
    /// Default is 100 iterations (h parameter in the paper).
    pub fn with_shrinking_iterations(mut self, iterations: usize) -> Self {
        self.config.shrinking_iterations = iterations;
        self
    }

    /// Set working set selection strategy
    ///
    /// - `SMOHeuristic`: Uses max |E_i - E_j| criterion (default, fast)
    /// - `SteepestDescent`: Uses maximum KKT violation (SVMlight paper style, more rigorous)
    /// - `Random`: Random selection (for debugging/comparison)
    pub fn with_working_set_strategy(mut self, strategy: WorkingSetStrategy) -> Self {
        self.config.working_set_strategy = strategy;
        self
    }

    /// Enable automatic feature scaling
    ///
    /// Features will be scaled using the specified method before training.
    /// Scaling parameters are computed from training data and stored for
    /// consistent transformation of prediction data.
    ///
    /// # Arguments
    /// * `method` - Scaling method to use:
    ///   - `MinMax { min_val, max_val }`: Scale to specified range (default: [-1, 1])
    ///   - `StandardScore`: Z-score normalization (mean=0, std=1)
    ///   - `UnitScale`: Scale by maximum absolute value
    pub fn with_feature_scaling(mut self, method: ScalingMethod) -> Self {
        self.scaling_method = Some(method);
        self
    }

    /// Train on a dataset
    pub fn train<D: Dataset>(self, dataset: &D) -> Result<TrainedModel<K>> {
        let samples: Vec<Sample> = (0..dataset.len()).map(|i| dataset.get_sample(i)).collect();
        self.train_samples(&samples)
    }

    /// Train on samples
    pub fn train_samples(self, samples: &[Sample]) -> Result<TrainedModel<K>> {
        let (training_samples, scaling_params) = if let Some(method) = self.scaling_method {
            let params = ScalingParams::fit(samples, method);
            let scaled_samples = params.transform_samples(samples);
            (scaled_samples, Some(params))
        } else {
            (samples.to_vec(), None)
        };

        let optimizer = SVMOptimizer::new(self.kernel, self.config);
        let model = optimizer.train_samples(&training_samples)?;
        Ok(TrainedModel {
            model,
            scaling_params,
        })
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
    scaling_params: Option<ScalingParams>,
}

impl<K: Kernel> TrainedModel<K> {
    /// Create a TrainedModel from a TrainedSVM
    pub fn from_trained_svm(model: TrainedSVM<K>) -> Self {
        Self {
            model,
            scaling_params: None,
        }
    }

    /// Create a TrainedModel from a TrainedSVM with scaling parameters
    pub fn from_trained_svm_with_scaling(
        model: TrainedSVM<K>,
        scaling_params: Option<ScalingParams>,
    ) -> Self {
        Self {
            model,
            scaling_params,
        }
    }

    /// Predict a single sample
    pub fn predict(&self, sample: &Sample) -> Prediction {
        let scaled_sample = if let Some(ref params) = self.scaling_params {
            params.transform_sample(sample)
        } else {
            sample.clone()
        };
        self.model.predict(&scaled_sample)
    }

    /// Predict multiple samples
    pub fn predict_batch(&self, samples: &[Sample]) -> Vec<Prediction> {
        let scaled_samples = if let Some(ref params) = self.scaling_params {
            params.transform_samples(samples)
        } else {
            samples.to_vec()
        };
        self.model.predict_batch(&scaled_samples)
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
        evaluate_split_with_params(train_path, test_path, 1.0, None)
    }

    /// Quick evaluation with custom parameters and scaling
    pub fn evaluate_split_with_params<P1: AsRef<Path>, P2: AsRef<Path>>(
        train_path: P1,
        test_path: P2,
        c: f64,
        scaling: Option<ScalingMethod>,
    ) -> Result<f64> {
        let mut svm_builder = SVM::new().with_c(c);

        if let Some(method) = scaling {
            svm_builder = svm_builder.with_feature_scaling(method);
        }

        let model = svm_builder.train_from_file(train_path)?;
        model.evaluate_from_file(test_path)
    }

    /// Cross-validation helper (simple random split)
    pub fn simple_validation<D: Dataset>(dataset: &D, train_ratio: f64, c: f64) -> Result<f64> {
        simple_validation_with_strategy(dataset, train_ratio, c, WorkingSetStrategy::SMOHeuristic)
    }

    /// Cross-validation helper with configurable working set strategy
    pub fn simple_validation_with_strategy<D: Dataset>(
        dataset: &D,
        train_ratio: f64,
        c: f64,
        strategy: WorkingSetStrategy,
    ) -> Result<f64> {
        simple_validation_with_strategy_and_scaling(dataset, train_ratio, c, strategy, None)
    }

    /// Cross-validation helper with configurable working set strategy and scaling
    pub fn simple_validation_with_strategy_and_scaling<D: Dataset>(
        dataset: &D,
        train_ratio: f64,
        c: f64,
        strategy: WorkingSetStrategy,
        scaling: Option<ScalingMethod>,
    ) -> Result<f64> {
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

        let mut svm_builder = SVM::new().with_c(c).with_working_set_strategy(strategy);

        if let Some(method) = scaling {
            svm_builder = svm_builder.with_feature_scaling(method);
        }

        let model = svm_builder.train_samples(&train_samples)?;

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

    #[test]
    fn test_file_operation_errors() {
        // Test I/O errors for train_from_file
        let result = SVM::new().train_from_file("/nonexistent/path.libsvm");
        assert!(result.is_err());

        // Test I/O errors for train_from_csv
        let result = SVM::new().train_from_csv("/nonexistent/path.csv");
        assert!(result.is_err());

        // Create a valid model first for predict/evaluate error tests
        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![1.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-1.0]), -1.0),
        ];
        let model = SVM::new()
            .train_samples(&samples)
            .expect("Training should succeed");

        // Test I/O errors for predict_from_file
        let result = model.predict_from_file("/nonexistent/path.libsvm");
        assert!(result.is_err());

        // Test I/O errors for predict_from_csv
        let result = model.predict_from_csv("/nonexistent/path.csv");
        assert!(result.is_err());

        // Test I/O errors for evaluate_from_file
        let result = model.evaluate_from_file("/nonexistent/path.libsvm");
        assert!(result.is_err());

        // Test I/O errors for evaluate_from_csv
        let result = model.evaluate_from_csv("/nonexistent/path.csv");
        assert!(result.is_err());
    }

    #[test]
    fn test_evaluation_metrics_edge_cases() {
        // Test edge case: no true positives, no false positives (only true negatives)
        // All samples are negative and correctly predicted
        let metrics = EvaluationMetrics::new(0, 3, 0, 0); // tp=0, tn=3, fp=0, fn=0

        // With no positives, precision should be 0.0 and recall should be 0.0
        assert_eq!(metrics.precision(), 0.0);
        assert_eq!(metrics.recall(), 0.0);
        assert_eq!(metrics.f1_score(), 0.0);
        assert_eq!(metrics.specificity(), 1.0); // All negatives correct

        // Test edge case: no true negatives, no false negatives (only true positives)
        // All samples are positive and correctly predicted
        let metrics = EvaluationMetrics::new(3, 0, 0, 0); // tp=3, tn=0, fp=0, fn=0

        assert_eq!(metrics.precision(), 1.0);
        assert_eq!(metrics.recall(), 1.0);
        assert_eq!(metrics.f1_score(), 1.0);
        assert_eq!(metrics.specificity(), 0.0); // No negatives to be specific about

        // Test edge case: no true positives (all false negatives and true/false negatives)
        let metrics = EvaluationMetrics::new(0, 1, 1, 2); // tp=0, tn=1, fp=1, fn=2

        assert_eq!(metrics.precision(), 0.0); // No true positives
        assert_eq!(metrics.recall(), 0.0); // No true positives
        assert_eq!(metrics.f1_score(), 0.0); // F1 is 0 when precision and recall are 0

        // Test edge case: all false positives (no true positives or true negatives)
        let metrics = EvaluationMetrics::new(0, 0, 3, 0); // tp=0, tn=0, fp=3, fn=0

        assert_eq!(metrics.precision(), 0.0); // No true positives
        assert_eq!(metrics.specificity(), 0.0); // No true negatives

        // Test edge case: division by zero in accuracy with all zeros
        let metrics = EvaluationMetrics::new(0, 0, 0, 0);
        assert_eq!(metrics.accuracy(), 0.0);
    }

    #[test]
    fn test_quick_module_functions() {
        use std::io::Write;

        // Test quick::train_csv
        let mut csv_file = NamedTempFile::new().expect("Failed to create temp file");
        writeln!(csv_file, "2.0,1").expect("Failed to write");
        writeln!(csv_file, "-2.0,-1").expect("Failed to write");
        csv_file.flush().expect("Failed to flush");

        let model = quick::train_csv(csv_file.path()).expect("CSV training should succeed");
        assert!(model.info().n_support_vectors > 0);

        // Test quick::train_libsvm_with_c
        let mut libsvm_file = NamedTempFile::new().expect("Failed to create temp file");
        writeln!(libsvm_file, "+1 1:1.0").expect("Failed to write");
        writeln!(libsvm_file, "-1 1:-1.0").expect("Failed to write");
        libsvm_file.flush().expect("Failed to flush");

        let model = quick::train_libsvm_with_c(libsvm_file.path(), 2.0)
            .expect("LibSVM training with C should succeed");
        assert!(model.info().n_support_vectors > 0);

        // Test quick::evaluate_split
        let accuracy = quick::evaluate_split(libsvm_file.path(), libsvm_file.path())
            .expect("Evaluate split should succeed");
        assert!((0.0..=1.0).contains(&accuracy));
    }

    #[test]
    fn test_svm_with_kernel() {
        use crate::kernel::LinearKernel;

        let kernel = LinearKernel::new();
        let svm = SVM::with_kernel(kernel);

        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![1.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-1.0]), -1.0),
        ];

        let model = svm
            .train_samples(&samples)
            .expect("Training should succeed");
        assert!(model.info().n_support_vectors > 0);
    }

    #[test]
    fn test_svm_default() {
        let svm = SVM::default();

        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![1.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-1.0]), -1.0),
        ];

        let model = svm
            .train_samples(&samples)
            .expect("Training should succeed");
        assert!(model.info().n_support_vectors > 0);
    }

    #[test]
    fn test_svm_with_working_set_strategy() {
        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![2.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-2.0]), -1.0),
            Sample::new(SparseVector::new(vec![0], vec![1.8]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-1.8]), -1.0),
        ];

        // Test SMO heuristic
        let model_smo = SVM::new()
            .with_working_set_strategy(WorkingSetStrategy::SMOHeuristic)
            .train_samples(&samples)
            .expect("Training with SMO heuristic should succeed");
        assert!(model_smo.info().n_support_vectors > 0);

        // Test steepest descent
        let model_steepest = SVM::new()
            .with_working_set_strategy(WorkingSetStrategy::SteepestDescent)
            .train_samples(&samples)
            .expect("Training with steepest descent should succeed");
        assert!(model_steepest.info().n_support_vectors > 0);

        // Test random selection
        let model_random = SVM::new()
            .with_working_set_strategy(WorkingSetStrategy::Random)
            .train_samples(&samples)
            .expect("Training with random selection should succeed");
        assert!(model_random.info().n_support_vectors > 0);
    }

    #[test]
    fn test_svm_builder_pattern_full_configuration() {
        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![2.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-2.0]), -1.0),
        ];

        let model = SVM::new()
            .with_c(0.5)
            .with_epsilon(0.01)
            .with_max_iterations(50)
            .with_cache_size(50_000_000)
            .with_shrinking(true)
            .with_shrinking_iterations(10)
            .with_working_set_strategy(WorkingSetStrategy::SteepestDescent)
            .train_samples(&samples)
            .expect("Training with full configuration should succeed");

        assert!(model.info().n_support_vectors > 0);
    }

    #[test]
    fn test_svm_with_feature_scaling() {
        use crate::utils::scaling::ScalingMethod;

        let samples = vec![
            Sample::new(SparseVector::new(vec![0, 1], vec![100.0, 1000.0]), 1.0),
            Sample::new(SparseVector::new(vec![0, 1], vec![200.0, 2000.0]), -1.0),
            Sample::new(SparseVector::new(vec![0, 1], vec![150.0, 1500.0]), 1.0),
            Sample::new(SparseVector::new(vec![0, 1], vec![50.0, 500.0]), -1.0),
        ];

        // Test MinMax scaling
        let model_minmax = SVM::new()
            .with_feature_scaling(ScalingMethod::MinMax {
                min_val: -1.0,
                max_val: 1.0,
            })
            .train_samples(&samples)
            .expect("Training with MinMax scaling should succeed");

        // Test StandardScore scaling
        let model_std = SVM::new()
            .with_feature_scaling(ScalingMethod::StandardScore)
            .train_samples(&samples)
            .expect("Training with StandardScore scaling should succeed");

        // Test UnitScale scaling
        let model_unit = SVM::new()
            .with_feature_scaling(ScalingMethod::UnitScale)
            .train_samples(&samples)
            .expect("Training with UnitScale scaling should succeed");

        // All models should train successfully
        assert!(model_minmax.info().n_support_vectors > 0);
        assert!(model_std.info().n_support_vectors > 0);
        assert!(model_unit.info().n_support_vectors > 0);

        // Test prediction with scaling (should work with raw unscaled data)
        let test_sample = Sample::new(SparseVector::new(vec![0, 1], vec![125.0, 1250.0]), 1.0);

        let pred_minmax = model_minmax.predict(&test_sample);
        let pred_std = model_std.predict(&test_sample);
        let pred_unit = model_unit.predict(&test_sample);

        // Predictions should be finite and reasonable
        assert!(pred_minmax.decision_value.is_finite());
        assert!(pred_std.decision_value.is_finite());
        assert!(pred_unit.decision_value.is_finite());
    }

    #[test]
    fn test_scaling_preserves_accuracy() {
        use crate::utils::scaling::ScalingMethod;

        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![1000.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-1000.0]), -1.0),
            Sample::new(SparseVector::new(vec![0], vec![800.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-800.0]), -1.0),
        ];

        // Train without scaling
        let model_unscaled = SVM::new()
            .with_c(10.0) // Higher C for better convergence with large values
            .train_samples(&samples)
            .expect("Training without scaling should succeed");

        // Train with scaling
        let model_scaled = SVM::new()
            .with_c(1.0) // Standard C with scaling
            .with_feature_scaling(ScalingMethod::MinMax {
                min_val: -1.0,
                max_val: 1.0,
            })
            .train_samples(&samples)
            .expect("Training with scaling should succeed");

        // Both should achieve good accuracy on this simple linearly separable data
        let accuracy_unscaled = model_unscaled
            .predict_batch(&samples)
            .iter()
            .zip(samples.iter())
            .filter(|(pred, sample)| pred.label == sample.label)
            .count() as f64
            / samples.len() as f64;

        let accuracy_scaled = model_scaled
            .predict_batch(&samples)
            .iter()
            .zip(samples.iter())
            .filter(|(pred, sample)| pred.label == sample.label)
            .count() as f64
            / samples.len() as f64;

        // Both models should work, but scaling helps with numerical stability
        assert!(accuracy_unscaled >= 0.5); // At least better than random
        assert!(accuracy_scaled >= 0.8); // Should be good with scaling

        // The scaled model should generally perform as well or better
        assert!(accuracy_scaled >= accuracy_unscaled - 0.1);
    }

    #[test]
    fn test_simple_validation_edge_cases() {
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
            Sample::new(SparseVector::new(vec![0], vec![1.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-1.0]), -1.0),
        ];
        let dataset = MockDataset { samples };

        // Test invalid train_ratio (> 1.0)
        let result = quick::simple_validation(&dataset, 1.5, 1.0);
        assert!(result.is_err());

        // Test invalid train_ratio (< 0.0)
        let result = quick::simple_validation(&dataset, -0.1, 1.0);
        assert!(result.is_err());

        // Test invalid train_ratio (= 0.0) - should result in no training data
        let result = quick::simple_validation(&dataset, 0.0, 1.0);
        assert!(result.is_err());

        // Test invalid train_ratio (= 1.0) - should result in no test data
        let result = quick::simple_validation(&dataset, 1.0, 1.0);
        assert!(result.is_err());

        // Test very small dataset
        let small_samples = vec![Sample::new(SparseVector::new(vec![0], vec![1.0]), 1.0)];
        let small_dataset = MockDataset {
            samples: small_samples,
        };
        let result = quick::simple_validation(&small_dataset, 0.5, 1.0);
        assert!(result.is_err()); // Should fail with insufficient data
    }

    #[test]
    fn test_evaluate_detailed_edge_cases() {
        // Create mock dataset for evaluate_detailed
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
            Sample::new(SparseVector::new(vec![0], vec![1.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-1.0]), -1.0),
        ];

        let model = SVM::new()
            .train_samples(&samples)
            .expect("Training should succeed");

        // Test with same training data (should get perfect results)
        let dataset = MockDataset {
            samples: samples.clone(),
        };
        let detailed = model.evaluate_detailed(&dataset);
        assert_eq!(detailed.accuracy(), 1.0);
        assert_eq!(detailed.true_positives, 1);
        assert_eq!(detailed.true_negatives, 1);
        assert_eq!(detailed.false_positives, 0);
        assert_eq!(detailed.false_negatives, 0);
    }

    #[test]
    fn test_shrinking_api() {
        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![2.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![1.8]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-2.0]), -1.0),
            Sample::new(SparseVector::new(vec![0], vec![-1.8]), -1.0),
        ];

        // Test with shrinking enabled
        let model_with_shrinking = SVM::new()
            .with_shrinking(true)
            .with_shrinking_iterations(5)
            .train_samples(&samples)
            .expect("Training with shrinking should succeed");

        // Test with shrinking disabled
        let model_without_shrinking = SVM::new()
            .with_shrinking(false)
            .train_samples(&samples)
            .expect("Training without shrinking should succeed");

        // Both should produce valid models
        assert!(model_with_shrinking.info().n_support_vectors > 0);
        assert!(model_without_shrinking.info().n_support_vectors > 0);

        // Test predictions should be similar
        let test_sample = Sample::new(SparseVector::new(vec![0], vec![1.0]), 1.0);
        let pred_with = model_with_shrinking.predict(&test_sample);
        let pred_without = model_without_shrinking.predict(&test_sample);

        // Labels should be the same
        assert_eq!(pred_with.label, pred_without.label);
    }
}
