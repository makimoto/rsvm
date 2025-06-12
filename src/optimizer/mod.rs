//! Optimization algorithms for SVM
//!
//! This module provides high-level optimization interfaces that integrate
//! kernels and solvers to provide complete SVM training functionality.

use crate::core::{
    Dataset, OptimizationResult, OptimizerConfig, Prediction, Result, SVMModel, Sample,
};
use crate::kernel::Kernel;
use crate::solver::SMOSolver;
use std::sync::Arc;

/// High-level SVM optimizer that integrates kernel functions and solving algorithms
pub struct SVMOptimizer<K: Kernel> {
    kernel: Arc<K>,
    config: OptimizerConfig,
}

impl<K: Kernel> SVMOptimizer<K> {
    /// Create a new SVM optimizer with the given kernel and configuration
    pub fn new(kernel: K, config: OptimizerConfig) -> Self {
        Self {
            kernel: Arc::new(kernel),
            config,
        }
    }

    /// Create a new SVM optimizer with default configuration
    pub fn with_kernel(kernel: K) -> Self {
        Self::new(kernel, OptimizerConfig::default())
    }

    /// Train an SVM model on the given dataset
    pub fn train<D: Dataset>(&self, dataset: &D) -> Result<TrainedSVM<K>> {
        // Convert dataset to samples for the solver
        let samples: Vec<Sample> = (0..dataset.len()).map(|i| dataset.get_sample(i)).collect();

        // Create and run the SMO solver
        let solver = SMOSolver::new(Arc::clone(&self.kernel), self.config.clone());
        let result = solver.solve(&samples)?;

        // Create the trained model
        Ok(TrainedSVM::new(Arc::clone(&self.kernel), samples, result))
    }

    /// Train an SVM model on a slice of samples
    pub fn train_samples(&self, samples: &[Sample]) -> Result<TrainedSVM<K>> {
        let solver = SMOSolver::new(Arc::clone(&self.kernel), self.config.clone());
        let result = solver.solve(samples)?;

        Ok(TrainedSVM::new(
            Arc::clone(&self.kernel),
            samples.to_vec(),
            result,
        ))
    }

    /// Get the optimizer configuration
    pub fn config(&self) -> &OptimizerConfig {
        &self.config
    }

    /// Get the kernel
    pub fn kernel(&self) -> &K {
        &self.kernel
    }
}

/// A trained SVM model that can make predictions
pub struct TrainedSVM<K: Kernel> {
    kernel: Arc<K>,
    support_vectors: Vec<Sample>,
    alpha: Vec<f64>,
    bias: f64,
    support_indices: Vec<usize>,
}

impl<K: Kernel> TrainedSVM<K> {
    /// Create a new trained SVM model
    pub(crate) fn new(
        kernel: Arc<K>,
        training_samples: Vec<Sample>,
        optimization_result: OptimizationResult,
    ) -> Self {
        // Extract support vectors and their alpha values
        let mut support_vectors = Vec::new();
        let mut alpha_values = Vec::new();

        for &sv_idx in &optimization_result.support_vectors {
            support_vectors.push(training_samples[sv_idx].clone());
            alpha_values.push(optimization_result.alpha[sv_idx]);
        }

        Self {
            kernel,
            support_vectors,
            alpha: alpha_values,
            bias: optimization_result.b,
            support_indices: optimization_result.support_vectors,
        }
    }

    /// Get the decision function value for a sample
    pub fn decision_function(&self, sample: &Sample) -> f64 {
        let mut result = 0.0;

        for (i, support_vector) in self.support_vectors.iter().enumerate() {
            let kernel_value = self
                .kernel
                .compute(&sample.features, &support_vector.features);
            result += self.alpha[i] * support_vector.label * kernel_value;
        }

        result + self.bias
    }

    /// Get the support vectors
    pub fn support_vectors(&self) -> &[Sample] {
        &self.support_vectors
    }

    /// Get the alpha values for support vectors
    pub fn alpha_values(&self) -> &[f64] {
        &self.alpha
    }

    /// Get the indices of support vectors in the original training set
    pub fn support_vector_indices(&self) -> &[usize] {
        &self.support_indices
    }
}

impl<K: Kernel> SVMModel for TrainedSVM<K> {
    fn predict(&self, sample: &Sample) -> Prediction {
        let decision_value = self.decision_function(sample);
        let label = if decision_value >= 0.0 { 1.0 } else { -1.0 };
        Prediction::new(label, decision_value)
    }

    fn n_support_vectors(&self) -> usize {
        self.support_vectors.len()
    }

    fn bias(&self) -> f64 {
        self.bias
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Sample, SparseVector};
    use crate::kernel::LinearKernel;

    #[test]
    fn test_svm_optimizer_creation() {
        let kernel = LinearKernel::new();
        let config = OptimizerConfig::default();
        let optimizer = SVMOptimizer::new(kernel, config.clone());

        assert_eq!(optimizer.config().c, config.c);
        assert_eq!(optimizer.config().epsilon, config.epsilon);
    }

    #[test]
    fn test_svm_optimizer_with_kernel() {
        let kernel = LinearKernel::new();
        let optimizer = SVMOptimizer::with_kernel(kernel);

        // Should use default config
        assert_eq!(optimizer.config().c, 1.0);
        assert_eq!(optimizer.config().epsilon, 0.001);
    }

    #[test]
    fn test_svm_training_simple_case() {
        let kernel = LinearKernel::new();
        let optimizer = SVMOptimizer::with_kernel(kernel);

        // Create simple linearly separable dataset
        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![2.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-2.0]), -1.0),
            Sample::new(SparseVector::new(vec![0], vec![1.5]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-1.5]), -1.0),
        ];

        let model = optimizer
            .train_samples(&samples)
            .expect("Training should succeed");

        // Test basic model properties
        assert!(model.n_support_vectors() > 0);
        assert!(model.support_vectors().len() > 0);
        assert_eq!(model.alpha_values().len(), model.support_vectors().len());

        // Test predictions on training data
        for sample in &samples {
            let prediction = model.predict(sample);
            assert_eq!(prediction.label, sample.label);
        }
    }

    #[test]
    fn test_trained_svm_decision_function() {
        let kernel = LinearKernel::new();
        let optimizer = SVMOptimizer::with_kernel(kernel);

        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![1.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-1.0]), -1.0),
        ];

        let model = optimizer
            .train_samples(&samples)
            .expect("Training should succeed");

        // Test decision function values
        let test_sample_positive = Sample::new(SparseVector::new(vec![0], vec![0.5]), 1.0);
        let decision_positive = model.decision_function(&test_sample_positive);

        let test_sample_negative = Sample::new(SparseVector::new(vec![0], vec![-0.5]), -1.0);
        let decision_negative = model.decision_function(&test_sample_negative);

        // For linearly separable case, decision values should have correct signs
        assert!(decision_positive > decision_negative);
    }

    #[test]
    fn test_svm_model_trait_implementation() {
        let kernel = LinearKernel::new();
        let optimizer = SVMOptimizer::with_kernel(kernel);

        let samples = vec![
            Sample::new(SparseVector::new(vec![0, 1], vec![1.0, 1.0]), 1.0),
            Sample::new(SparseVector::new(vec![0, 1], vec![-1.0, -1.0]), -1.0),
        ];

        let model = optimizer
            .train_samples(&samples)
            .expect("Training should succeed");

        // Test SVMModel trait methods
        assert!(model.n_support_vectors() > 0);

        let test_samples = vec![
            Sample::new(SparseVector::new(vec![0, 1], vec![0.5, 0.5]), 1.0),
            Sample::new(SparseVector::new(vec![0, 1], vec![-0.5, -0.5]), -1.0),
        ];

        let predictions = model.predict_batch(&test_samples);
        assert_eq!(predictions.len(), 2);

        // Each prediction should have a valid label and decision value
        for prediction in predictions {
            assert!(prediction.label == 1.0 || prediction.label == -1.0);
            assert!(prediction.confidence() >= 0.0);
        }
    }

    #[test]
    fn test_support_vector_access() {
        let kernel = LinearKernel::new();
        let optimizer = SVMOptimizer::with_kernel(kernel);

        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![2.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-2.0]), -1.0),
            Sample::new(SparseVector::new(vec![0], vec![1.0]), 1.0),
        ];

        let model = optimizer
            .train_samples(&samples)
            .expect("Training should succeed");

        // Test access to support vector information
        let support_vectors = model.support_vectors();
        let alpha_values = model.alpha_values();
        let indices = model.support_vector_indices();

        assert_eq!(support_vectors.len(), alpha_values.len());
        assert_eq!(support_vectors.len(), indices.len());

        // All alpha values should be positive (definition of support vectors)
        for &alpha in alpha_values {
            assert!(alpha > 0.0);
        }

        // Indices should be valid
        for &idx in indices {
            assert!(idx < samples.len());
        }
    }
}
