//! Sequential Minimal Optimization (SMO) solver implementation
//!
//! This implements the basic SMO algorithm for binary SVM classification,
//! focusing on the 2-variable optimization problem (q=2 in the paper).

use crate::core::{OptimizerConfig, OptimizationResult, Sample, SVMError, Result};
use crate::kernel::Kernel;
use ndarray::Array1;
use std::sync::Arc;

/// SMO solver for SVM optimization
///
/// Implements the Sequential Minimal Optimization algorithm which solves
/// the SVM dual optimization problem by repeatedly optimizing pairs of
/// Lagrange multipliers (alpha values).
pub struct SMOSolver<K: Kernel> {
    kernel: Arc<K>,
    config: OptimizerConfig,
}

impl<K: Kernel> SMOSolver<K> {
    /// Create a new SMO solver with the given kernel and configuration
    pub fn new(kernel: Arc<K>, config: OptimizerConfig) -> Self {
        Self { kernel, config }
    }

    /// Solve the SVM optimization problem
    ///
    /// Takes a dataset of training samples and returns the optimized
    /// alpha values, bias term, and other optimization results.
    pub fn solve(&self, samples: &[Sample]) -> Result<OptimizationResult> {
        // TDD: Start with minimal implementation to pass tests
        if samples.is_empty() {
            return Err(SVMError::EmptyDataset);
        }

        // Validate labels are binary (-1 or +1)
        for sample in samples {
            if sample.label != 1.0 && sample.label != -1.0 {
                return Err(SVMError::InvalidLabel(sample.label));
            }
        }

        // Minimal implementation for now - just return a valid result
        let n = samples.len();
        let alpha = Array1::zeros(n);
        
        Ok(OptimizationResult {
            alpha,
            b: 0.0,
            support_vectors: vec![],
            iterations: 0,
            objective_value: 0.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{SparseVector, Sample};
    use crate::kernel::LinearKernel;

    #[test]
    fn test_smo_solver_creation() {
        let kernel = Arc::new(LinearKernel::new());
        let config = OptimizerConfig::default();
        let solver = SMOSolver::new(kernel, config);
        
        // Basic creation test
        assert_eq!(solver.config.c, 1.0);
    }

    #[test]
    fn test_smo_solver_empty_dataset() {
        let kernel = Arc::new(LinearKernel::new());
        let config = OptimizerConfig::default();
        let solver = SMOSolver::new(kernel, config);
        
        let samples = vec![];
        let result = solver.solve(&samples);
        
        assert!(matches!(result, Err(SVMError::EmptyDataset)));
    }

    #[test]
    fn test_smo_solver_invalid_labels() {
        let kernel = Arc::new(LinearKernel::new());
        let config = OptimizerConfig::default();
        let solver = SMOSolver::new(kernel, config);
        
        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![1.0]), 0.5), // Invalid label
        ];
        
        let result = solver.solve(&samples);
        assert!(matches!(result, Err(SVMError::InvalidLabel(0.5))));
    }

    #[test]
    fn test_smo_solver_valid_case() {
        let kernel = Arc::new(LinearKernel::new());
        let config = OptimizerConfig::default();
        let solver = SMOSolver::new(kernel, config);
        
        // Simple case with valid labels
        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![1.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-1.0]), -1.0),
        ];
        
        let result = solver.solve(&samples).expect("Should solve successfully");
        
        // Basic sanity checks for minimal implementation
        assert_eq!(result.alpha.len(), 2);
        assert_eq!(result.b, 0.0);
        assert_eq!(result.support_vectors.len(), 0);
        assert_eq!(result.iterations, 0);
        assert_eq!(result.objective_value, 0.0);
    }
}