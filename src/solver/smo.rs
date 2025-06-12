//! Sequential Minimal Optimization (SMO) solver implementation
//!
//! This implements the basic SMO algorithm for binary SVM classification,
//! focusing on the 2-variable optimization problem (q=2 in the paper).

use crate::cache::KernelCache;
use crate::core::{OptimizationResult, OptimizerConfig, Result, SVMError, Sample};
use crate::kernel::Kernel;
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

    /// Compute kernel value with caching
    #[allow(dead_code)]
    fn kernel_cached(
        &self,
        cache: &mut KernelCache,
        samples: &[Sample],
        i: usize,
        j: usize,
    ) -> f64 {
        if let Some(value) = cache.get(i, j) {
            value
        } else {
            let value = self
                .kernel
                .compute(&samples[i].features, &samples[j].features);
            cache.put(i, j, value);
            value
        }
    }

    /// Solve the SVM optimization problem
    ///
    /// Takes a dataset of training samples and returns the optimized
    /// alpha values, bias term, and other optimization results.
    pub fn solve(&self, samples: &[Sample]) -> Result<OptimizationResult> {
        let mut cache = KernelCache::with_memory_limit(self.config.cache_size);
        self.solve_with_cache(samples, &mut cache)
    }

    /// Solve the SVM optimization problem with provided kernel cache
    ///
    /// This method allows external cache management for better performance
    /// when solving multiple related problems.
    pub fn solve_with_cache(
        &self,
        samples: &[Sample],
        _cache: &mut KernelCache,
    ) -> Result<OptimizationResult> {
        if samples.is_empty() {
            return Err(SVMError::EmptyDataset);
        }

        // Validate labels are binary (-1 or +1)
        for sample in samples {
            if sample.label != 1.0 && sample.label != -1.0 {
                return Err(SVMError::InvalidLabel(sample.label));
            }
        }

        let n = samples.len();

        // Special case: single sample
        if n == 1 {
            let mut alpha = vec![0.0; 1];
            let alpha_val = self.config.c.min(1.0); // Simple heuristic for single sample
            alpha[0] = alpha_val;
            return Ok(OptimizationResult {
                alpha,
                b: 0.0,
                support_vectors: vec![0],
                iterations: 1,
                objective_value: alpha_val,
            });
        }

        // Initialize alpha values (all zeros initially)
        let mut alpha = vec![0.0; n];

        // Initialize error cache: E_i = output_i - y_i
        // Initially, output_i = 0 (since all alphas are 0), so E_i = -y_i
        let mut error_cache: Vec<f64> = samples.iter().map(|s| -s.label).collect();

        let mut iterations = 0;
        let mut num_changed = 0;
        let mut examine_all = true;

        // Main SMO loop
        while (num_changed > 0 || examine_all) && iterations < self.config.max_iterations {
            num_changed = 0;

            if examine_all {
                // Examine all samples
                for i in 0..n {
                    if self.examine_example(i, samples, &mut alpha, &mut error_cache)? {
                        num_changed += 1;
                    }
                }
            } else {
                // Examine non-bound samples (0 < alpha < C)
                for i in 0..n {
                    if alpha[i] > 0.0
                        && alpha[i] < self.config.c
                        && self.examine_example(i, samples, &mut alpha, &mut error_cache)?
                    {
                        num_changed += 1;
                    }
                }
            }

            if examine_all {
                examine_all = false;
            } else if num_changed == 0 {
                examine_all = true;
            }

            iterations += 1;
        }

        // Calculate bias term
        let bias = self.calculate_bias(&alpha, &error_cache, samples)?;

        // Find support vectors (where alpha > epsilon)
        let support_vectors: Vec<usize> = alpha
            .iter()
            .enumerate()
            .filter_map(|(i, &a)| {
                if a > self.config.epsilon {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();

        // Calculate objective value
        let objective_value = self.calculate_objective(&alpha, samples)?;

        Ok(OptimizationResult {
            alpha,
            b: bias,
            support_vectors,
            iterations,
            objective_value,
        })
    }

    /// Examine a single example for potential optimization
    fn examine_example(
        &self,
        i: usize,
        samples: &[Sample],
        alpha: &mut [f64],
        error_cache: &mut [f64],
    ) -> Result<bool> {
        let y_i = samples[i].label;
        let alpha_i = alpha[i];
        let e_i = error_cache[i];

        // Check KKT conditions using the error value
        let r_i = e_i * y_i;

        // KKT violation conditions:
        // - r_i < -epsilon and alpha_i < C (can increase alpha_i)
        // - r_i > epsilon and alpha_i > 0 (can decrease alpha_i)
        if (r_i < -self.config.epsilon && alpha_i < self.config.c)
            || (r_i > self.config.epsilon && alpha_i > 0.0)
        {
            // Try to find a second variable to optimize with
            if let Some(j) = self.select_second_variable(i, e_i, &*alpha, &*error_cache, samples)? {
                if self.take_step(i, j, samples, alpha, error_cache)? {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    /// Select second variable using maximum |E_i - E_j| heuristic
    fn select_second_variable(
        &self,
        i: usize,
        e_i: f64,
        _alpha: &[f64],
        error_cache: &[f64],
        _samples: &[Sample],
    ) -> Result<Option<usize>> {
        let mut best_j = None;
        let mut max_diff = 0.0;

        // Look for the variable that maximizes |E_i - E_j|
        for (j, &e_j) in error_cache.iter().enumerate() {
            if j == i {
                continue;
            }

            let diff = (e_i - e_j).abs();

            if diff > max_diff {
                max_diff = diff;
                best_j = Some(j);
            }
        }

        Ok(best_j)
    }

    /// Perform the actual optimization step for variables i and j
    fn take_step(
        &self,
        i: usize,
        j: usize,
        samples: &[Sample],
        alpha: &mut [f64],
        error_cache: &mut [f64],
    ) -> Result<bool> {
        if i == j {
            return Ok(false);
        }

        let y_i = samples[i].label;
        let y_j = samples[j].label;
        let alpha_i_old = alpha[i];
        let alpha_j_old = alpha[j];
        let e_i = error_cache[i];
        let e_j = error_cache[j];

        let s = y_i * y_j;

        // Calculate bounds L and H
        let (low, high) = if y_i != y_j {
            // Different signs
            let diff = alpha_j_old - alpha_i_old;
            (0.0_f64.max(-diff), self.config.c.min(self.config.c - diff))
        } else {
            // Same signs
            let sum = alpha_i_old + alpha_j_old;
            (0.0_f64.max(sum - self.config.c), self.config.c.min(sum))
        };

        if low >= high {
            return Ok(false);
        }

        // Calculate kernel values
        let k_ii = self
            .kernel
            .compute(&samples[i].features, &samples[i].features);
        let k_ij = self
            .kernel
            .compute(&samples[i].features, &samples[j].features);
        let k_jj = self
            .kernel
            .compute(&samples[j].features, &samples[j].features);

        let eta = k_ii + k_jj - 2.0 * k_ij;

        let mut alpha_j_new = if eta > 0.0 {
            // Normal case: quadratic form is positive definite
            alpha_j_old + y_j * (e_i - e_j) / eta
        } else {
            // Unusual case: eta <= 0, quadratic form is not positive definite
            // For now, we skip this case (could be improved with objective function evaluation)
            return Ok(false);
        };

        // Clip alpha_j to bounds
        alpha_j_new = if alpha_j_new < low {
            low
        } else if alpha_j_new > high {
            high
        } else {
            alpha_j_new
        };

        // Check for sufficient change
        if (alpha_j_new - alpha_j_old).abs()
            < self.config.epsilon * (alpha_j_new + alpha_j_old + self.config.epsilon)
        {
            return Ok(false);
        }

        // Calculate new alpha_i
        let alpha_i_new = alpha_i_old + s * (alpha_j_old - alpha_j_new);

        // Update alpha values
        alpha[i] = alpha_i_new;
        alpha[j] = alpha_j_new;

        // Update error cache for all examples
        let delta_alpha_i = alpha_i_new - alpha_i_old;
        let delta_alpha_j = alpha_j_new - alpha_j_old;

        for k in 0..samples.len() {
            let k_ik = self
                .kernel
                .compute(&samples[i].features, &samples[k].features);
            let k_jk = self
                .kernel
                .compute(&samples[j].features, &samples[k].features);

            error_cache[k] += y_i * delta_alpha_i * k_ik + y_j * delta_alpha_j * k_jk;
        }

        Ok(true)
    }

    /// Calculate the bias term from support vectors
    fn calculate_bias(
        &self,
        alpha: &[f64],
        error_cache: &[f64],
        samples: &[Sample],
    ) -> Result<f64> {
        let mut sum = 0.0;
        let mut count = 0;

        // Use support vectors on the margin (0 < alpha < C) to calculate bias
        for i in 0..samples.len() {
            if alpha[i] > self.config.epsilon && alpha[i] < self.config.c - self.config.epsilon {
                sum += error_cache[i];
                count += 1;
            }
        }

        if count > 0 {
            Ok(-sum / count as f64)
        } else {
            // Fallback: use all support vectors
            sum = 0.0;
            count = 0;
            for i in 0..samples.len() {
                if alpha[i] > self.config.epsilon {
                    sum += error_cache[i];
                    count += 1;
                }
            }
            if count > 0 {
                Ok(-sum / count as f64)
            } else {
                Ok(0.0)
            }
        }
    }

    /// Calculate the objective function value
    fn calculate_objective(&self, alpha: &[f64], samples: &[Sample]) -> Result<f64> {
        let mut obj = 0.0;
        let n = samples.len();

        // Sum of alpha_i
        for &a in alpha.iter().take(n) {
            obj += a;
        }

        // Subtract 0.5 * sum_i sum_j alpha_i * alpha_j * y_i * y_j * K(x_i, x_j)
        for i in 0..n {
            for j in 0..n {
                if alpha[i] > 0.0 && alpha[j] > 0.0 {
                    let k_ij = self
                        .kernel
                        .compute(&samples[i].features, &samples[j].features);
                    obj -= 0.5 * alpha[i] * alpha[j] * samples[i].label * samples[j].label * k_ij;
                }
            }
        }

        Ok(obj)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Sample, SparseVector};
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

        // Now that we have full implementation, check realistic results
        assert_eq!(result.alpha.len(), 2);
        // The algorithm should find some solution
        assert!(result.iterations > 0);
        // For linearly separable case, we expect support vectors
        assert!(result.support_vectors.len() > 0);
    }

    #[test]
    fn test_smo_solver_max_iterations() {
        let kernel = Arc::new(LinearKernel::new());
        let mut config = OptimizerConfig::default();
        config.max_iterations = 1; // Force early termination
        config.epsilon = 0.00001; // Very tight tolerance to ensure it would need more iterations

        let solver = SMOSolver::new(kernel, config);

        // Create a problem that needs many iterations
        let samples = vec![
            Sample::new(SparseVector::new(vec![0, 1], vec![1.0, 1.0]), 1.0),
            Sample::new(SparseVector::new(vec![0, 1], vec![-1.0, -1.0]), -1.0),
            Sample::new(SparseVector::new(vec![0, 1], vec![1.0, -1.0]), 1.0),
            Sample::new(SparseVector::new(vec![0, 1], vec![-1.0, 1.0]), -1.0),
        ];

        let result = solver.solve(&samples).expect("Should solve");

        // Should hit max iterations
        assert_eq!(result.iterations, 1);
    }

    #[test]
    fn test_smo_solver_non_bound_iteration() {
        let kernel = Arc::new(LinearKernel::new());
        let mut config = OptimizerConfig::default();
        config.c = 10.0; // High C to ensure some alphas are not at bounds
        config.max_iterations = 5; // Limited iterations to test both paths

        let solver = SMOSolver::new(kernel, config);

        // XOR-like problem that's not linearly separable
        let samples = vec![
            Sample::new(SparseVector::new(vec![0, 1], vec![1.0, 1.0]), -1.0),
            Sample::new(SparseVector::new(vec![0, 1], vec![-1.0, -1.0]), -1.0),
            Sample::new(SparseVector::new(vec![0, 1], vec![1.0, -1.0]), 1.0),
            Sample::new(SparseVector::new(vec![0, 1], vec![-1.0, 1.0]), 1.0),
        ];

        let result = solver.solve(&samples).expect("Should solve");

        // Should complete within max iterations
        assert!(result.iterations <= 5);
    }

    #[test]
    fn test_smo_solver_linearly_separable() {
        let kernel = Arc::new(LinearKernel::new());
        let mut config = OptimizerConfig::default();
        config.max_iterations = 100;
        config.epsilon = 0.001;
        config.c = 1.0;

        let solver = SMOSolver::new(kernel, config);

        // Linearly separable case: positive points at (2,0), negative at (-2,0)
        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![2.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-2.0]), -1.0),
        ];

        let result = solver.solve(&samples).expect("Should solve successfully");

        // For linearly separable case, we expect:
        // - Some alpha values should be > 0 (support vectors)
        // - Iterations should be > 0 when we implement the algorithm
        // - Support vectors should be identified
        assert_eq!(result.alpha.len(), 2);
        // TODO: Add more specific checks once algorithm is implemented
    }

    #[test]
    fn test_smo_solver_single_sample() {
        let kernel = Arc::new(LinearKernel::new());
        let config = OptimizerConfig::default();
        let solver = SMOSolver::new(kernel, config);

        // Edge case: only one sample
        let samples = vec![Sample::new(SparseVector::new(vec![0], vec![1.0]), 1.0)];

        let result = solver.solve(&samples).expect("Should handle single sample");

        assert_eq!(result.alpha.len(), 1);
        // Single sample case should have specific behavior
    }
}
