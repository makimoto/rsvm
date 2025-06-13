//! Sequential Minimal Optimization (SMO) solver implementation
//!
//! This implements the basic SMO algorithm for binary SVM classification,
//! focusing on the 2-variable optimization problem (q=2 in the paper).

use crate::cache::KernelCache;
use crate::core::{
    OptimizationResult, OptimizerConfig, Result, SVMError, Sample, WorkingSetStrategy,
};
use crate::kernel::Kernel;
use crate::solver::shrinking::ShrinkingStrategy;
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

        // Initialize shrinking strategy if enabled
        let mut shrinking_strategy = if self.config.shrinking {
            Some(ShrinkingStrategy::new(n, self.config.shrinking_iterations))
        } else {
            None
        };

        // Track active variables (not shrunk)
        let mut active_set: Vec<usize> = (0..n).collect();

        let mut iterations = 0;
        let mut num_changed = 0;
        let mut examine_all = true;
        let mut shrinking_counter = 0;

        // Main SMO loop
        while (num_changed > 0 || examine_all) && iterations < self.config.max_iterations {
            num_changed = 0;

            if examine_all {
                // Examine all active samples
                for &i in &active_set {
                    if self.examine_example(
                        i,
                        samples,
                        &mut alpha,
                        &mut error_cache,
                        &active_set,
                    )? {
                        num_changed += 1;
                    }
                }
            } else {
                // Examine non-bound active samples (0 < alpha < C)
                for &i in &active_set {
                    if alpha[i] > 0.0
                        && alpha[i] < self.config.c
                        && self.examine_example(
                            i,
                            samples,
                            &mut alpha,
                            &mut error_cache,
                            &active_set,
                        )?
                    {
                        num_changed += 1;
                    }
                }
            }

            // Update shrinking strategy and apply shrinking periodically
            if let Some(ref mut strategy) = shrinking_strategy {
                strategy.update(&alpha, &error_cache, samples, self.config.c);

                shrinking_counter += 1;
                if shrinking_counter >= self.config.shrinking_iterations
                    && strategy.has_sufficient_history()
                {
                    let (shrink_to_lower, shrink_to_upper) = strategy.get_shrinkable_variables();

                    // Apply shrinking: remove variables from active set
                    if !shrink_to_lower.is_empty() || !shrink_to_upper.is_empty() {
                        let mut shrunk_count = 0;

                        // Remove variables that should be shrunk to bounds
                        active_set.retain(|&i| {
                            let should_shrink =
                                shrink_to_lower.contains(&i) || shrink_to_upper.contains(&i);
                            if should_shrink {
                                // Fix variables at their bounds
                                if shrink_to_lower.contains(&i) {
                                    alpha[i] = 0.0;
                                } else if shrink_to_upper.contains(&i) {
                                    alpha[i] = self.config.c;
                                }
                                shrunk_count += 1;
                            }
                            !should_shrink
                        });

                        if shrunk_count > 0 {
                            // Update error cache for remaining variables after shrinking
                            self.update_error_cache_after_shrinking(
                                &mut error_cache,
                                &alpha,
                                samples,
                                &active_set,
                            )?;
                        }
                    }

                    shrinking_counter = 0;
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
        active_set: &[usize],
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
            if let Some(j) = self.select_second_variable(
                i,
                e_i,
                &alpha[..],
                &error_cache[..],
                samples,
                &active_set,
            )? {
                if self.take_step(i, j, samples, alpha, error_cache)? {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    /// Select second variable using configured strategy
    fn select_second_variable(
        &self,
        i: usize,
        e_i: f64,
        alpha: &[f64],
        error_cache: &[f64],
        samples: &[Sample],
        active_set: &[usize],
    ) -> Result<Option<usize>> {
        match self.config.working_set_strategy {
            WorkingSetStrategy::SMOHeuristic => {
                self.select_second_variable_smo_heuristic(i, e_i, error_cache, active_set)
            }
            WorkingSetStrategy::SteepestDescent => self.select_second_variable_steepest_descent(
                i,
                e_i,
                alpha,
                error_cache,
                samples,
                active_set,
            ),
            WorkingSetStrategy::Random => self.select_second_variable_random(i, active_set),
        }
    }

    /// Select second variable using SMO heuristic: maximum |E_i - E_j|
    fn select_second_variable_smo_heuristic(
        &self,
        i: usize,
        e_i: f64,
        error_cache: &[f64],
        active_set: &[usize],
    ) -> Result<Option<usize>> {
        let mut best_j = None;
        let mut max_diff = 0.0;

        // Look for the variable that maximizes |E_i - E_j| among active variables
        for &j in active_set {
            if j == i {
                continue;
            }

            let e_j = error_cache[j];
            let diff = (e_i - e_j).abs();

            if diff > max_diff {
                max_diff = diff;
                best_j = Some(j);
            }
        }

        Ok(best_j)
    }

    /// Select second variable using steepest descent: maximum KKT violation
    fn select_second_variable_steepest_descent(
        &self,
        i: usize,
        _e_i: f64,
        alpha: &[f64],
        error_cache: &[f64],
        samples: &[Sample],
        active_set: &[usize],
    ) -> Result<Option<usize>> {
        let mut best_j = None;
        let mut max_violation = 0.0;

        // Find variable with maximum KKT violation among active variables
        for &j in active_set {
            if j == i {
                continue;
            }

            let y_j = samples[j].label;
            let e_j = error_cache[j];
            let alpha_j = alpha[j];

            // Calculate KKT violation: r_j = e_j * y_j
            let r_j = e_j * y_j;

            // KKT violation magnitude
            let violation = if r_j < -self.config.epsilon && alpha_j < self.config.c {
                (-r_j - self.config.epsilon).max(0.0)
            } else if r_j > self.config.epsilon && alpha_j > 0.0 {
                (r_j - self.config.epsilon).max(0.0)
            } else {
                0.0
            };

            if violation > max_violation {
                max_violation = violation;
                best_j = Some(j);
            }
        }

        Ok(best_j)
    }

    /// Select second variable randomly
    fn select_second_variable_random(
        &self,
        i: usize,
        active_set: &[usize],
    ) -> Result<Option<usize>> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Simple deterministic "random" selection based on iteration count
        // In a real implementation, you might use a proper PRNG
        let mut hasher = DefaultHasher::new();
        i.hash(&mut hasher);
        let hash = hasher.finish();

        let available_indices: Vec<usize> =
            active_set.iter().cloned().filter(|&j| j != i).collect();

        if available_indices.is_empty() {
            Ok(None)
        } else {
            let idx = (hash as usize) % available_indices.len();
            Ok(Some(available_indices[idx]))
        }
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

    /// Update error cache after shrinking variables
    ///
    /// When variables are shrunk (fixed at bounds), we need to update
    /// the error cache for remaining active variables to reflect the
    /// contribution of the newly fixed variables.
    fn update_error_cache_after_shrinking(
        &self,
        error_cache: &mut [f64],
        alpha: &[f64],
        samples: &[Sample],
        active_set: &[usize],
    ) -> Result<()> {
        // Recompute error cache for active variables
        // E_i = Σⱼ αⱼ yⱼ K(xᵢ,xⱼ) + b - yᵢ
        // We'll approximate by recomputing from scratch for active variables

        for &i in active_set {
            let mut output = 0.0;

            // Sum over all variables (including shrunk ones)
            for j in 0..samples.len() {
                if alpha[j] > 1e-12 {
                    let k_ij = self
                        .kernel
                        .compute(&samples[i].features, &samples[j].features);
                    output += alpha[j] * samples[j].label * k_ij;
                }
            }

            // Update error: E_i = output_i - y_i
            // Note: we're not adding bias here since it will be computed later
            error_cache[i] = output - samples[i].label;
        }

        Ok(())
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

    #[test]
    fn test_kernel_cached_method() {
        let kernel = Arc::new(LinearKernel::new());
        let config = OptimizerConfig::default();
        let solver = SMOSolver::new(kernel, config);
        let mut cache = KernelCache::with_memory_limit(1000);

        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![1.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![2.0]), -1.0),
        ];

        // Test cache miss then cache hit
        let value1 = solver.kernel_cached(&mut cache, &samples, 0, 1);
        let value2 = solver.kernel_cached(&mut cache, &samples, 0, 1);
        assert_eq!(value1, value2);
        assert!(cache.get(0, 1).is_some());
    }

    #[test]
    fn test_smo_examine_all_transitions() {
        let kernel = Arc::new(LinearKernel::new());
        let mut config = OptimizerConfig::default();
        config.c = 0.5; // Low C to create bound samples
        config.max_iterations = 10;

        let solver = SMOSolver::new(kernel, config);

        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![1.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-1.0]), -1.0),
            Sample::new(SparseVector::new(vec![0], vec![0.5]), 1.0),
        ];

        let result = solver.solve(&samples).expect("Should solve");
        // This should exercise the non-bound examination path
        assert!(result.iterations >= 1);
    }

    #[test]
    fn test_take_step_same_labels() {
        let kernel = Arc::new(LinearKernel::new());
        let config = OptimizerConfig::default();
        let solver = SMOSolver::new(kernel, config);

        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![1.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![2.0]), 1.0), // Same label
            Sample::new(SparseVector::new(vec![0], vec![-1.0]), -1.0),
        ];

        let result = solver.solve(&samples).expect("Should solve");
        // This exercises the same signs branch in take_step
        assert!(result.alpha.len() == 3);
    }

    #[test]
    fn test_take_step_small_c_boundary() {
        let kernel = Arc::new(LinearKernel::new());
        let mut config = OptimizerConfig::default();
        config.c = 0.001; // Very small C to force boundary conditions
        let c_value = config.c; // Store C value before moving config

        let solver = SMOSolver::new(kernel, config);

        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![1.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-1.0]), -1.0),
        ];

        let result = solver.solve(&samples).expect("Should solve");
        // This should exercise boundary conditions in take_step
        assert!(result.alpha.iter().all(|&a| a <= c_value + 1e-10));
    }

    #[test]
    fn test_error_cache_with_larger_dataset() {
        let kernel = Arc::new(LinearKernel::new());
        let config = OptimizerConfig::default();
        let solver = SMOSolver::new(kernel, config);

        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![1.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-1.0]), -1.0),
            Sample::new(SparseVector::new(vec![0], vec![0.5]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-0.5]), -1.0),
            Sample::new(SparseVector::new(vec![0], vec![1.5]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-1.5]), -1.0),
        ];

        let result = solver.solve(&samples).expect("Should solve");
        // Larger dataset ensures error cache update loops run multiple times
        assert!(result.support_vectors.len() > 0);
        assert_eq!(result.alpha.len(), 6);
    }

    #[test]
    fn test_bias_calculation_extreme_cases() {
        let kernel = Arc::new(LinearKernel::new());
        let mut config = OptimizerConfig::default();
        config.c = 0.0001; // Very small C to force specific bias calculation paths

        let solver = SMOSolver::new(kernel, config);

        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![1.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-1.0]), -1.0),
        ];

        let result = solver.solve(&samples).expect("Should solve");
        // This should exercise extreme bias calculation scenarios
        assert!(result.b.is_finite());
    }

    #[test]
    fn test_identical_features_different_labels() {
        let kernel = Arc::new(LinearKernel::new());
        let config = OptimizerConfig::default();
        let solver = SMOSolver::new(kernel, config);

        // Create samples with identical features but different labels
        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![1.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![1.0]), -1.0), // Same features, different label
            Sample::new(SparseVector::new(vec![0], vec![2.0]), 1.0),
        ];

        let result = solver.solve(&samples).expect("Should solve");
        // This exercises challenging second variable selection cases
        assert_eq!(result.alpha.len(), 3);
    }

    #[test]
    fn test_objective_calculation_mixed_alphas() {
        let kernel = Arc::new(LinearKernel::new());
        let mut config = OptimizerConfig::default();
        config.max_iterations = 2; // Limited iterations to keep some alphas at zero

        let solver = SMOSolver::new(kernel, config);

        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![1.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-1.0]), -1.0),
            Sample::new(SparseVector::new(vec![0], vec![0.5]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-0.5]), -1.0),
        ];

        let result = solver.solve(&samples).expect("Should solve");
        // This should exercise both zero and non-zero alpha paths in objective calculation
        assert!(result.objective_value.is_finite());
        assert!(result.objective_value >= 0.0); // Objective should be non-negative
    }

    #[test]
    fn test_zero_kernel_values() {
        let kernel = Arc::new(LinearKernel::new());
        let config = OptimizerConfig::default();
        let solver = SMOSolver::new(kernel, config);

        // Create samples that produce zero kernel values
        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![0.0]), 1.0), // Zero feature
            Sample::new(SparseVector::new(vec![0], vec![0.0]), -1.0), // Zero feature
            Sample::new(SparseVector::new(vec![1], vec![1.0]), 1.0), // Different dimension
        ];

        let result = solver.solve(&samples).expect("Should solve");
        // This exercises zero kernel value handling and eta edge cases
        assert_eq!(result.alpha.len(), 3);
    }

    #[test]
    fn test_high_tolerance_convergence() {
        let kernel = Arc::new(LinearKernel::new());
        let mut config = OptimizerConfig::default();
        config.epsilon = 1.0; // Very high tolerance for quick convergence

        let solver = SMOSolver::new(kernel, config);

        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![0.1]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-0.1]), -1.0),
        ];

        let result = solver.solve(&samples).expect("Should solve");
        // High tolerance should lead to quick convergence with few iterations
        assert!(result.iterations <= 3);
    }

    #[test]
    fn test_shrinking_enabled() {
        let kernel = Arc::new(LinearKernel::new());
        let mut config = OptimizerConfig::default();
        config.shrinking = true;
        config.shrinking_iterations = 5; // Shrink every 5 iterations
        config.max_iterations = 50;

        let solver = SMOSolver::new(kernel, config);

        // Create a larger dataset to test shrinking effectiveness
        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![2.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![1.8]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![1.9]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-2.0]), -1.0),
            Sample::new(SparseVector::new(vec![0], vec![-1.8]), -1.0),
            Sample::new(SparseVector::new(vec![0], vec![-1.9]), -1.0),
        ];

        let result = solver.solve(&samples).expect("Should solve with shrinking");

        // Should find solution
        assert!(result.support_vectors.len() > 0);
        assert!(result.iterations > 0);

        // Should have valid objective value
        assert!(result.objective_value.is_finite());
    }

    #[test]
    fn test_shrinking_vs_no_shrinking() {
        let kernel = Arc::new(LinearKernel::new());

        // Test without shrinking
        let mut config_no_shrinking = OptimizerConfig::default();
        config_no_shrinking.shrinking = false;
        config_no_shrinking.max_iterations = 100;
        let solver_no_shrinking = SMOSolver::new(kernel.clone(), config_no_shrinking);

        // Test with shrinking
        let mut config_with_shrinking = OptimizerConfig::default();
        config_with_shrinking.shrinking = true;
        config_with_shrinking.shrinking_iterations = 10;
        config_with_shrinking.max_iterations = 100;
        let solver_with_shrinking = SMOSolver::new(kernel, config_with_shrinking);

        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![3.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![2.5]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![2.8]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-3.0]), -1.0),
            Sample::new(SparseVector::new(vec![0], vec![-2.5]), -1.0),
            Sample::new(SparseVector::new(vec![0], vec![-2.8]), -1.0),
        ];

        let result_no_shrinking = solver_no_shrinking.solve(&samples).expect("Should solve");
        let result_with_shrinking = solver_with_shrinking.solve(&samples).expect("Should solve");

        // Both should converge to similar solutions
        assert!(result_no_shrinking.support_vectors.len() > 0);
        assert!(result_with_shrinking.support_vectors.len() > 0);

        // Objective values should be close
        let obj_diff =
            (result_no_shrinking.objective_value - result_with_shrinking.objective_value).abs();
        assert!(
            obj_diff < 0.1,
            "Objective values should be similar: {} vs {}",
            result_no_shrinking.objective_value,
            result_with_shrinking.objective_value
        );
    }

    #[test]
    fn test_working_set_strategy_smo_heuristic() {
        let kernel = Arc::new(LinearKernel::new());
        let mut config = OptimizerConfig::default();
        config.working_set_strategy = WorkingSetStrategy::SMOHeuristic;
        config.max_iterations = 50;

        let solver = SMOSolver::new(kernel, config);

        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![2.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-2.0]), -1.0),
            Sample::new(SparseVector::new(vec![0], vec![1.8]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-1.8]), -1.0),
        ];

        let result = solver
            .solve(&samples)
            .expect("Should solve with SMO heuristic");
        assert!(result.support_vectors.len() > 0);
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_working_set_strategy_steepest_descent() {
        let kernel = Arc::new(LinearKernel::new());
        let mut config = OptimizerConfig::default();
        config.working_set_strategy = WorkingSetStrategy::SteepestDescent;
        config.max_iterations = 50;

        let solver = SMOSolver::new(kernel, config);

        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![2.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-2.0]), -1.0),
            Sample::new(SparseVector::new(vec![0], vec![1.8]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-1.8]), -1.0),
        ];

        let result = solver
            .solve(&samples)
            .expect("Should solve with steepest descent");
        assert!(result.support_vectors.len() > 0);
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_working_set_strategy_random() {
        let kernel = Arc::new(LinearKernel::new());
        let mut config = OptimizerConfig::default();
        config.working_set_strategy = WorkingSetStrategy::Random;
        config.max_iterations = 50;

        let solver = SMOSolver::new(kernel, config);

        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![2.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-2.0]), -1.0),
            Sample::new(SparseVector::new(vec![0], vec![1.8]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-1.8]), -1.0),
        ];

        let result = solver
            .solve(&samples)
            .expect("Should solve with random selection");
        assert!(result.support_vectors.len() > 0);
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_working_set_strategies_comparison() {
        let kernel = Arc::new(LinearKernel::new());

        let strategies = [
            WorkingSetStrategy::SMOHeuristic,
            WorkingSetStrategy::SteepestDescent,
            WorkingSetStrategy::Random,
        ];

        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![3.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-3.0]), -1.0),
            Sample::new(SparseVector::new(vec![0], vec![2.5]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-2.5]), -1.0),
            Sample::new(SparseVector::new(vec![0], vec![2.8]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-2.8]), -1.0),
        ];

        let mut results = Vec::new();

        for &strategy in &strategies {
            let mut config = OptimizerConfig::default();
            config.working_set_strategy = strategy;
            config.max_iterations = 100;

            let solver = SMOSolver::new(kernel.clone(), config);
            let result = solver.solve(&samples).expect("Should solve");
            results.push(result);
        }

        // All strategies should converge to reasonable solutions
        for result in &results {
            assert!(result.support_vectors.len() > 0);
            assert!(result.objective_value.is_finite());
            assert!(result.objective_value >= 0.0);
        }

        // Objective values should be similar (within 10% of each other)
        let obj_values: Vec<f64> = results.iter().map(|r| r.objective_value).collect();
        let max_obj = obj_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_obj = obj_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        if max_obj > 0.0 {
            let relative_diff = (max_obj - min_obj) / max_obj;
            assert!(
                relative_diff < 0.1,
                "Objective values should be similar across strategies: {:?}",
                obj_values
            );
        }
    }
}
