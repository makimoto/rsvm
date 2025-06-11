# SVM Implementation Design Document in Rust

## Overview

This document describes the implementation of a Support Vector Machine (SVM) library in Rust, based on the SVMlight paper "Making Large-Scale SVM Learning Practical" by Thorsten Joachims.

## Mathematical Background

The implementation solves the dual optimization problem from equation 11.1-11.3 in the paper:

```
minimize: W(α) = -Σᵢ αᵢ + (1/2) Σᵢⱼ αᵢαⱼyᵢyⱼK(xᵢ,xⱼ)
subject to: Σᵢ yᵢαᵢ = 0
           0 ≤ αᵢ ≤ C for all i
```

The gradient of W(α) is (equation 11.14):
```
g(α) = -1 + Qα where Q_ij = yᵢyⱼK(xᵢ,xⱼ)
```

The KKT conditions for optimality (equations 11.32-11.34):
- If αᵢ = 0: yᵢ(Σⱼ αⱼyⱼK(xᵢ,xⱼ) + λ^eq) ≥ 1 - ε
- If 0 < αᵢ < C: |yᵢ(Σⱼ αⱼyⱼK(xᵢ,xⱼ) + λ^eq) - 1| ≤ ε
- If αᵢ = C: yᵢ(Σⱼ αⱼyⱼK(xᵢ,xⱼ) + λ^eq) ≤ 1 + ε

where λ^eq is the Lagrange multiplier for the equality constraint, equivalent to the bias term b.

## Key Design Principles

1. **Memory Efficiency**: Linear memory requirement in number of training examples and support vectors
2. **Cache Efficiency**: Exploit kernel matrix symmetry and temporal locality
3. **Numerical Stability**: Careful handling of floating-point operations and convergence criteria
4. **Extensibility**: Easy addition of new kernels and optimization strategies
5. **Type Safety**: Leverage Rust's type system to prevent runtime errors

## Architecture

### Project Structure

```
svm-rust/
├── Cargo.toml
├── README.md          # To be written during development
├── DESIGN.md          # This technical design document
├── LICENSE            # MIT or BSD-3-Clause license file
├── src/
│   ├── lib.rs
│   ├── core/
│   │   ├── mod.rs
│   │   ├── types.rs      # Basic type definitions
│   │   └── traits.rs     # Core traits
│   ├── kernel/
│   │   ├── mod.rs
│   │   ├── traits.rs     # Kernel trait
│   │   ├── linear.rs     # Linear kernel implementation
│   │   └── rbf.rs        # RBF kernel (future)
│   ├── optimizer/
│   │   ├── mod.rs
│   │   ├── traits.rs     # Optimizer trait
│   │   ├── working_set.rs
│   │   ├── decomposition.rs
│   │   └── shrinking.rs
│   ├── solver/
│   │   ├── mod.rs
│   │   ├── traits.rs     # QP solver trait
│   │   └── external.rs   # External solver wrapper
│   ├── data/
│   │   ├── mod.rs
│   │   ├── dense.rs      # Dense matrix support
│   │   └── sparse.rs     # Sparse matrix support
│   ├── cache/
│   │   ├── mod.rs
│   │   └── lru.rs        # LRU cache implementation
│   └── utils/
│       ├── mod.rs
│       └── parallel.rs   # Parallel processing utilities
├── examples/
├── benches/
└── tests/
```

## Core Components

### 1. Core Types and Traits (src/core/)

```rust
// src/core/types.rs
use ndarray::Array1;

#[derive(Debug, Clone)]
pub struct Prediction {
    pub label: f64,
    pub decision_value: f64,
}

#[derive(Clone, Debug)]
pub struct SparseVector {
    pub indices: Vec<usize>,  // Must be sorted in ascending order
    pub values: Vec<f64>,
}

impl SparseVector {
    /// Create a new sparse vector, ensuring indices are sorted
    pub fn new(mut indices: Vec<usize>, mut values: Vec<f64>) -> Self {
        // Sort by indices
        let mut pairs: Vec<_> = indices.into_iter().zip(values).collect();
        pairs.sort_by_key(|&(idx, _)| idx);

        let (indices, values): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
        Self { indices, values }
    }

    /// Get the value at a specific index (0 if not present)
    pub fn get(&self, index: usize) -> f64 {
        match self.indices.binary_search(&index) {
            Ok(pos) => self.values[pos],
            Err(_) => 0.0,
        }
    }

    /// Compute squared L2 norm
    pub fn norm_squared(&self) -> f64 {
        self.values.iter().map(|&v| v * v).sum()
    }
}

pub struct Sample {
    pub features: SparseVector,
    pub label: f64,  // +1 or -1 for binary classification
}

pub struct OptimizationResult {
    pub alpha: Array1<f64>,          // Lagrange multipliers
    pub b: f64,                      // Bias term
    pub support_vectors: Vec<usize>, // Indices of support vectors
    pub iterations: usize,
    pub objective_value: f64,
}

/// Configuration for optimizer
pub struct OptimizerConfig {
    pub c: f64,                      // Regularization parameter (upper bound for alpha)
    pub epsilon: f64,                // Tolerance for KKT conditions
    pub max_iterations: usize,
    pub working_set_size: usize,     // q in the paper (must be even)
    pub cache_size: usize,           // Kernel cache size in bytes
    pub shrinking: bool,             // Enable shrinking heuristic
    pub shrinking_iterations: usize, // h in the paper
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            c: 1.0,
            epsilon: 0.001,
            max_iterations: 10000,
            working_set_size: 2,
            cache_size: 100_000_000, // 100MB
            shrinking: true,
            shrinking_iterations: 100,
        }
    }
}
```

```rust
// src/core/traits.rs
use std::error::Error;

/// Dataset abstraction for efficient data access
pub trait Dataset: Send + Sync {
    /// Number of samples in the dataset
    fn len(&self) -> usize;

    /// Number of features (dimensionality)
    fn dim(&self) -> usize;

    /// Get a single sample by index
    /// # Panics
    /// Panics if index >= len()
    fn get_sample(&self, i: usize) -> Sample;

    /// Get multiple samples efficiently (for parallel processing)
    fn get_batch(&self, indices: &[usize]) -> Vec<Sample>;

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
    fn predict_batch(&self, samples: &[Sample]) -> Vec<Prediction>;

    /// Get the number of support vectors
    fn n_support_vectors(&self) -> usize;
}
```

### 2. Kernel Implementation (src/kernel/)

```rust
// src/kernel/traits.rs
pub trait Kernel: Send + Sync + Clone {
    fn compute(&self, x1: &SparseVector, x2: &SparseVector) -> f64;
    fn compute_row(&self, x: &SparseVector, dataset: &dyn Dataset, indices: &[usize]) -> Vec<f64>;
    fn params(&self) -> KernelParams;
}

#[derive(Debug, Clone)]
pub enum KernelParams {
    Linear,
    RBF { gamma: f64 },
    Polynomial { degree: u32, gamma: f64, coef0: f64 },
}
```

```rust
// src/kernel/linear.rs
#[derive(Clone)]
pub struct LinearKernel;

impl LinearKernel {
    // Efficient sparse dot product - O(n1 + n2) complexity
    fn sparse_dot_product(x1: &SparseVector, x2: &SparseVector) -> f64 {
        let mut result = 0.0;
        let mut i = 0;
        let mut j = 0;

        while i < x1.indices.len() && j < x2.indices.len() {
            if x1.indices[i] == x2.indices[j] {
                result += x1.values[i] * x2.values[j];
                i += 1;
                j += 1;
            } else if x1.indices[i] < x2.indices[j] {
                i += 1;
            } else {
                j += 1;
            }
        }

        result
    }
}

impl Kernel for LinearKernel {
    fn compute(&self, x1: &SparseVector, x2: &SparseVector) -> f64 {
        Self::sparse_dot_product(x1, x2)
    }

    fn compute_row(&self, x: &SparseVector, dataset: &dyn Dataset, indices: &[usize]) -> Vec<f64> {
        use rayon::prelude::*;
        indices.par_iter()
            .map(|&i| {
                let sample = dataset.get_sample(i);
                self.compute(x, &sample.features)
            })
            .collect()
    }

    fn params(&self) -> KernelParams {
        KernelParams::Linear
    }
}
```

### 3. Optimizer (src/optimizer/)

```rust
// src/optimizer/traits.rs
pub trait Optimizer {
    fn optimize<K: Kernel, S: QPSolver>(
        &mut self,
        dataset: &dyn Dataset,
        kernel: &K,
        solver: &S,
        config: &OptimizerConfig,
    ) -> Result<OptimizationResult, Box<dyn Error>>;
}

pub struct OptimizerConfig {
    pub c: f64,                    // Regularization parameter
    pub epsilon: f64,              // Convergence threshold
    pub max_iterations: usize,
    pub working_set_size: usize,   // q in the paper
    pub cache_size: usize,         // In bytes
    pub shrinking: bool,
    pub shrinking_iterations: usize, // h in the paper
}
```

```rust
// src/optimizer/working_set.rs
// Implements working set selection from Section 11.3.2 of the paper
pub struct WorkingSetSelector;

impl WorkingSetSelector {
    /// Select working set using most violating pair strategy from Section 11.3.2
    /// Returns indices of variables to optimize
    pub fn select_most_violating(
        gradient: &Array1<f64>,  // This is g(α) = -1 + s
        alpha: &Array1<f64>,
        y: &Array1<f64>,
        c: f64,
        size: usize,
    ) -> Vec<usize> {
        assert!(size >= 2 && size % 2 == 0, "Working set size must be even and >= 2");

        // Compute violating scores: ωᵢ = yᵢ * gᵢ(α) as in paper
        let mut scores: Vec<(usize, f64)> = gradient.iter()
            .enumerate()
            .map(|(i, &g)| (i, y[i] * g))
            .collect();

        // Sort by ωᵢ in decreasing order
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut working_set = Vec::with_capacity(size);
        let half_size = size / 2;

        // Select top q/2 elements with feasible descent direction (d_i = y_i)
        for &(i, _) in scores.iter() {
            if working_set.len() >= half_size { break; }

            // Check constraints from OP3 (equations 11.21-11.22)
            let feasible = match (alpha[i] == 0.0, alpha[i] == c) {
                (true, false) => y[i] > 0.0,   // d_i = y_i ≥ 0 when α_i = 0
                (false, true) => y[i] < 0.0,   // d_i = y_i ≤ 0 when α_i = C
                (false, false) => true,         // Free variable, any direction ok
                (true, true) => false,          // Impossible state
            };

            if feasible {
                working_set.push(i);
            }
        }

        // Select bottom q/2 elements with feasible ascent direction (d_i = -y_i)
        for &(i, _) in scores.iter().rev() {
            if working_set.len() >= size { break; }
            if working_set.contains(&i) { continue; }

            // Check constraints for d_i = -y_i
            let feasible = match (alpha[i] == 0.0, alpha[i] == c) {
                (true, false) => y[i] < 0.0,   // d_i = -y_i ≥ 0 when α_i = 0
                (false, true) => y[i] > 0.0,   // d_i = -y_i ≤ 0 when α_i = C
                (false, false) => true,         // Free variable
                (true, true) => false,          // Impossible
            };

            if feasible {
                working_set.push(i);
            }
        }

        working_set
    }

    /// Alternative: select maximally violating pair (SMO-style)
    pub fn select_maximal_violating_pair(
        gradient: &Array1<f64>,
        alpha: &Array1<f64>,
        y: &Array1<f64>,
        c: f64,
    ) -> Option<(usize, usize)> {
        let mut i_up = None;
        let mut i_low = None;
        let mut max_up = f64::NEG_INFINITY;
        let mut min_low = f64::INFINITY;

        for i in 0..alpha.len() {
            let g_i = gradient[i];
            let y_i = y[i];
            let alpha_i = alpha[i];

            // Check if i can move up
            if (alpha_i < c && y_i > 0.0) || (alpha_i > 0.0 && y_i < 0.0) {
                if -y_i * g_i > max_up {
                    max_up = -y_i * g_i;
                    i_up = Some(i);
                }
            }

            // Check if i can move down
            if (alpha_i < c && y_i < 0.0) || (alpha_i > 0.0 && y_i > 0.0) {
                if -y_i * g_i < min_low {
                    min_low = -y_i * g_i;
                    i_low = Some(i);
                }
            }
        }

        match (i_up, i_low) {
            (Some(i), Some(j)) if max_up + min_low > 2.0 * 1e-3 => Some((i, j)),
            _ => None, // No violating pair found
        }
    }
}
```

```rust
// src/optimizer/decomposition.rs
use ndarray::Array1;
use std::collections::HashSet;

pub struct DecompositionOptimizer {
    // Core optimization state
    alpha: Array1<f64>,      // Lagrange multipliers
    gradient: Array1<f64>,   // Gradient vector g(α) = 1 + Qα

    // Cache for kernel values
    kernel_cache: KernelCache,

    // Active set (indices not at bounds)
    active_set: HashSet<usize>,

    // Shrinking support
    working_indices: Vec<usize>, // Indices still in optimization
}

impl DecompositionOptimizer {
    pub fn new(n_samples: usize, cache_size: usize) -> Self {
        Self {
            alpha: Array1::zeros(n_samples),
            gradient: Array1::from_elem(n_samples, -1.0), // g = -1 initially (from ∇W(α))
            kernel_cache: KernelCache::new(cache_size),
            active_set: HashSet::new(),
            working_indices: (0..n_samples).collect(),
        }
    }

    /// Main optimization loop implementing Algorithm from Section 11.2 of the paper
    pub fn optimize<K: Kernel>(
        &mut self,
        dataset: &dyn Dataset,
        kernel: &K,
        solver: &dyn QPSolver,
        config: &OptimizerConfig,
    ) -> Result<OptimizationResult, Box<dyn Error>> {
        let labels = dataset.get_labels();
        let y = Array1::from_vec(labels);

        // Validate labels are binary (-1 or +1)
        for &label in y.iter() {
            if label != 1.0 && label != -1.0 {
                return Err("Labels must be -1 or +1 for binary classification".into());
            }
        }

        // Initialize shrinking if enabled
        if config.shrinking {
            self.shrinking_strategy = Some(ShrinkingStrategy::new(
                dataset.len(),
                config.shrinking_iterations
            ));
        }

        let mut iter = 0;
        let mut shrinking_counter = 0;

        while iter < config.max_iterations {
            // Get current gradient g(α) = -1 + s
            let gradient = self.get_gradient();

            // Step 1: Select working set B using algorithm from Section 11.3
            let working_set = if config.working_set_size == 2 {
                // Use SMO-style pair selection for size 2
                match WorkingSetSelector::select_maximal_violating_pair(
                    &gradient, &self.alpha, &y, config.c
                ) {
                    Some((i, j)) => vec![i, j],
                    None => {
                        log::info!("No violating pair found, checking convergence");
                        if self.check_kkt_conditions(&y, config.c, config.epsilon) {
                            break;
                        }
                        vec![] // Will trigger final convergence check
                    }
                }
            } else {
                WorkingSetSelector::select_most_violating(
                    &gradient, &self.alpha, &y, config.c, config.working_set_size
                )
            };

            if working_set.is_empty() {
                log::info!("Empty working set, optimization complete");
                break;
            }

            // Step 2: Solve QP subproblem on B (Section 11.2)
            let old_alpha: Vec<f64> = working_set.iter().map(|&i| self.alpha[i]).collect();
            let new_alpha_b = self.solve_subproblem(
                dataset, kernel, solver, &working_set, &y, config.c
            )?;

            // Step 3: Update alpha for working set
            for (idx, &i) in working_set.iter().enumerate() {
                self.alpha[i] = new_alpha_b[idx];
            }

            // Step 4: Update s vector efficiently (equation 11.37)
            self.update_s_vector(dataset, kernel, &working_set, &old_alpha, &y);

            // Step 5: Apply shrinking heuristic (Section 11.4)
            if config.shrinking {
                shrinking_counter += 1;
                if shrinking_counter >= config.shrinking_iterations {
                    self.apply_shrinking(&y, config.c, config.epsilon);
                    shrinking_counter = 0;
                }
            }

            // Step 6: Check convergence every 10 iterations for efficiency
            if iter % 10 == 0 && self.check_kkt_conditions(&y, config.c, config.epsilon) {
                log::info!("KKT conditions satisfied at iteration {}", iter);
                break;
            }

            iter += 1;

            if iter % 100 == 0 {
                log::debug!("Iteration {}: {} support vectors, {} working",
                          iter,
                          self.alpha.iter().filter(|&&a| a > 1e-8).count(),
                          self.working_indices.len());
            }
        }

        if iter == config.max_iterations {
            log::warn!("Maximum iterations reached without convergence");
        }

        // Final convergence check on all variables (including shrunk ones)
        if config.shrinking && !self.working_indices.is_empty() {
            self.restore_shrunk_variables();
            if !self.check_kkt_conditions(&y, config.c, config.epsilon) {
                log::info!("Reoptimizing with all variables");
                // Could continue optimization here with shrinking disabled
            }
        }

        // Compute bias term b (which is λ^eq from equation 11.29)
        let b = self.estimate_lambda_eq(&y, config.c);

        // Compute objective value W(α) = -Σᵢ αᵢ + 0.5 * Σᵢⱼ αᵢαⱼyᵢyⱼK(xᵢ,xⱼ)
        let objective_value = self.compute_objective_value(dataset, kernel, &y);

        // Identify support vectors
        let support_vectors: Vec<usize> = self.alpha.iter()
            .enumerate()
            .filter(|(_, &alpha)| alpha > 1e-8)
            .map(|(i, _)| i)
            .collect();

        log::info!("Optimization complete: {} iterations, {} SVs (objective: {})",
                  iter, support_vectors.len(), objective_value);

        Ok(OptimizationResult {
            alpha: self.alpha.clone(),
            b,
            support_vectors,
            iterations: iter,
            objective_value,
        })
    }

    /// Compute objective value W(α) from equation 11.1
    /// W(α) = -Σᵢ αᵢ + 0.5 * Σᵢⱼ αᵢαⱼyᵢyⱼK(xᵢ,xⱼ)
    fn compute_objective_value<K: Kernel>(
        &self,
        dataset: &dyn Dataset,
        kernel: &K,
        y: &Array1<f64>,
    ) -> f64 {
        // Linear term
        let mut obj = -self.alpha.sum();

        // Quadratic term using cached kernel values
        for i in 0..self.alpha.len() {
            if self.alpha[i] == 0.0 { continue; }

            // Note: we can use s[i] = Σⱼ αⱼyⱼK(xᵢ,xⱼ) for efficiency
            obj += 0.5 * self.alpha[i] * y[i] * self.s[i];
        }

        obj
    }

    /// Update gradient after alpha changes
    /// gradient[i] = -1 + sum_j (alpha[j] * y[j] * K(i,j))
    fn update_gradient(
        &mut self,
        dataset: &dyn Dataset,
        kernel: &impl Kernel,
        changed_indices: &[usize],
        old_alpha: &[f64],
        y: &Array1<f64>,
    ) {
        // For each changed alpha
        for (idx, &j) in changed_indices.iter().enumerate() {
            let delta_alpha_j = self.alpha[j] - old_alpha[idx];
            if delta_alpha_j.abs() < 1e-12 {
                continue; // No significant change
            }

            let sample_j = dataset.get_sample(j);

            // Update gradient for all working indices
            // This is the key optimization: only compute kernel values as needed
            for &i in &self.working_indices {
                let k_ij = self.kernel_cache.get_or_compute(i, j, || {
                    let sample_i = dataset.get_sample(i);
                    kernel.compute(&sample_i.features, &sample_j.features)
                });

                self.gradient[i] += delta_alpha_j * y[j] * k_ij;
            }
        }
    }

    /// Check KKT conditions for convergence (equations 11.32-11.34 from paper)
    fn check_kkt_conditions(&self, y: &Array1<f64>, c: f64, epsilon: f64) -> bool {
        for &i in &self.working_indices {
            let alpha_i = self.alpha[i];
            let y_i = y[i];
            let grad_i = self.gradient[i];

            // Case 1: alpha_i = 0 (equation 11.33)
            if alpha_i < epsilon {
                if y_i * grad_i < -epsilon {
                    return false;
                }
            }
            // Case 2: alpha_i = C (equation 11.34)
            else if alpha_i > c - epsilon {
                if y_i * grad_i > epsilon {
                    return false;
                }
            }
            // Case 3: 0 < alpha_i < C (equation 11.32)
            else {
                if (y_i * grad_i).abs() > epsilon {
                    return false;
                }
            }
        }

        true
    }

    /// Compute bias term from KKT conditions
    fn compute_bias(&self, y: &Array1<f64>, c: f64) -> f64 {
        let mut sum = 0.0;
        let mut count = 0;

        // Use free support vectors (0 < alpha < C) for stable computation
        for i in 0..self.alpha.len() {
            if self.alpha[i] > 1e-8 && self.alpha[i] < c - 1e-8 {
                sum += y[i] - self.gradient[i];
                count += 1;
            }
        }

        if count > 0 {
            sum / count as f64
        } else {
            0.0
        }
    }
}
```

### 4. Shrinking Implementation (src/optimizer/shrinking.rs)

```rust
// src/optimizer/shrinking.rs
use std::collections::VecDeque;

/// Tracks Lagrange multiplier estimates for shrinking heuristic
pub struct ShrinkingStrategy {
    // History of multiplier estimates for each variable
    lower_bound_history: Vec<VecDeque<bool>>,
    upper_bound_history: Vec<VecDeque<bool>>,
    history_size: usize,
}

impl ShrinkingStrategy {
    pub fn new(n_samples: usize, history_size: usize) -> Self {
        Self {
            lower_bound_history: vec![VecDeque::with_capacity(history_size); n_samples],
            upper_bound_history: vec![VecDeque::with_capacity(history_size); n_samples],
            history_size,
        }
    }

    /// Update history with current multiplier estimates
    pub fn update(
        &mut self,
        gradient: &Array1<f64>,
        alpha: &Array1<f64>,
        y: &Array1<f64>,
        c: f64,
    ) {
        for i in 0..alpha.len() {
            // Compute multiplier estimates (equations 11.30-11.31)
            let lambda_lower = y[i] * gradient[i] - 1.0;
            let lambda_upper = y[i] * gradient[i] + 1.0;

            // Track if estimates indicate variable at bound
            let at_lower = alpha[i] < 1e-8 && lambda_lower > 0.0;
            let at_upper = alpha[i] > c - 1e-8 && lambda_upper < 0.0;

            // Update history
            if self.lower_bound_history[i].len() >= self.history_size {
                self.lower_bound_history[i].pop_front();
            }
            self.lower_bound_history[i].push_back(at_lower);

            if self.upper_bound_history[i].len() >= self.history_size {
                self.upper_bound_history[i].pop_front();
            }
            self.upper_bound_history[i].push_back(at_upper);
        }
    }

    /// Identify variables that can be shrunk
    pub fn get_shrinkable_indices(&self) -> (Vec<usize>, Vec<usize>) {
        let mut shrink_to_lower = Vec::new();
        let mut shrink_to_upper = Vec::new();

        for i in 0..self.lower_bound_history.len() {
            // Check if consistently at lower bound
            if self.lower_bound_history[i].len() == self.history_size &&
               self.lower_bound_history[i].iter().all(|&x| x) {
                shrink_to_lower.push(i);
            }

            // Check if consistently at upper bound
            if self.upper_bound_history[i].len() == self.history_size &&
               self.upper_bound_history[i].iter().all(|&x| x) {
                shrink_to_upper.push(i);
            }
        }

        (shrink_to_lower, shrink_to_upper)
    }
}
```

### 5. QP Solver Integration (src/solver/)

```rust
// src/solver/traits.rs
pub trait QPSolver: Send + Sync {
    /// Solve quadratic programming problem:
    /// minimize: 0.5 * x^T * Q * x + p^T * x
    /// subject to: A_eq * x = b_eq, lb <= x <= ub
    fn solve(
        &self,
        q: &Array2<f64>,      // Quadratic term (must be symmetric)
        p: &Array1<f64>,      // Linear term
        a_eq: &Array2<f64>,   // Equality constraint matrix
        b_eq: &Array1<f64>,   // Equality constraint vector
        bounds: &[(f64, f64)], // Variable bounds (lower, upper)
    ) -> Result<Array1<f64>, Box<dyn Error>>;

    /// Get solver name for logging/debugging
    fn name(&self) -> &str;

    /// Downcast for specialized handling
    fn as_any(&self) -> &dyn std::any::Any;
}

// src/solver/external.rs
// Wrapper for external QP solvers with MIT/BSD-compatible licenses

#[cfg(feature = "osqp")]
pub struct OSQPSolver {
    settings: osqp::Settings,
}

#[cfg(feature = "osqp")]
impl OSQPSolver {
    pub fn new() -> Self {
        let mut settings = osqp::Settings::default();
        settings.verbose(false);
        settings.scaled_termination(true);
        settings.max_iter(10000);
        settings.eps_abs(1e-6);
        settings.eps_rel(1e-6);
        Self { settings }
    }
}

#[cfg(feature = "osqp")]
impl QPSolver for OSQPSolver {
    fn solve(
        &self,
        q: &Array2<f64>,
        p: &Array1<f64>,
        a_eq: &Array2<f64>,
        b_eq: &Array1<f64>,
        bounds: &[(f64, f64)],
    ) -> Result<Array1<f64>, Box<dyn Error>> {
        // Convert to OSQP sparse format
        let P = CscMatrix::from_dense(q);
        let A = CscMatrix::from_dense(a_eq);

        // Set up bounds
        let (l, u): (Vec<f64>, Vec<f64>) = bounds.iter().cloned().unzip();

        // Create and solve problem
        let mut prob = osqp::Problem::new(P, p, A, &l, &u, &self.settings)?;
        let result = prob.solve();

        match result.status() {
            osqp::Status::Solved => Ok(Array1::from_vec(result.x().to_vec())),
            _ => Err(format!("OSQP failed: {:?}", result.status()).into()),
        }
    }
}

// Analytical solver for 2-variable QP (SMO-style)
pub struct AnalyticalQPSolver;

impl AnalyticalQPSolver {
    pub fn new() -> Self {
        Self
    }

    /// Solve 2-variable QP analytically (for working sets of size 2)
    /// This is the core of SMO algorithm and serves as fallback
    fn solve_two_variable(
        &self,
        q11: f64, q12: f64, q22: f64,
        p1: f64, p2: f64,
        y1: f64, y2: f64,
        c: f64,
        old_alpha1: f64, old_alpha2: f64,
    ) -> (f64, f64) {
        // Compute bounds for alpha2
        let (l, h) = if y1 != y2 {
            (
                f64::max(0.0, old_alpha2 - old_alpha1),
                f64::min(c, c + old_alpha2 - old_alpha1),
            )
        } else {
            (
                f64::max(0.0, old_alpha1 + old_alpha2 - c),
                f64::min(c, old_alpha1 + old_alpha2),
            )
        };

        if l >= h {
            return (old_alpha1, old_alpha2);
        }

        // Compute unconstrained optimum for alpha2
        let eta = q11 + q22 - 2.0 * q12;
        if eta <= 0.0 {
            return (old_alpha1, old_alpha2); // Non-positive definite
        }

        let delta = -y2 * (p1 - p2) / eta;
        let alpha2_new = (old_alpha2 + delta).clamp(l, h);

        // Compute alpha1 from equality constraint
        let alpha1_new = old_alpha1 + y1 * y2 * (old_alpha2 - alpha2_new);

        (alpha1_new, alpha2_new)
    }
}

impl QPSolver for AnalyticalQPSolver {
    fn solve(
        &self,
        q: &Array2<f64>,
        p: &Array1<f64>,
        _a_eq: &Array2<f64>,
        _b_eq: &Array1<f64>,
        bounds: &[(f64, f64)],
    ) -> Result<Array1<f64>, Box<dyn Error>> {
        if q.nrows() != 2 {
            return Err("Analytical solver only supports 2-variable problems".into());
        }

        // Extract problem data
        let (old_alpha1, old_alpha2) = (bounds[0].0, bounds[1].0);
        let c = bounds[0].1; // Assuming same upper bound

        let (alpha1, alpha2) = self.solve_two_variable(
            q[(0, 0)], q[(0, 1)], q[(1, 1)],
            p[0], p[1],
            1.0, -1.0, // y values would come from context
            c,
            old_alpha1, old_alpha2,
        );

        Ok(Array1::from_vec(vec![alpha1, alpha2]))
    }
}

// Factory function to get best available solver
pub fn get_qp_solver() -> Box<dyn QPSolver> {
    #[cfg(feature = "osqp")]
    {
        Box::new(OSQPSolver::new())
    }
    #[cfg(not(feature = "osqp"))]
    {
        log::info!("Using analytical QP solver for 2-variable problems");
        Box::new(AnalyticalQPSolver::new())
    }
}

```rust
// src/cache/lru.rs
use lru::LruCache;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct KernelCache {
    cache: Arc<Mutex<LruCache<(usize, usize), f64>>>,
    hits: AtomicUsize,
    misses: AtomicUsize,
}

impl KernelCache {
    pub fn new(capacity: usize) -> Self {
        // Convert bytes to number of entries (8 bytes per f64)
        let n_entries = capacity / 8;
        Self {
            cache: Arc::new(Mutex::new(LruCache::new(n_entries))),
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
        }
    }

    pub fn get_or_compute<F>(&self, i: usize, j: usize, compute: F) -> f64
    where
        F: FnOnce() -> f64,
    {
        // Ensure i <= j for cache key (exploit symmetry)
        let key = if i <= j { (i, j) } else { (j, i) };

        // Try to get from cache
        if let Some(&value) = self.cache.lock().unwrap().get(&key) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return value;
        }

        // Compute and cache
        self.misses.fetch_add(1, Ordering::Relaxed);
        let value = compute();
        self.cache.lock().unwrap().put(key, value);
        value
    }

    pub fn stats(&self) -> (usize, usize) {
        (self.hits.load(Ordering::Relaxed),
         self.misses.load(Ordering::Relaxed))
    }
}
```

### 6. Main API (src/lib.rs)

```rust
// src/lib.rs
use std::error::Error;
use log::{info, debug};

pub struct SVM<K: Kernel> {
    kernel: K,
    model: Option<TrainedModel>,
    config: SVMConfig,
}

pub struct SVMConfig {
    pub c: f64,                  // Regularization parameter
    pub epsilon: f64,            // KKT tolerance (default: 0.001)
    pub cache_size: Option<usize>, // Kernel cache size in bytes (None = auto)
    pub max_iterations: usize,
    pub working_set_size: usize, // Must be even, typically 2
    pub shrinking: bool,
    pub shrinking_iterations: usize, // h in paper (default: 100)
    pub n_jobs: Option<usize>,   // Number of threads for parallelization
}

impl Default for SVMConfig {
    fn default() -> Self {
        Self {
            c: 1.0,
            epsilon: 0.001,
            cache_size: None, // Auto-detect based on available memory
            max_iterations: 10000,
            working_set_size: 2,
            shrinking: true,
            shrinking_iterations: 100,
            n_jobs: None, // Use all available cores
        }
    }
}

impl SVMConfig {
    /// Calculate appropriate cache size based on available memory
    fn auto_cache_size() -> usize {
        // Get available memory and use 25% for cache
        // This is a placeholder - implement actual memory detection
        100_000_000 // 100MB default
    }
}

/// Trained SVM model
pub struct TrainedModel {
    support_vectors: Vec<Sample>,
    alpha_y: Vec<f64>,  // alpha[i] * y[i] for each support vector
    b: f64,
    kernel: Box<dyn Kernel>,
}

impl TrainedModel {
    fn from_optimization_result<K: Kernel>(
        result: OptimizationResult,
        dataset: &dyn Dataset,
        kernel: K,
    ) -> Result<Self, Box<dyn Error>> {
        let mut support_vectors = Vec::new();
        let mut alpha_y = Vec::new();

        for &idx in &result.support_vectors {
            let sample = dataset.get_sample(idx);
            let alpha = result.alpha[idx];
            alpha_y.push(alpha * sample.label);
            support_vectors.push(sample);
        }

        Ok(Self {
            support_vectors,
            alpha_y,
            b: result.b,
            kernel: Box::new(kernel),
        })
    }
}

impl SVMModel for TrainedModel {
    fn predict(&self, sample: &Sample) -> Prediction {
        let mut decision_value = self.b;

        // Compute decision function: f(x) = sum(alpha_i * y_i * K(x_i, x)) + b
        for (i, sv) in self.support_vectors.iter().enumerate() {
            let k = self.kernel.compute(&sv.features, &sample.features);
            decision_value += self.alpha_y[i] * k;
        }

        Prediction {
            label: if decision_value > 0.0 { 1.0 } else { -1.0 },
            decision_value,
        }
    }

    fn predict_batch(&self, samples: &[Sample]) -> Vec<Prediction> {
        use rayon::prelude::*;

        samples.par_iter()
            .map(|sample| self.predict(sample))
            .collect()
    }

    fn n_support_vectors(&self) -> usize {
        self.support_vectors.len()
    }
}

impl<K: Kernel> SVM<K> {
    /// Create a new SVM with specified kernel and configuration
    pub fn new(kernel: K, config: SVMConfig) -> Self {
        // Validate configuration
        assert!(config.c > 0.0, "C must be positive");
        assert!(config.epsilon > 0.0, "Epsilon must be positive");
        assert!(config.working_set_size % 2 == 0, "Working set size must be even");
        assert!(config.working_set_size >= 2, "Working set size must be at least 2");

        Self {
            kernel,
            model: None,
            config,
        }
    }

    /// Train the SVM model
    pub fn fit<D: Dataset>(&mut self, dataset: &D) -> Result<(), Box<dyn Error>> {
        info!("Starting SVM training with {} samples", dataset.len());

        // Validate dataset
        if dataset.is_empty() {
            return Err("Dataset is empty".into());
        }

        // Set up parallel processing
        if let Some(n_jobs) = self.config.n_jobs {
            rayon::ThreadPoolBuilder::new()
                .num_threads(n_jobs)
                .build_global()?;
            info!("Using {} threads for parallel processing", n_jobs);
        }

        // Configure optimizer
        let optimizer_config = OptimizerConfig {
            c: self.config.c,
            epsilon: self.config.epsilon,
            max_iterations: self.config.max_iterations,
            working_set_size: self.config.working_set_size,
            cache_size: self.config.cache_size,
            shrinking: self.config.shrinking,
            shrinking_iterations: self.config.shrinking_iterations,
        };

        // Run optimization
        let mut optimizer = DecompositionOptimizer::new(dataset.len(), self.config.cache_size);
        let solver = get_qp_solver();

        let result = optimizer.optimize(dataset, &self.kernel, &*solver, &optimizer_config)?;

        info!(
            "Training completed: {} iterations, {} support vectors",
            result.iterations, result.support_vectors.len()
        );

        // Store trained model
        self.model = Some(TrainedModel::from_optimization_result(
            result,
            dataset,
            self.kernel.clone(),
        )?);

        Ok(())
    }

    /// Predict a single sample
    pub fn predict(&self, sample: &Sample) -> Result<Prediction, Box<dyn Error>> {
        self.model
            .as_ref()
            .ok_or("Model not trained")?
            .predict(sample)
            .into()
    }

    /// Predict multiple samples in parallel
    pub fn predict_batch(&self, samples: &[Sample]) -> Result<Vec<Prediction>, Box<dyn Error>> {
        Ok(self.model
            .as_ref()
            .ok_or("Model not trained")?
            .predict_batch(samples))
    }

    /// Get model information
    pub fn model_info(&self) -> Option<ModelInfo> {
        self.model.as_ref().map(|m| ModelInfo {
            n_support_vectors: m.n_support_vectors(),
            bias: m.b,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub n_support_vectors: usize,
    pub bias: f64,
}
```

## Implementation Notes

### Performance Optimizations

1. **Sparse Vector Operations**: Always maintain sorted indices for O(n) dot products
2. **Kernel Caching**: LRU cache exploits kernel matrix symmetry
3. **Gradient Updates**: Only update based on changed alphas (equation 11.37)
4. **Shrinking**: Remove bounded support vectors during optimization (Section 11.4)
5. **Parallel Processing**: Use rayon for kernel row computations

### Key Algorithms from the Paper

1. **Working Set Selection** (Section 11.3.2): Select q/2 most violating examples from top and bottom
2. **Convergence Check**: KKT conditions with epsilon tolerance (equations 11.32-11.34)
3. **Shrinking Heuristic** (Section 11.4): Track Lagrange multiplier estimates over h iterations
4. **Caching Strategy**: LRU replacement with symmetry exploitation

### Error Handling

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SVMError {
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Optimization failed: {0}")]
    OptimizationError(String),

    #[error("Model not trained")]
    ModelNotTrained,

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, SVMError>;
```

## Dependencies and License Considerations

### License Policy
This project targets MIT/BSD-3-Clause licensing. All dependencies must be compatible:
- Prefer MIT, BSD-3-Clause, or Apache-2.0 licensed libraries
- Avoid GPL, LGPL dependencies
- Check transitive dependencies for license compatibility

### Core Dependencies

```toml
[dependencies]
# Linear algebra - MIT/Apache-2.0
ndarray = { version = "0.15", features = ["rayon"] }

# Parallelization - MIT/Apache-2.0
rayon = "1.7"

# Caching - MIT
lru = "0.11"

# Numeric traits - MIT/Apache-2.0
num-traits = "0.2"

# Error handling - MIT/Apache-2.0
thiserror = "1.0"

# Logging facade - MIT/Apache-2.0
log = "0.4"

# Optional: Better synchronization primitives - MIT/Apache-2.0
parking_lot = { version = "0.12", optional = true }

# Optional: Serialization - MIT/Apache-2.0
serde = { version = "1.0", features = ["derive"], optional = true }

[dev-dependencies]
# Benchmarking - MIT/Apache-2.0
criterion = "0.5"

# Property testing - MIT/Apache-2.0
proptest = "1.2"

# Float comparison - Apache-2.0
approx = "0.5"

# Test logging - MIT/Apache-2.0
env_logger = "0.10"
```

### QP Solver Options

For the quadratic programming solver, we have several MIT/BSD-compatible options:

1. **osqp** (Apache-2.0) - Recommended
   ```toml
   osqp = { version = "0.6", optional = true }
   ```

2. **clarabel** (Apache-2.0) - Pure Rust alternative
   ```toml
   clarabel = { version = "0.6", optional = true }
   ```

3. **good_lp** (MIT) - Linear programming with QP support
   ```toml
   good_lp = { version = "1.7", default-features = false, features = ["minilp"], optional = true }
   ```

Note: Avoid GPL-licensed solvers like GLPK bindings.

### License Verification

Before adding new dependencies:
```bash
# Install cargo-license
cargo install cargo-license

# Check all licenses
cargo license

# Check dependency tree with licenses
cargo tree --format "{p} {l}"
```

## Development Guidelines

### Development Environment
- Rust: Latest stable version
- Linting: clippy with recommended settings
- Testing: cargo test with coverage via cargo-tarpaulin
- CI/CD: GitHub Actions for automated testing

### Code Standards
- Language: American English for all code, comments, and commit messages
- Testing: TDD approach with 90% coverage target
- Documentation: Rough documentation during development, refined at milestone completion

### Performance Considerations
- No specific performance targets initially
- Benchmark against existing implementations in same environment
- Memory usage: Sensible automatic defaults with manual override capability

### API Design Philosophy
- Modern, idiomatic Rust interface
- Not constrained by compatibility with other implementations
- Design for future extensibility (multi-class, GPU support)

### Communication
- Avoid over-assumptions - ask when unclear
- Frequent progress updates initially, then adjust cadence
- Japanese for discussions, English for codebase

## Usage Example

```rust
use svm_rust::{SVM, SVMConfig, LinearKernel, DenseDataset};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load dataset (libsvm format or CSV)
    let dataset = DenseDataset::from_file("data.txt")?;

    // Configure SVM
    let config = SVMConfig {
        c: 1.0,
        epsilon: 0.001,
        cache_size: 100_000_000, // 100MB
        working_set_size: 2,     // SMO-style
        shrinking: true,
        n_jobs: Some(4),         // Use 4 threads
        ..Default::default()
    };

    // Train model
    let mut svm = SVM::new(LinearKernel, config);
    svm.fit(&dataset)?;

    // Get model info
    if let Some(info) = svm.model_info() {
        println!("Support vectors: {}", info.n_support_vectors);
        println!("Bias term: {}", info.bias);
    }

    // Make predictions
    let sample = dataset.get_sample(0);
    let prediction = svm.predict(&sample)?;
    println!("Prediction: {}, confidence: {}",
             prediction.label, prediction.decision_value.abs());

    Ok(())
}
```
