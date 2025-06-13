//! RBF (Radial Basis Function) kernel implementation
//!
//! The RBF kernel is defined as: K(x, y) = exp(-γ * ||x - y||²)
//! where γ (gamma) is a hyperparameter that controls the kernel width.

use crate::core::SparseVector;
use crate::kernel::Kernel;

/// RBF (Radial Basis Function) kernel: K(x, y) = exp(-γ * ||x - y||²)
///
/// This is one of the most popular kernels for non-linear classification.
/// The gamma parameter controls the "reach" of each training example:
/// - High gamma: close points have high influence (potential overfitting)
/// - Low gamma: distant points have influence (potential underfitting)
///
/// Common gamma values:
/// - 1.0 / n_features: good default starting point
/// - 'scale': 1.0 / (n_features * X.var()) - similar to sklearn default
/// - Manual tuning based on validation performance
#[derive(Debug, Clone, Copy)]
pub struct RBFKernel {
    gamma: f64,
}

impl RBFKernel {
    /// Create a new RBF kernel with specified gamma parameter
    ///
    /// # Arguments
    /// * `gamma` - The gamma parameter (must be positive)
    ///
    /// # Panics
    /// Panics if gamma is not positive
    pub fn new(gamma: f64) -> Self {
        assert!(gamma > 0.0, "Gamma must be positive, got: {}", gamma);
        Self { gamma }
    }

    /// Create RBF kernel with gamma = 1.0 / n_features
    ///
    /// This is a common default choice that scales inversely with dimensionality.
    ///
    /// # Arguments
    /// * `n_features` - Number of features in the dataset
    pub fn with_auto_gamma(n_features: usize) -> Self {
        assert!(n_features > 0, "Number of features must be positive");
        let gamma = 1.0 / n_features as f64;
        Self::new(gamma)
    }

    /// Create RBF kernel with gamma = 1.0 (unit gamma)
    ///
    /// Useful for normalized/scaled data where feature variance is around 1.
    pub fn unit_gamma() -> Self {
        Self::new(1.0)
    }

    /// Get the gamma parameter
    pub fn gamma(&self) -> f64 {
        self.gamma
    }
}

impl Default for RBFKernel {
    /// Default RBF kernel with gamma = 1.0
    fn default() -> Self {
        Self::unit_gamma()
    }
}

impl Kernel for RBFKernel {
    fn compute(&self, x: &SparseVector, y: &SparseVector) -> f64 {
        let squared_distance = compute_squared_euclidean_distance(x, y);
        (-self.gamma * squared_distance).exp()
    }

    fn compute_with_norms(
        &self,
        x: &SparseVector,
        y: &SparseVector,
        x_norm_sq: f64,
        y_norm_sq: f64,
    ) -> f64 {
        // ||x - y||² = ||x||² + ||y||² - 2*x^T*y
        let dot_product = dot_product_sparse(x, y);
        let squared_distance = x_norm_sq + y_norm_sq - 2.0 * dot_product;

        // Ensure non-negative distance due to numerical precision
        let squared_distance = squared_distance.max(0.0);

        (-self.gamma * squared_distance).exp()
    }
}

/// Compute squared Euclidean distance between two sparse vectors
///
/// ||x - y||² = Σᵢ (xᵢ - yᵢ)²
///
/// For sparse vectors, this can be computed efficiently by considering:
/// - Indices where both vectors have non-zero values: (xᵢ - yᵢ)²
/// - Indices where only x has non-zero values: xᵢ²
/// - Indices where only y has non-zero values: yᵢ²
fn compute_squared_euclidean_distance(x: &SparseVector, y: &SparseVector) -> f64 {
    let mut distance_sq = 0.0;
    let mut i = 0;
    let mut j = 0;

    // Process overlapping and non-overlapping indices
    while i < x.indices.len() && j < y.indices.len() {
        let x_idx = x.indices[i];
        let y_idx = y.indices[j];

        if x_idx == y_idx {
            // Both vectors have values at this index
            let diff = x.values[i] - y.values[j];
            distance_sq += diff * diff;
            i += 1;
            j += 1;
        } else if x_idx < y_idx {
            // Only x has a value at this index (y is implicitly 0)
            distance_sq += x.values[i] * x.values[i];
            i += 1;
        } else {
            // Only y has a value at this index (x is implicitly 0)
            distance_sq += y.values[j] * y.values[j];
            j += 1;
        }
    }

    // Handle remaining elements
    while i < x.indices.len() {
        distance_sq += x.values[i] * x.values[i];
        i += 1;
    }

    while j < y.indices.len() {
        distance_sq += y.values[j] * y.values[j];
        j += 1;
    }

    distance_sq
}

/// Compute dot product between two sparse vectors
///
/// This is the same as the linear kernel computation, but needed here
/// for the optimized compute_with_norms implementation.
fn dot_product_sparse(x: &SparseVector, y: &SparseVector) -> f64 {
    let mut result = 0.0;
    let mut i = 0;
    let mut j = 0;

    while i < x.indices.len() && j < y.indices.len() {
        let x_idx = x.indices[i];
        let y_idx = y.indices[j];

        if x_idx == y_idx {
            result += x.values[i] * y.values[j];
            i += 1;
            j += 1;
        } else if x_idx < y_idx {
            i += 1;
        } else {
            j += 1;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rbf_kernel_creation() {
        let kernel = RBFKernel::new(0.5);
        assert_eq!(kernel.gamma(), 0.5);

        let kernel_auto = RBFKernel::with_auto_gamma(10);
        assert_eq!(kernel_auto.gamma(), 0.1);

        let kernel_unit = RBFKernel::unit_gamma();
        assert_eq!(kernel_unit.gamma(), 1.0);

        let kernel_default = RBFKernel::default();
        assert_eq!(kernel_default.gamma(), 1.0);
    }

    #[test]
    #[should_panic(expected = "Gamma must be positive")]
    fn test_rbf_kernel_invalid_gamma() {
        RBFKernel::new(-0.5);
    }

    #[test]
    #[should_panic(expected = "Gamma must be positive")]
    fn test_rbf_kernel_zero_gamma() {
        RBFKernel::new(0.0);
    }

    #[test]
    #[should_panic(expected = "Number of features must be positive")]
    fn test_rbf_kernel_zero_features() {
        RBFKernel::with_auto_gamma(0);
    }

    #[test]
    fn test_rbf_kernel_identical_vectors() {
        let kernel = RBFKernel::new(1.0);
        let x = SparseVector::new(vec![0, 1, 2], vec![1.0, 2.0, 3.0]);

        // K(x, x) should always be 1.0 for RBF kernel
        assert!((kernel.compute(&x, &x) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rbf_kernel_orthogonal_vectors() {
        let kernel = RBFKernel::new(1.0);
        let x = SparseVector::new(vec![0, 2], vec![1.0, 1.0]);
        let y = SparseVector::new(vec![1, 3], vec![1.0, 1.0]);

        // ||x - y||² = 1² + 1² + 1² + 1² = 4 (no overlap)
        // K(x, y) = exp(-1.0 * 4) = exp(-4)
        let expected = (-4.0_f64).exp();
        let result = kernel.compute(&x, &y);
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_rbf_kernel_different_gammas() {
        let x = SparseVector::new(vec![0], vec![1.0]);
        let y = SparseVector::new(vec![0], vec![3.0]);

        // ||x - y||² = (1 - 3)² = 4

        let kernel_low = RBFKernel::new(0.1);
        let kernel_high = RBFKernel::new(10.0);

        let result_low = kernel_low.compute(&x, &y);
        let result_high = kernel_high.compute(&x, &y);

        // Low gamma should give higher similarity (less sensitive to distance)
        // High gamma should give lower similarity (more sensitive to distance)
        assert!(result_low > result_high);

        // Check actual values
        assert!((result_low - (-0.1 * 4.0_f64).exp()).abs() < 1e-10);
        assert!((result_high - (-10.0 * 4.0_f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_rbf_kernel_symmetry() {
        let kernel = RBFKernel::new(0.5);
        let x = SparseVector::new(vec![0, 2, 4], vec![1.0, 2.0, 3.0]);
        let y = SparseVector::new(vec![1, 2, 3], vec![1.0, 2.0, 3.0]);

        // K(x, y) should equal K(y, x)
        assert_eq!(kernel.compute(&x, &y), kernel.compute(&y, &x));
    }

    #[test]
    fn test_rbf_kernel_with_norms() {
        let kernel = RBFKernel::new(2.0);
        let x = SparseVector::new(vec![0, 1], vec![3.0, 4.0]);
        let y = SparseVector::new(vec![0, 1], vec![1.0, 2.0]);

        let x_norm_sq = 3.0 * 3.0 + 4.0 * 4.0; // 25.0
        let y_norm_sq = 1.0 * 1.0 + 2.0 * 2.0; // 5.0

        let result1 = kernel.compute(&x, &y);
        let result2 = kernel.compute_with_norms(&x, &y, x_norm_sq, y_norm_sq);

        assert!((result1 - result2).abs() < 1e-10);
    }

    #[test]
    fn test_squared_euclidean_distance() {
        // Test with overlapping indices
        let x = SparseVector::new(vec![0, 2, 5], vec![1.0, 3.0, 2.0]);
        let y = SparseVector::new(vec![2, 3, 5], vec![2.0, 1.0, 4.0]);

        // Distance calculation:
        // Index 0: x=1, y=0 -> (1-0)² = 1
        // Index 2: x=3, y=2 -> (3-2)² = 1
        // Index 3: x=0, y=1 -> (0-1)² = 1
        // Index 5: x=2, y=4 -> (2-4)² = 4
        // Total: 1 + 1 + 1 + 4 = 7
        assert_eq!(compute_squared_euclidean_distance(&x, &y), 7.0);
    }

    #[test]
    fn test_squared_euclidean_distance_identical() {
        let x = SparseVector::new(vec![0, 1, 2], vec![1.0, 2.0, 3.0]);
        assert_eq!(compute_squared_euclidean_distance(&x, &x), 0.0);
    }

    #[test]
    fn test_squared_euclidean_distance_no_overlap() {
        let x = SparseVector::new(vec![0, 2], vec![1.0, 2.0]);
        let y = SparseVector::new(vec![1, 3], vec![1.0, 2.0]);

        // ||x - y||² = 1² + 1² + 2² + 2² = 10
        assert_eq!(compute_squared_euclidean_distance(&x, &y), 10.0);
    }

    #[test]
    fn test_squared_euclidean_distance_empty() {
        let x = SparseVector::empty();
        let y = SparseVector::new(vec![0, 1], vec![1.0, 2.0]);

        // ||empty - y||² = 1² + 2² = 5
        assert_eq!(compute_squared_euclidean_distance(&x, &y), 5.0);
        assert_eq!(compute_squared_euclidean_distance(&y, &x), 5.0);
    }

    #[test]
    fn test_dot_product_sparse() {
        let x = SparseVector::new(vec![0, 2, 5], vec![1.0, 3.0, 2.0]);
        let y = SparseVector::new(vec![2, 3, 5], vec![2.0, 1.0, 4.0]);

        // Overlapping at indices 2 and 5: 3.0 * 2.0 + 2.0 * 4.0 = 6.0 + 8.0 = 14.0
        assert_eq!(dot_product_sparse(&x, &y), 14.0);
    }

    #[test]
    fn test_rbf_kernel_convergence_properties() {
        let kernel = RBFKernel::new(1.0);

        // Test that kernel value decreases as distance increases
        let x = SparseVector::new(vec![0], vec![0.0]);
        let y1 = SparseVector::new(vec![0], vec![1.0]); // distance = 1
        let y2 = SparseVector::new(vec![0], vec![2.0]); // distance = 2
        let y3 = SparseVector::new(vec![0], vec![3.0]); // distance = 3

        let k1 = kernel.compute(&x, &y1);
        let k2 = kernel.compute(&x, &y2);
        let k3 = kernel.compute(&x, &y3);

        // Kernel values should decrease as distance increases
        assert!(k1 > k2);
        assert!(k2 > k3);

        // All kernel values should be in [0, 1]
        assert!(k1 >= 0.0 && k1 <= 1.0);
        assert!(k2 >= 0.0 && k2 <= 1.0);
        assert!(k3 >= 0.0 && k3 <= 1.0);
    }

    #[test]
    fn test_rbf_kernel_numerical_stability() {
        let kernel = RBFKernel::new(1e-6); // Very small gamma
        let x = SparseVector::new(vec![0], vec![1e6]);
        let y = SparseVector::new(vec![0], vec![-1e6]);

        // Even with large distances and small gamma, should not overflow/underflow
        let result = kernel.compute(&x, &y);
        assert!(result.is_finite());
        assert!(result >= 0.0);
        assert!(result <= 1.0);
    }
}
