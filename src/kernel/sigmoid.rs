//! Sigmoid (Tanh) Kernel Implementation
//!
//! The Sigmoid kernel, also known as the Hyperbolic Tangent kernel, is a non-stationary
//! kernel that can model complex decision boundaries. Unlike RBF or polynomial kernels,
//! the Sigmoid kernel is not always positive semi-definite, which means it may not always
//! correspond to a valid inner product in some feature space.
//!
//! The Sigmoid kernel is defined as:
//! K(x, y) = tanh(γ * <x, y> + r)
//!
//! where:
//! - γ (gamma) is the scaling parameter for the dot product
//! - r (coef0) is the bias/offset parameter
//! - <x, y> is the dot product between vectors x and y
//!
//! Key characteristics:
//! - Non-stationary: kernel value depends on both magnitude and direction
//! - Bounded output: values always in range [-1, 1] due to tanh
//! - Origin-sensitive: behavior changes based on distance from origin
//! - Neural network connection: mimics activation in neural networks
//!
//! Applications:
//! - Neural network-inspired classification
//! - Problems where RBF/polynomial kernels underperform
//! - Non-linear pattern recognition
//! - Signal processing and time series analysis
//!
//! Note: The kernel may not be positive semi-definite for all parameter values.
//! Careful parameter selection is required for good performance.

use crate::core::SparseVector;
use crate::kernel::traits::Kernel;

/// Sigmoid (Hyperbolic Tangent) kernel for non-linear classification
#[derive(Debug, Clone)]
pub struct SigmoidKernel {
    /// Scaling parameter for the dot product (must be positive)
    pub gamma: f64,
    /// Bias/offset parameter (can be positive, negative, or zero)
    pub coef0: f64,
}

impl SigmoidKernel {
    /// Creates a new Sigmoid kernel with specified parameters
    ///
    /// # Arguments
    /// * `gamma` - Scaling parameter for the dot product (must be positive)
    /// * `coef0` - Bias/offset parameter
    ///
    /// # Panics
    /// Panics if gamma is not positive
    ///
    /// # Examples
    /// ```
    /// use rsvm::kernel::SigmoidKernel;
    ///
    /// // Standard sigmoid kernel
    /// let kernel = SigmoidKernel::new(0.1, -1.0);
    /// assert_eq!(kernel.gamma, 0.1);
    /// assert_eq!(kernel.coef0, -1.0);
    /// ```
    pub fn new(gamma: f64, coef0: f64) -> Self {
        if gamma <= 0.0 {
            panic!("Gamma must be positive, got: {}", gamma);
        }
        Self { gamma, coef0 }
    }

    /// Creates a sigmoid kernel with default parameters
    ///
    /// Uses gamma = 0.01 and coef0 = 0.0, which often work well in practice.
    ///
    /// # Examples
    /// ```
    /// use rsvm::kernel::SigmoidKernel;
    ///
    /// let kernel = SigmoidKernel::default_params();
    /// assert_eq!(kernel.gamma, 0.01);
    /// assert_eq!(kernel.coef0, 0.0);
    /// ```
    pub fn default_params() -> Self {
        Self::new(0.01, 0.0)
    }

    /// Creates a sigmoid kernel with neural network-inspired parameters
    ///
    /// Uses gamma = 1/n_features and coef0 = -1.0, mimicking neural network behavior.
    ///
    /// # Arguments
    /// * `n_features` - Number of features in the dataset
    ///
    /// # Examples
    /// ```
    /// use rsvm::kernel::SigmoidKernel;
    ///
    /// let kernel = SigmoidKernel::neural_network(100);
    /// assert_eq!(kernel.gamma, 0.01);
    /// assert_eq!(kernel.coef0, -1.0);
    /// ```
    pub fn neural_network(n_features: usize) -> Self {
        if n_features == 0 {
            panic!("Number of features must be positive");
        }
        Self::new(1.0 / n_features as f64, -1.0)
    }

    /// Creates a sigmoid kernel optimized for binary classification
    ///
    /// Uses gamma = 0.1 and coef0 = -1.0, often effective for binary problems.
    ///
    /// # Examples
    /// ```
    /// use rsvm::kernel::SigmoidKernel;
    ///
    /// let kernel = SigmoidKernel::for_binary_classification();
    /// assert_eq!(kernel.gamma, 0.1);
    /// assert_eq!(kernel.coef0, -1.0);
    /// ```
    pub fn for_binary_classification() -> Self {
        Self::new(0.1, -1.0)
    }

    /// Creates a sigmoid kernel with zero bias
    ///
    /// Uses specified gamma with coef0 = 0.0, creating an origin-centered kernel.
    ///
    /// # Arguments
    /// * `gamma` - Scaling parameter for the dot product
    ///
    /// # Examples
    /// ```
    /// use rsvm::kernel::SigmoidKernel;
    ///
    /// let kernel = SigmoidKernel::zero_bias(0.05);
    /// assert_eq!(kernel.gamma, 0.05);
    /// assert_eq!(kernel.coef0, 0.0);
    /// ```
    pub fn zero_bias(gamma: f64) -> Self {
        Self::new(gamma, 0.0)
    }

    /// Creates a sigmoid kernel with positive bias
    ///
    /// Uses specified gamma with coef0 = 1.0, shifting the sigmoid curve right.
    ///
    /// # Arguments
    /// * `gamma` - Scaling parameter for the dot product
    ///
    /// # Examples
    /// ```
    /// use rsvm::kernel::SigmoidKernel;
    ///
    /// let kernel = SigmoidKernel::positive_bias(0.05);
    /// assert_eq!(kernel.gamma, 0.05);
    /// assert_eq!(kernel.coef0, 1.0);
    /// ```
    pub fn positive_bias(gamma: f64) -> Self {
        Self::new(gamma, 1.0)
    }

    /// Creates a sigmoid kernel with normalized gamma
    ///
    /// Uses gamma = 1/√n_features and specified bias, providing scale-invariance.
    ///
    /// # Arguments
    /// * `n_features` - Number of features in the dataset
    /// * `coef0` - Bias/offset parameter
    ///
    /// # Examples
    /// ```
    /// use rsvm::kernel::SigmoidKernel;
    ///
    /// let kernel = SigmoidKernel::normalized(100, -0.5);
    /// assert_eq!(kernel.gamma, 0.1);
    /// assert_eq!(kernel.coef0, -0.5);
    /// ```
    pub fn normalized(n_features: usize, coef0: f64) -> Self {
        if n_features == 0 {
            panic!("Number of features must be positive");
        }
        Self::new(1.0 / (n_features as f64).sqrt(), coef0)
    }
}

impl Kernel for SigmoidKernel {
    fn compute(&self, x: &SparseVector, y: &SparseVector) -> f64 {
        // Compute dot product efficiently for sparse vectors
        let dot_product = compute_sparse_dot_product(x, y);

        // Apply sigmoid transformation: tanh(γ * <x,y> + r)
        (self.gamma * dot_product + self.coef0).tanh()
    }
}

/// Efficient dot product computation for sparse vectors
///
/// Computes <x, y> = Σᵢ xᵢ * yᵢ using optimized sparse vector traversal.
fn compute_sparse_dot_product(x: &SparseVector, y: &SparseVector) -> f64 {
    let mut dot_product = 0.0;
    let mut i = 0;
    let mut j = 0;

    let x_indices = &x.indices;
    let x_values = &x.values;
    let y_indices = &y.indices;
    let y_values = &y.values;

    // Two-pointer technique for sorted sparse vectors
    while i < x_indices.len() && j < y_indices.len() {
        if x_indices[i] == y_indices[j] {
            // Both vectors have non-zero values at this index
            dot_product += x_values[i] * y_values[j];
            i += 1;
            j += 1;
        } else if x_indices[i] < y_indices[j] {
            // Only x has non-zero value, y is implicitly 0
            i += 1;
        } else {
            // Only y has non-zero value, x is implicitly 0
            j += 1;
        }
    }

    dot_product
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_sigmoid_kernel_creation() {
        let kernel = SigmoidKernel::new(0.1, -1.0);
        assert_eq!(kernel.gamma, 0.1);
        assert_eq!(kernel.coef0, -1.0);
    }

    #[test]
    #[should_panic(expected = "Gamma must be positive")]
    fn test_invalid_gamma() {
        SigmoidKernel::new(-0.1, 0.0);
    }

    #[test]
    #[should_panic(expected = "Gamma must be positive")]
    fn test_zero_gamma() {
        SigmoidKernel::new(0.0, 0.0);
    }

    #[test]
    fn test_default_params() {
        let kernel = SigmoidKernel::default_params();
        assert_eq!(kernel.gamma, 0.01);
        assert_eq!(kernel.coef0, 0.0);
    }

    #[test]
    fn test_neural_network_params() {
        let kernel = SigmoidKernel::neural_network(100);
        assert_eq!(kernel.gamma, 0.01);
        assert_eq!(kernel.coef0, -1.0);

        let kernel2 = SigmoidKernel::neural_network(50);
        assert_eq!(kernel2.gamma, 0.02);
        assert_eq!(kernel2.coef0, -1.0);
    }

    #[test]
    #[should_panic(expected = "Number of features must be positive")]
    fn test_neural_network_zero_features() {
        SigmoidKernel::neural_network(0);
    }

    #[test]
    fn test_binary_classification_params() {
        let kernel = SigmoidKernel::for_binary_classification();
        assert_eq!(kernel.gamma, 0.1);
        assert_eq!(kernel.coef0, -1.0);
    }

    #[test]
    fn test_zero_bias() {
        let kernel = SigmoidKernel::zero_bias(0.05);
        assert_eq!(kernel.gamma, 0.05);
        assert_eq!(kernel.coef0, 0.0);
    }

    #[test]
    fn test_positive_bias() {
        let kernel = SigmoidKernel::positive_bias(0.05);
        assert_eq!(kernel.gamma, 0.05);
        assert_eq!(kernel.coef0, 1.0);
    }

    #[test]
    fn test_normalized_params() {
        let kernel = SigmoidKernel::normalized(100, -0.5);
        assert_eq!(kernel.gamma, 0.1);
        assert_eq!(kernel.coef0, -0.5);

        let kernel2 = SigmoidKernel::normalized(4, 0.0);
        assert_eq!(kernel2.gamma, 0.5);
        assert_eq!(kernel2.coef0, 0.0);
    }

    #[test]
    #[should_panic(expected = "Number of features must be positive")]
    fn test_normalized_zero_features() {
        SigmoidKernel::normalized(0, 0.0);
    }

    #[test]
    fn test_sigmoid_kernel_zero_vectors() {
        let kernel = SigmoidKernel::new(0.1, -1.0);

        let x = SparseVector::new(vec![], vec![]);
        let y = SparseVector::new(vec![], vec![]);

        // tanh(0.1 * 0 + (-1.0)) = tanh(-1.0) ≈ -0.7616
        let expected = (-1.0_f64).tanh();
        let result = kernel.compute(&x, &y);
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_sigmoid_kernel_identical_vectors() {
        let kernel = SigmoidKernel::new(0.1, -1.0);

        let x = SparseVector::new(vec![0, 1, 2], vec![1.0, 2.0, 3.0]);
        let y = SparseVector::new(vec![0, 1, 2], vec![1.0, 2.0, 3.0]);

        // Dot product = 1*1 + 2*2 + 3*3 = 1 + 4 + 9 = 14
        // tanh(0.1 * 14 + (-1.0)) = tanh(1.4 - 1.0) = tanh(0.4) ≈ 0.3799
        let expected = (0.1_f64 * 14.0 - 1.0).tanh();
        let result = kernel.compute(&x, &y);
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_sigmoid_kernel_orthogonal_vectors() {
        let kernel = SigmoidKernel::new(0.1, -1.0);

        let x = SparseVector::new(vec![0, 1], vec![1.0, 0.0]);
        let y = SparseVector::new(vec![0, 1], vec![0.0, 1.0]);

        // Dot product = 1*0 + 0*1 = 0
        // tanh(0.1 * 0 + (-1.0)) = tanh(-1.0) ≈ -0.7616
        let expected = (-1.0_f64).tanh();
        let result = kernel.compute(&x, &y);
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_sigmoid_kernel_sparse_vectors() {
        let kernel = SigmoidKernel::new(0.1, -1.0);

        let x = SparseVector::new(vec![0, 2, 5], vec![1.0, 2.0, 3.0]);
        let y = SparseVector::new(vec![1, 2, 4], vec![4.0, 5.0, 6.0]);

        // Only index 2 overlaps: 2 * 5 = 10
        // tanh(0.1 * 10 + (-1.0)) = tanh(1.0 - 1.0) = tanh(0.0) = 0.0
        let expected = 0.0;
        let result = kernel.compute(&x, &y);
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_sigmoid_kernel_no_overlap() {
        let kernel = SigmoidKernel::new(0.1, -1.0);

        let x = SparseVector::new(vec![0, 1], vec![1.0, 2.0]);
        let y = SparseVector::new(vec![2, 3], vec![3.0, 4.0]);

        // No overlap, dot product = 0
        // tanh(0.1 * 0 + (-1.0)) = tanh(-1.0) ≈ -0.7616
        let expected = (-1.0_f64).tanh();
        let result = kernel.compute(&x, &y);
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_sigmoid_kernel_positive_bias() {
        let kernel = SigmoidKernel::new(0.1, 1.0);

        let x = SparseVector::new(vec![0, 1], vec![2.0, 3.0]);
        let y = SparseVector::new(vec![0, 1], vec![1.0, 2.0]);

        // Dot product = 2*1 + 3*2 = 2 + 6 = 8
        // tanh(0.1 * 8 + 1.0) = tanh(0.8 + 1.0) = tanh(1.8) ≈ 0.9468
        let expected = (0.1_f64 * 8.0 + 1.0).tanh();
        let result = kernel.compute(&x, &y);
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_sigmoid_kernel_zero_bias() {
        let kernel = SigmoidKernel::new(0.1, 0.0);

        let x = SparseVector::new(vec![0, 1], vec![2.0, 3.0]);
        let y = SparseVector::new(vec![0, 1], vec![1.0, 2.0]);

        // Dot product = 2*1 + 3*2 = 2 + 6 = 8
        // tanh(0.1 * 8 + 0.0) = tanh(0.8) ≈ 0.6640
        let expected = (0.1_f64 * 8.0).tanh();
        let result = kernel.compute(&x, &y);
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_sigmoid_kernel_large_values() {
        let kernel = SigmoidKernel::new(0.01, -1.0);

        let x = SparseVector::new(vec![0, 1], vec![100.0, 200.0]);
        let y = SparseVector::new(vec![0, 1], vec![50.0, 100.0]);

        // Dot product = 100*50 + 200*100 = 5000 + 20000 = 25000
        // tanh(0.01 * 25000 + (-1.0)) = tanh(250 - 1) = tanh(249) ≈ 1.0
        let expected = (0.01_f64 * 25000.0 - 1.0).tanh();
        let result = kernel.compute(&x, &y);
        assert_relative_eq!(result, expected, epsilon = 1e-10);
        assert!(result > 0.99); // Should be very close to 1.0
    }

    #[test]
    fn test_sigmoid_kernel_negative_values() {
        let kernel = SigmoidKernel::new(0.1, -1.0);

        let x = SparseVector::new(vec![0, 1], vec![-2.0, 3.0]);
        let y = SparseVector::new(vec![0, 1], vec![1.0, -2.0]);

        // Dot product = (-2)*1 + 3*(-2) = -2 - 6 = -8
        // tanh(0.1 * (-8) + (-1.0)) = tanh(-0.8 - 1.0) = tanh(-1.8) ≈ -0.9468
        let expected = (0.1_f64 * (-8.0) - 1.0).tanh();
        let result = kernel.compute(&x, &y);
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_sigmoid_kernel_bounds() {
        let kernel = SigmoidKernel::new(0.1, -1.0);

        // Test multiple vectors to ensure output is always in [-1, 1]
        let vectors = vec![
            SparseVector::new(vec![0], vec![100.0]),
            SparseVector::new(vec![0], vec![-100.0]),
            SparseVector::new(vec![0, 1, 2], vec![1.0, 2.0, 3.0]),
            SparseVector::new(vec![], vec![]),
        ];

        for x in &vectors {
            for y in &vectors {
                let result = kernel.compute(x, y);
                assert!(result >= -1.0);
                assert!(result <= 1.0);
            }
        }
    }

    #[test]
    fn test_sparse_dot_product_computation() {
        // Test the helper function directly
        let x = SparseVector::new(vec![0, 2, 4], vec![1.0, 2.0, 3.0]);
        let y = SparseVector::new(vec![1, 2, 3], vec![4.0, 5.0, 6.0]);

        // Only index 2 overlaps: 2 * 5 = 10
        let result = compute_sparse_dot_product(&x, &y);
        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_sparse_dot_product_empty() {
        let x = SparseVector::new(vec![], vec![]);
        let y = SparseVector::new(vec![], vec![]);

        let result = compute_sparse_dot_product(&x, &y);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_sigmoid_kernel_symmetry() {
        let kernel = SigmoidKernel::new(0.1, -1.0);

        let x = SparseVector::new(vec![0, 1, 2], vec![1.0, 2.0, 3.0]);
        let y = SparseVector::new(vec![0, 1, 2], vec![4.0, 5.0, 6.0]);

        // Test that K(x,y) = K(y,x)
        let result_xy = kernel.compute(&x, &y);
        let result_yx = kernel.compute(&y, &x);
        assert_relative_eq!(result_xy, result_yx, epsilon = 1e-10);
    }

    #[test]
    fn test_sigmoid_kernel_parameter_effects() {
        let x = SparseVector::new(vec![0, 1], vec![1.0, 2.0]);
        let y = SparseVector::new(vec![0, 1], vec![2.0, 1.0]);

        // Dot product = 1*2 + 2*1 = 4

        // Test different gamma values
        let kernel1 = SigmoidKernel::new(0.1, 0.0);
        let kernel2 = SigmoidKernel::new(0.5, 0.0);
        let kernel3 = SigmoidKernel::new(1.0, 0.0);

        let result1 = kernel1.compute(&x, &y);
        let result2 = kernel2.compute(&x, &y);
        let result3 = kernel3.compute(&x, &y);

        // Higher gamma should lead to steeper sigmoid curve
        assert!(result1 < result2);
        assert!(result2 < result3);

        // Test different bias values
        let kernel4 = SigmoidKernel::new(0.1, -1.0);
        let kernel5 = SigmoidKernel::new(0.1, 0.0);
        let kernel6 = SigmoidKernel::new(0.1, 1.0);

        let result4 = kernel4.compute(&x, &y);
        let result5 = kernel5.compute(&x, &y);
        let result6 = kernel6.compute(&x, &y);

        // Higher bias should shift sigmoid curve right (higher values)
        assert!(result4 < result5);
        assert!(result5 < result6);
    }
}
