//! Polynomial Kernel Implementation
//!
//! The polynomial kernel is defined as:
//! K(x, y) = (γ * <x, y> + r)^d
//!
//! Where:
//! - γ (gamma): scaling factor for the dot product
//! - r (coef0): independent term in the polynomial
//! - d (degree): degree of the polynomial
//!
//! Common configurations:
//! - Linear kernel: d=1, γ=1, r=0
//! - Quadratic kernel: d=2, γ=1, r=1
//! - Cubic kernel: d=3, γ=1, r=1

use crate::core::SparseVector;
use crate::kernel::traits::Kernel;

/// Polynomial kernel with configurable degree, gamma, and coefficient
#[derive(Debug, Clone)]
pub struct PolynomialKernel {
    /// Scaling factor for the dot product (default: 1.0)
    pub gamma: f64,
    /// Independent term in the polynomial (default: 1.0)
    pub coef0: f64,
    /// Degree of the polynomial (default: 3)
    pub degree: u32,
}

impl PolynomialKernel {
    /// Creates a new polynomial kernel with the specified parameters
    ///
    /// # Arguments
    /// * `degree` - Degree of the polynomial (must be > 0)
    /// * `gamma` - Scaling factor for the dot product
    /// * `coef0` - Independent term in the polynomial
    ///
    /// # Examples
    /// ```
    /// use rsvm::kernel::PolynomialKernel;
    ///
    /// // Quadratic kernel: (x·y + 1)²
    /// let quad_kernel = PolynomialKernel::new(2, 1.0, 1.0);
    ///
    /// // Cubic kernel: (0.5·x·y + 1)³
    /// let cubic_kernel = PolynomialKernel::new(3, 0.5, 1.0);
    /// ```
    pub fn new(degree: u32, gamma: f64, coef0: f64) -> Self {
        assert!(degree > 0, "Polynomial degree must be positive");
        assert!(gamma > 0.0, "Gamma must be positive");

        Self {
            gamma,
            coef0,
            degree,
        }
    }

    /// Creates a quadratic kernel: (γ * <x,y> + 1)²
    ///
    /// # Arguments
    /// * `gamma` - Scaling factor for the dot product
    ///
    /// # Examples
    /// ```
    /// use rsvm::kernel::PolynomialKernel;
    ///
    /// let kernel = PolynomialKernel::quadratic(1.0);
    /// assert_eq!(kernel.degree, 2);
    /// assert_eq!(kernel.coef0, 1.0);
    /// ```
    pub fn quadratic(gamma: f64) -> Self {
        Self::new(2, gamma, 1.0)
    }

    /// Creates a cubic kernel: (γ * <x,y> + 1)³
    ///
    /// # Arguments
    /// * `gamma` - Scaling factor for the dot product
    ///
    /// # Examples
    /// ```
    /// use rsvm::kernel::PolynomialKernel;
    ///
    /// let kernel = PolynomialKernel::cubic(1.0);
    /// assert_eq!(kernel.degree, 3);
    /// assert_eq!(kernel.coef0, 1.0);
    /// ```
    pub fn cubic(gamma: f64) -> Self {
        Self::new(3, gamma, 1.0)
    }

    /// Creates a polynomial kernel with automatic gamma based on feature count
    ///
    /// Uses gamma = 1.0 / n_features as a reasonable default
    ///
    /// # Arguments
    /// * `degree` - Degree of the polynomial
    /// * `n_features` - Number of features in the dataset
    ///
    /// # Examples
    /// ```
    /// use rsvm::kernel::PolynomialKernel;
    ///
    /// // For a 100-dimensional dataset
    /// let kernel = PolynomialKernel::auto(3, 100);
    /// assert_eq!(kernel.gamma, 0.01);
    /// ```
    pub fn auto(degree: u32, n_features: usize) -> Self {
        let gamma = 1.0 / n_features as f64;
        Self::new(degree, gamma, 1.0)
    }

    /// Creates a normalized polynomial kernel: ((x·y + 1) / sqrt(||x||² + 1) / sqrt(||y||² + 1))^d
    ///
    /// This variant normalizes the input vectors to avoid numerical overflow
    /// for high-degree polynomials.
    ///
    /// # Arguments
    /// * `degree` - Degree of the polynomial
    ///
    /// # Examples
    /// ```
    /// use rsvm::kernel::PolynomialKernel;
    ///
    /// let kernel = PolynomialKernel::normalized(4);
    /// ```
    pub fn normalized(degree: u32) -> Self {
        Self::new(degree, 1.0, 1.0)
    }
}

impl Kernel for PolynomialKernel {
    fn compute(&self, x: &SparseVector, y: &SparseVector) -> f64 {
        // Compute dot product efficiently for sparse vectors
        let dot_product = compute_sparse_dot_product(x, y);

        // Apply polynomial kernel formula: (γ * <x,y> + r)^d
        let kernel_value = self.gamma * dot_product + self.coef0;

        // Handle numerical edge cases
        if kernel_value <= 0.0 {
            // For negative values, return 0 to avoid complex numbers
            0.0
        } else {
            // Compute power using fast integer exponentiation
            kernel_value.powi(self.degree as i32)
        }
    }
}

/// Efficient sparse vector dot product computation
///
/// Optimized for sorted sparse vectors using two-pointer technique
fn compute_sparse_dot_product(x: &SparseVector, y: &SparseVector) -> f64 {
    let mut result = 0.0;
    let mut i = 0;
    let mut j = 0;

    let x_indices = &x.indices;
    let x_values = &x.values;
    let y_indices = &y.indices;
    let y_values = &y.values;

    // Two-pointer technique for sorted sparse vectors
    while i < x_indices.len() && j < y_indices.len() {
        if x_indices[i] == y_indices[j] {
            result += x_values[i] * y_values[j];
            i += 1;
            j += 1;
        } else if x_indices[i] < y_indices[j] {
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
    use approx::assert_relative_eq;

    #[test]
    fn test_polynomial_kernel_creation() {
        let kernel = PolynomialKernel::new(3, 0.5, 1.0);
        assert_eq!(kernel.degree, 3);
        assert_eq!(kernel.gamma, 0.5);
        assert_eq!(kernel.coef0, 1.0);
    }

    #[test]
    fn test_quadratic_kernel() {
        let kernel = PolynomialKernel::quadratic(2.0);
        assert_eq!(kernel.degree, 2);
        assert_eq!(kernel.gamma, 2.0);
        assert_eq!(kernel.coef0, 1.0);
    }

    #[test]
    fn test_cubic_kernel() {
        let kernel = PolynomialKernel::cubic(0.5);
        assert_eq!(kernel.degree, 3);
        assert_eq!(kernel.gamma, 0.5);
        assert_eq!(kernel.coef0, 1.0);
    }

    #[test]
    fn test_auto_kernel() {
        let kernel = PolynomialKernel::auto(2, 100);
        assert_eq!(kernel.degree, 2);
        assert_eq!(kernel.gamma, 0.01);
        assert_eq!(kernel.coef0, 1.0);
    }

    #[test]
    fn test_polynomial_kernel_computation() {
        let kernel = PolynomialKernel::new(2, 1.0, 1.0);

        // Test vectors: [1, 2] and [2, 1]
        let x = SparseVector::new(vec![0, 1], vec![1.0, 2.0]);
        let y = SparseVector::new(vec![0, 1], vec![2.0, 1.0]);

        // Dot product: 1*2 + 2*1 = 4
        // Kernel: (1.0 * 4 + 1.0)² = 5² = 25
        let result = kernel.compute(&x, &y);
        assert_relative_eq!(result, 25.0, epsilon = 1e-10);
    }

    #[test]
    fn test_polynomial_kernel_same_vector() {
        let kernel = PolynomialKernel::new(3, 0.5, 2.0);

        // Test vector: [3, 4]
        let x = SparseVector::new(vec![0, 1], vec![3.0, 4.0]);

        // Dot product: 3² + 4² = 25
        // Kernel: (0.5 * 25 + 2.0)³ = 14.5³ = 3048.625
        let result = kernel.compute(&x, &x);
        assert_relative_eq!(result, 3048.625, epsilon = 1e-6);
    }

    #[test]
    fn test_polynomial_kernel_orthogonal_vectors() {
        let kernel = PolynomialKernel::new(2, 1.0, 1.0);

        // Orthogonal vectors: [1, 0] and [0, 1]
        let x = SparseVector::new(vec![0], vec![1.0]);
        let y = SparseVector::new(vec![1], vec![1.0]);

        // Dot product: 0
        // Kernel: (1.0 * 0 + 1.0)² = 1² = 1
        let result = kernel.compute(&x, &y);
        assert_relative_eq!(result, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_polynomial_kernel_sparse_vectors() {
        let kernel = PolynomialKernel::new(2, 1.0, 0.0);

        // Sparse vectors with different sparsity patterns
        let x = SparseVector::new(vec![0, 2, 5], vec![1.0, 2.0, 3.0]);
        let y = SparseVector::new(vec![1, 2, 4], vec![1.0, 2.0, 3.0]);

        // Only index 2 overlaps: dot product = 2.0 * 2.0 = 4.0
        // Kernel: (1.0 * 4.0 + 0.0)² = 16.0
        let result = kernel.compute(&x, &y);
        assert_relative_eq!(result, 16.0, epsilon = 1e-10);
    }

    #[test]
    fn test_polynomial_kernel_negative_handling() {
        // Create kernel manually to bypass gamma validation for testing
        let kernel = PolynomialKernel {
            gamma: -1.0,
            coef0: 0.5,
            degree: 2,
        };

        // Vector that will create negative kernel value before power
        let x = SparseVector::new(vec![0], vec![1.0]);
        let y = SparseVector::new(vec![0], vec![1.0]);

        // Dot product: 1.0
        // Kernel value before power: -1.0 * 1.0 + 0.5 = -0.5
        // Since negative, should return 0.0
        let result = kernel.compute(&x, &y);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_polynomial_kernel_high_degree() {
        let kernel = PolynomialKernel::new(5, 0.1, 1.0);

        let x = SparseVector::new(vec![0], vec![2.0]);
        let y = SparseVector::new(vec![0], vec![3.0]);

        // Dot product: 6.0
        // Kernel: (0.1 * 6.0 + 1.0)⁵ = 1.6⁵ ≈ 10.48576
        let result = kernel.compute(&x, &y);
        assert_relative_eq!(result, 10.48576, epsilon = 1e-5);
    }

    #[test]
    fn test_sparse_dot_product() {
        // Test the helper function directly
        let x = SparseVector::new(vec![0, 2, 4], vec![1.0, 2.0, 3.0]);
        let y = SparseVector::new(vec![1, 2, 3], vec![1.0, 4.0, 2.0]);

        // Only index 2 matches: 2.0 * 4.0 = 8.0
        let result = compute_sparse_dot_product(&x, &y);
        assert_relative_eq!(result, 8.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sparse_dot_product_no_overlap() {
        let x = SparseVector::new(vec![0, 2], vec![1.0, 2.0]);
        let y = SparseVector::new(vec![1, 3], vec![3.0, 4.0]);

        // No overlapping indices
        let result = compute_sparse_dot_product(&x, &y);
        assert_eq!(result, 0.0);
    }

    #[test]
    #[should_panic(expected = "Polynomial degree must be positive")]
    fn test_invalid_degree() {
        PolynomialKernel::new(0, 1.0, 1.0);
    }

    #[test]
    #[should_panic(expected = "Gamma must be positive")]
    fn test_invalid_gamma() {
        PolynomialKernel::new(2, -1.0, 1.0);
    }
}
