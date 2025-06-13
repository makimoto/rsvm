//! Hellinger Kernel Implementation
//!
//! The Hellinger kernel is designed for probability distributions and normalized data.
//! It's particularly effective for applications where features represent probabilities,
//! frequencies, or other non-negative normalized values.
//!
//! The Hellinger kernel is defined as:
//! K(x, y) = Σᵢ √(xᵢ * yᵢ)
//!
//! This kernel measures the similarity between probability distributions using the
//! square root of the element-wise product, which is related to the Hellinger distance
//! between distributions.
//!
//! Key applications:
//! - Text mining with normalized term frequencies (TF-IDF)
//! - Bioinformatics with species abundance data
//! - Statistical analysis of probability distributions
//! - Image analysis with normalized histograms
//! - Natural language processing with word embeddings
//! - Machine learning with probability vectors
//!
//! Advantages:
//! - Excellent for probability distributions and normalized data
//! - Symmetric and positive semi-definite (valid kernel)
//! - Robust to noise in probability distributions
//! - No hyperparameters required
//! - Computationally efficient for sparse data
//! - Naturally handles normalized features

use crate::core::SparseVector;
use crate::kernel::traits::Kernel;

/// Hellinger kernel optimized for probability distributions and normalized data
#[derive(Debug, Clone)]
pub struct HellingerKernel {
    /// Whether to apply square root normalization to the result
    pub normalized: bool,
}

impl HellingerKernel {
    /// Creates a new Hellinger kernel
    ///
    /// # Arguments
    /// * `normalized` - Whether to normalize the result by √(||x||₁ * ||y||₁)
    ///
    /// # Examples
    /// ```
    /// use rsvm::kernel::HellingerKernel;
    ///
    /// // Standard Hellinger kernel
    /// let kernel = HellingerKernel::new(false);
    ///
    /// // Normalized Hellinger kernel (values in [0,1])
    /// let normalized_kernel = HellingerKernel::new(true);
    /// ```
    pub fn new(normalized: bool) -> Self {
        Self { normalized }
    }

    /// Creates a standard (non-normalized) Hellinger kernel
    ///
    /// This is the most commonly used variant for probability distributions.
    ///
    /// # Examples
    /// ```
    /// use rsvm::kernel::HellingerKernel;
    ///
    /// let kernel = HellingerKernel::standard();
    /// assert_eq!(kernel.normalized, false);
    /// ```
    pub fn standard() -> Self {
        Self::new(false)
    }

    /// Creates a normalized Hellinger kernel
    ///
    /// Normalizes the result by √(||x||₁ * ||y||₁), giving values in [0,1].
    /// This is useful when distributions have different total probabilities.
    ///
    /// # Examples
    /// ```
    /// use rsvm::kernel::HellingerKernel;
    ///
    /// let kernel = HellingerKernel::normalized();
    /// assert_eq!(kernel.normalized, true);
    /// ```
    pub fn normalized() -> Self {
        Self::new(true)
    }

    /// Creates a kernel optimized for text mining applications
    ///
    /// Uses standard Hellinger kernel, which works well with TF-IDF vectors
    /// and other normalized text representations.
    ///
    /// # Examples
    /// ```
    /// use rsvm::kernel::HellingerKernel;
    ///
    /// let kernel = HellingerKernel::for_text_mining();
    /// assert_eq!(kernel.normalized, false);
    /// ```
    pub fn for_text_mining() -> Self {
        Self::standard()
    }

    /// Creates a kernel optimized for bioinformatics applications
    ///
    /// Uses normalized Hellinger kernel, suitable for species abundance data
    /// and other biological frequency measurements.
    ///
    /// # Examples
    /// ```
    /// use rsvm::kernel::HellingerKernel;
    ///
    /// let kernel = HellingerKernel::for_bioinformatics();
    /// assert_eq!(kernel.normalized, true);
    /// ```
    pub fn for_bioinformatics() -> Self {
        Self::normalized()
    }

    /// Creates a kernel optimized for probability vectors
    ///
    /// Uses standard Hellinger kernel, perfect for comparing probability
    /// distributions, mixture models, and other probabilistic representations.
    ///
    /// # Examples
    /// ```
    /// use rsvm::kernel::HellingerKernel;
    ///
    /// let kernel = HellingerKernel::for_probability_vectors();
    /// assert_eq!(kernel.normalized, false);
    /// ```
    pub fn for_probability_vectors() -> Self {
        Self::standard()
    }

    /// Creates a kernel optimized for statistical analysis
    ///
    /// Uses normalized Hellinger kernel, providing bounded similarity measures
    /// suitable for statistical hypothesis testing and distribution comparison.
    ///
    /// # Examples
    /// ```
    /// use rsvm::kernel::HellingerKernel;
    ///
    /// let kernel = HellingerKernel::for_statistical_analysis();
    /// assert_eq!(kernel.normalized, true);
    /// ```
    pub fn for_statistical_analysis() -> Self {
        Self::normalized()
    }
}

impl Kernel for HellingerKernel {
    fn compute(&self, x: &SparseVector, y: &SparseVector) -> f64 {
        // Compute Hellinger kernel efficiently for sparse vectors
        let hellinger_sum = compute_hellinger_sum(x, y);

        if self.normalized {
            // Normalize by √(||x||₁ * ||y||₁)
            let x_norm = compute_l1_norm(x);
            let y_norm = compute_l1_norm(y);
            let normalization = (x_norm * y_norm).sqrt();

            if normalization > 0.0 {
                hellinger_sum / normalization
            } else {
                0.0
            }
        } else {
            hellinger_sum
        }
    }
}

/// Efficient Hellinger sum computation for sparse vectors
///
/// Computes Σᵢ √(xᵢ * yᵢ) using optimized sparse vector traversal.
/// Only processes overlapping non-zero elements for maximum efficiency.
fn compute_hellinger_sum(x: &SparseVector, y: &SparseVector) -> f64 {
    let mut sum = 0.0;
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
            let product = x_values[i] * y_values[j];
            if product > 0.0 {
                // Only compute sqrt for positive products
                // (negative values would be invalid for probability distributions)
                sum += product.sqrt();
            }
            i += 1;
            j += 1;
        } else if x_indices[i] < y_indices[j] {
            // Only x has non-zero value, y is implicitly 0
            // √(x[i] * 0) = 0, so no contribution
            i += 1;
        } else {
            // Only y has non-zero value, x is implicitly 0
            // √(0 * y[j]) = 0, so no contribution
            j += 1;
        }
    }

    // Remaining elements in either vector have no overlap, so contribute 0
    sum
}

/// Efficient L1 norm computation for sparse vectors
///
/// Computes ||x||₁ = Σᵢ |xᵢ| for sparse vectors.
/// Assumes non-negative values (common for probability distributions).
fn compute_l1_norm(x: &SparseVector) -> f64 {
    x.values.iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_hellinger_kernel_creation() {
        let kernel = HellingerKernel::new(false);
        assert_eq!(kernel.normalized, false);

        let kernel = HellingerKernel::new(true);
        assert_eq!(kernel.normalized, true);
    }

    #[test]
    fn test_standard_kernel() {
        let kernel = HellingerKernel::standard();
        assert_eq!(kernel.normalized, false);
    }

    #[test]
    fn test_normalized_kernel() {
        let kernel = HellingerKernel::normalized();
        assert_eq!(kernel.normalized, true);
    }

    #[test]
    fn test_text_mining_kernel() {
        let kernel = HellingerKernel::for_text_mining();
        assert_eq!(kernel.normalized, false);
    }

    #[test]
    fn test_bioinformatics_kernel() {
        let kernel = HellingerKernel::for_bioinformatics();
        assert_eq!(kernel.normalized, true);
    }

    #[test]
    fn test_probability_vectors_kernel() {
        let kernel = HellingerKernel::for_probability_vectors();
        assert_eq!(kernel.normalized, false);
    }

    #[test]
    fn test_statistical_analysis_kernel() {
        let kernel = HellingerKernel::for_statistical_analysis();
        assert_eq!(kernel.normalized, true);
    }

    #[test]
    fn test_hellinger_kernel_identical_vectors() {
        let kernel = HellingerKernel::standard();

        // Identical probability distributions
        let x = SparseVector::new(vec![0, 1, 2], vec![0.3, 0.5, 0.2]);
        let y = SparseVector::new(vec![0, 1, 2], vec![0.3, 0.5, 0.2]);

        // Hellinger sum = √(0.3*0.3) + √(0.5*0.5) + √(0.2*0.2) = 0.3 + 0.5 + 0.2 = 1.0
        let result = kernel.compute(&x, &y);
        assert_relative_eq!(result, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hellinger_kernel_different_distributions() {
        let kernel = HellingerKernel::standard();

        // Different probability distributions
        let x = SparseVector::new(vec![0, 1, 2], vec![0.4, 0.4, 0.2]);
        let y = SparseVector::new(vec![0, 1, 2], vec![0.2, 0.3, 0.5]);

        // Hellinger sum = √(0.4*0.2) + √(0.4*0.3) + √(0.2*0.5)
        //               = √0.08 + √0.12 + √0.1
        //               ≈ 0.283 + 0.346 + 0.316 ≈ 0.945
        let expected = (0.4_f64 * 0.2).sqrt() + (0.4_f64 * 0.3).sqrt() + (0.2_f64 * 0.5).sqrt();
        let result = kernel.compute(&x, &y);
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_hellinger_kernel_sparse_vectors() {
        let kernel = HellingerKernel::standard();

        // Sparse vectors with different sparsity patterns
        let x = SparseVector::new(vec![0, 2, 5], vec![0.3, 0.4, 0.3]);
        let y = SparseVector::new(vec![1, 2, 4], vec![0.2, 0.5, 0.3]);

        // Only index 2 overlaps: √(0.4 * 0.5) = √0.2 ≈ 0.447
        let expected = (0.4_f64 * 0.5).sqrt();
        let result = kernel.compute(&x, &y);
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_hellinger_kernel_no_overlap() {
        let kernel = HellingerKernel::standard();

        // Completely disjoint sparse vectors
        let x = SparseVector::new(vec![0, 1], vec![0.6, 0.4]);
        let y = SparseVector::new(vec![2, 3], vec![0.3, 0.7]);

        // No overlap, so Hellinger sum = 0
        let result = kernel.compute(&x, &y);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_hellinger_kernel_one_empty() {
        let kernel = HellingerKernel::standard();

        let x = SparseVector::new(vec![0, 1], vec![0.6, 0.4]);
        let y = SparseVector::new(vec![], vec![]);

        // Empty vector has no overlap
        let result = kernel.compute(&x, &y);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_normalized_hellinger_kernel() {
        let kernel = HellingerKernel::normalized();

        // Different magnitude distributions
        let x = SparseVector::new(vec![0, 1, 2], vec![0.6, 0.8, 0.6]); // L1 norm = 2.0
        let y = SparseVector::new(vec![0, 1, 2], vec![0.3, 0.4, 0.3]); // L1 norm = 1.0

        // Hellinger sum = √(0.6*0.3) + √(0.8*0.4) + √(0.6*0.3)
        //               = √0.18 + √0.32 + √0.18
        //               = 2*√0.18 + √0.32
        let hellinger_sum = 2.0 * (0.18_f64).sqrt() + (0.32_f64).sqrt();

        // Normalized = hellinger_sum / √(2.0 * 1.0) = hellinger_sum / √2
        let expected = hellinger_sum / (2.0_f64).sqrt();

        let result = kernel.compute(&x, &y);
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_normalized_identical_distributions() {
        let kernel = HellingerKernel::normalized();

        // Identical distributions should give 1.0 when normalized
        let x = SparseVector::new(vec![0, 1, 2], vec![0.3, 0.5, 0.2]);
        let y = SparseVector::new(vec![0, 1, 2], vec![0.3, 0.5, 0.2]);

        let result = kernel.compute(&x, &y);
        assert_relative_eq!(result, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hellinger_sum_computation() {
        // Test the helper function directly
        let x = SparseVector::new(vec![0, 1, 2], vec![0.4, 0.3, 0.3]);
        let y = SparseVector::new(vec![0, 1, 2], vec![0.2, 0.5, 0.3]);

        let expected = (0.4_f64 * 0.2).sqrt() + (0.3_f64 * 0.5).sqrt() + (0.3_f64 * 0.3).sqrt();
        let result = compute_hellinger_sum(&x, &y);
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_hellinger_sum_empty_vectors() {
        let x = SparseVector::new(vec![], vec![]);
        let y = SparseVector::new(vec![], vec![]);

        let result = compute_hellinger_sum(&x, &y);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_l1_norm_computation() {
        let x = SparseVector::new(vec![0, 2, 4], vec![0.3, 0.5, 0.2]);
        let norm = compute_l1_norm(&x);
        assert_eq!(norm, 1.0);

        let empty = SparseVector::new(vec![], vec![]);
        let empty_norm = compute_l1_norm(&empty);
        assert_eq!(empty_norm, 0.0);
    }

    #[test]
    fn test_normalized_with_zero_norm() {
        let kernel = HellingerKernel::normalized();

        let x = SparseVector::new(vec![], vec![]); // L1 norm = 0
        let y = SparseVector::new(vec![0], vec![0.5]); // L1 norm = 0.5

        // Should handle zero norm gracefully
        let result = kernel.compute(&x, &y);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_hellinger_kernel_properties() {
        let kernel = HellingerKernel::standard();

        let x = SparseVector::new(vec![0, 1, 2], vec![0.4, 0.3, 0.3]);
        let y = SparseVector::new(vec![0, 1, 2], vec![0.2, 0.5, 0.3]);

        // Test symmetry: K(x,y) = K(y,x)
        let result_xy = kernel.compute(&x, &y);
        let result_yx = kernel.compute(&y, &x);
        assert_relative_eq!(result_xy, result_yx, epsilon = 1e-10);

        // Test self-similarity for normalized kernel
        let normalized_kernel = HellingerKernel::normalized();
        let self_sim = normalized_kernel.compute(&x, &x);
        assert_relative_eq!(self_sim, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kernel_bounds() {
        let kernel = HellingerKernel::standard();
        let normalized_kernel = HellingerKernel::normalized();

        let x = SparseVector::new(vec![0, 1, 2], vec![0.4, 0.3, 0.3]);
        let y = SparseVector::new(vec![0, 1, 2], vec![0.2, 0.5, 0.3]);

        // Standard kernel should be non-negative
        let result = kernel.compute(&x, &y);
        assert!(result >= 0.0);

        // Normalized kernel should be in [0, 1]
        let normalized_result = normalized_kernel.compute(&x, &y);
        assert!(normalized_result >= 0.0);
        assert!(normalized_result <= 1.0);
    }

    #[test]
    fn test_hellinger_with_negative_values() {
        let kernel = HellingerKernel::standard();

        // Test with one negative value (invalid for probability distributions)
        let x = SparseVector::new(vec![0, 1, 2], vec![0.3, -0.1, 0.4]);
        let y = SparseVector::new(vec![0, 1, 2], vec![0.2, 0.3, 0.5]);

        // Should handle negative values gracefully (skip negative products)
        // Only indices 0 and 2 contribute: √(0.3*0.2) + √(0.4*0.5)
        let expected = (0.3_f64 * 0.2).sqrt() + (0.4_f64 * 0.5).sqrt();
        let result = kernel.compute(&x, &y);
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_hellinger_probability_distributions() {
        let kernel = HellingerKernel::standard();

        // Test with actual probability distributions (sum to 1.0)
        let x = SparseVector::new(vec![0, 1, 2, 3], vec![0.25, 0.25, 0.25, 0.25]); // Uniform
        let y = SparseVector::new(vec![0, 1, 2, 3], vec![0.1, 0.2, 0.3, 0.4]); // Skewed

        // Hellinger sum = √(0.25*0.1) + √(0.25*0.2) + √(0.25*0.3) + √(0.25*0.4)
        let expected = (0.25_f64 * 0.1).sqrt()
            + (0.25_f64 * 0.2).sqrt()
            + (0.25_f64 * 0.3).sqrt()
            + (0.25_f64 * 0.4).sqrt();
        let result = kernel.compute(&x, &y);
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_hellinger_text_mining_scenario() {
        let kernel = HellingerKernel::for_text_mining();

        // Simulate TF-IDF vectors for two documents
        let doc1 = SparseVector::new(vec![0, 1, 2, 3, 4], vec![0.3, 0.4, 0.0, 0.2, 0.1]); // "machine learning"
        let doc2 = SparseVector::new(vec![0, 1, 2, 5, 6], vec![0.2, 0.3, 0.0, 0.3, 0.2]); // "machine intelligence"

        // Should compute overlap between common terms
        let result = kernel.compute(&doc1, &doc2);

        // Only indices 0 and 1 overlap: √(0.3*0.2) + √(0.4*0.3)
        let expected = (0.3_f64 * 0.2).sqrt() + (0.4_f64 * 0.3).sqrt();
        assert_relative_eq!(result, expected, epsilon = 1e-10);
        assert!(result > 0.0); // Should have similarity due to common terms
    }
}
