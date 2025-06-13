//! Histogram Intersection Kernel Implementation
//!
//! The Histogram Intersection kernel is specifically designed for histogram-based features
//! commonly used in computer vision applications. It's particularly effective for:
//! - Image classification with color histograms
//! - Object recognition with SIFT/SURF descriptors
//! - Texture analysis with Local Binary Patterns (LBP)
//! - Visual bag-of-words representations
//! - Any application where features represent frequency counts or probabilities
//!
//! The Histogram Intersection kernel is defined as:
//! K(x, y) = Σᵢ min(xᵢ, yᵢ)
//!
//! This kernel measures the overlap between two histograms, making it intuitive and
//! highly effective for histogram data where the "intersection" represents similarity.
//!
//! Key advantages:
//! - Bounded output: K(x,y) ∈ [0, min(||x||₁, ||y||₁)]
//! - Intuitive interpretation: measures histogram overlap
//! - Excellent performance on visual recognition tasks
//! - Computationally efficient for sparse histograms
//! - No hyperparameters to tune (unlike RBF or Chi-square)

use crate::core::SparseVector;
use crate::kernel::traits::Kernel;

/// Histogram Intersection kernel optimized for computer vision and histogram data
#[derive(Debug, Clone)]
pub struct HistogramIntersectionKernel {
    /// Normalization flag: if true, normalize by minimum L1 norm
    pub normalized: bool,
}

impl HistogramIntersectionKernel {
    /// Creates a new Histogram Intersection kernel
    ///
    /// # Arguments
    /// * `normalized` - Whether to normalize the intersection by the minimum L1 norm
    ///
    /// # Examples
    /// ```
    /// use rsvm::kernel::HistogramIntersectionKernel;
    ///
    /// // Standard histogram intersection
    /// let kernel = HistogramIntersectionKernel::new(false);
    ///
    /// // Normalized intersection (gives values in [0,1])
    /// let normalized_kernel = HistogramIntersectionKernel::new(true);
    /// ```
    pub fn new(normalized: bool) -> Self {
        Self { normalized }
    }

    /// Creates a standard (non-normalized) Histogram Intersection kernel
    ///
    /// This is the most commonly used variant, measuring absolute histogram overlap.
    ///
    /// # Examples
    /// ```
    /// use rsvm::kernel::HistogramIntersectionKernel;
    ///
    /// let kernel = HistogramIntersectionKernel::standard();
    /// assert_eq!(kernel.normalized, false);
    /// ```
    pub fn standard() -> Self {
        Self::new(false)
    }

    /// Creates a normalized Histogram Intersection kernel
    ///
    /// Normalizes the intersection by min(||x||₁, ||y||₁), giving values in [0,1].
    /// This is useful when histograms have different total counts.
    ///
    /// # Examples
    /// ```
    /// use rsvm::kernel::HistogramIntersectionKernel;
    ///
    /// let kernel = HistogramIntersectionKernel::normalized();
    /// assert_eq!(kernel.normalized, true);
    /// ```
    pub fn normalized() -> Self {
        Self::new(true)
    }

    /// Creates a kernel optimized for color histograms
    ///
    /// Uses normalized intersection, which is standard practice for color histogram
    /// comparison in computer vision applications.
    ///
    /// # Examples
    /// ```
    /// use rsvm::kernel::HistogramIntersectionKernel;
    ///
    /// let kernel = HistogramIntersectionKernel::for_color_histograms();
    /// assert_eq!(kernel.normalized, true);
    /// ```
    pub fn for_color_histograms() -> Self {
        Self::normalized()
    }

    /// Creates a kernel optimized for bag-of-visual-words
    ///
    /// Uses standard (non-normalized) intersection, which preserves the absolute
    /// frequency information important in visual word histograms.
    ///
    /// # Examples
    /// ```
    /// use rsvm::kernel::HistogramIntersectionKernel;
    ///
    /// let kernel = HistogramIntersectionKernel::for_visual_words();
    /// assert_eq!(kernel.normalized, false);
    /// ```
    pub fn for_visual_words() -> Self {
        Self::standard()
    }

    /// Creates a kernel optimized for texture analysis
    ///
    /// Uses normalized intersection, suitable for Local Binary Pattern (LBP)
    /// histograms and other texture descriptors.
    ///
    /// # Examples
    /// ```
    /// use rsvm::kernel::HistogramIntersectionKernel;
    ///
    /// let kernel = HistogramIntersectionKernel::for_texture_analysis();
    /// assert_eq!(kernel.normalized, true);
    /// ```
    pub fn for_texture_analysis() -> Self {
        Self::normalized()
    }
}

impl Kernel for HistogramIntersectionKernel {
    fn compute(&self, x: &SparseVector, y: &SparseVector) -> f64 {
        // Compute histogram intersection efficiently for sparse vectors
        let intersection = compute_histogram_intersection(x, y);

        if self.normalized {
            // Normalize by minimum L1 norm
            let x_norm = compute_l1_norm(x);
            let y_norm = compute_l1_norm(y);
            let min_norm = x_norm.min(y_norm);

            if min_norm > 0.0 {
                intersection / min_norm
            } else {
                0.0
            }
        } else {
            intersection
        }
    }
}

/// Efficient histogram intersection computation for sparse vectors
///
/// Computes Σᵢ min(xᵢ, yᵢ) using optimized sparse vector traversal.
/// Only processes non-zero elements for maximum efficiency.
fn compute_histogram_intersection(x: &SparseVector, y: &SparseVector) -> f64 {
    let mut intersection = 0.0;
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
            // Add minimum of the two values to intersection
            intersection += x_values[i].min(y_values[j]);
            i += 1;
            j += 1;
        } else if x_indices[i] < y_indices[j] {
            // Only x has non-zero value, y is implicitly 0
            // min(x[i], 0) = 0, so no contribution to intersection
            i += 1;
        } else {
            // Only y has non-zero value, x is implicitly 0
            // min(0, y[j]) = 0, so no contribution to intersection
            j += 1;
        }
    }

    // Remaining elements in either vector have no overlap, so contribute 0
    intersection
}

/// Efficient L1 norm computation for sparse vectors
///
/// Computes ||x||₁ = Σᵢ |xᵢ| for sparse vectors.
/// Assumes non-negative histogram values (common in computer vision).
fn compute_l1_norm(x: &SparseVector) -> f64 {
    x.values.iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_histogram_intersection_kernel_creation() {
        let kernel = HistogramIntersectionKernel::new(false);
        assert_eq!(kernel.normalized, false);

        let kernel = HistogramIntersectionKernel::new(true);
        assert_eq!(kernel.normalized, true);
    }

    #[test]
    fn test_standard_kernel() {
        let kernel = HistogramIntersectionKernel::standard();
        assert_eq!(kernel.normalized, false);
    }

    #[test]
    fn test_normalized_kernel() {
        let kernel = HistogramIntersectionKernel::normalized();
        assert_eq!(kernel.normalized, true);
    }

    #[test]
    fn test_color_histograms_kernel() {
        let kernel = HistogramIntersectionKernel::for_color_histograms();
        assert_eq!(kernel.normalized, true);
    }

    #[test]
    fn test_visual_words_kernel() {
        let kernel = HistogramIntersectionKernel::for_visual_words();
        assert_eq!(kernel.normalized, false);
    }

    #[test]
    fn test_texture_analysis_kernel() {
        let kernel = HistogramIntersectionKernel::for_texture_analysis();
        assert_eq!(kernel.normalized, true);
    }

    #[test]
    fn test_histogram_intersection_identical_vectors() {
        let kernel = HistogramIntersectionKernel::standard();

        // Identical histogram vectors
        let x = SparseVector::new(vec![0, 1, 2], vec![10.0, 20.0, 30.0]);
        let y = SparseVector::new(vec![0, 1, 2], vec![10.0, 20.0, 30.0]);

        // Intersection should equal the sum of values (60.0)
        let result = kernel.compute(&x, &y);
        assert_relative_eq!(result, 60.0, epsilon = 1e-10);
    }

    #[test]
    fn test_histogram_intersection_different_vectors() {
        let kernel = HistogramIntersectionKernel::standard();

        // Different histogram vectors
        let x = SparseVector::new(vec![0, 1, 2], vec![10.0, 20.0, 30.0]);
        let y = SparseVector::new(vec![0, 1, 2], vec![15.0, 10.0, 25.0]);

        // Intersection = min(10,15) + min(20,10) + min(30,25) = 10 + 10 + 25 = 45
        let result = kernel.compute(&x, &y);
        assert_relative_eq!(result, 45.0, epsilon = 1e-10);
    }

    #[test]
    fn test_histogram_intersection_sparse_vectors() {
        let kernel = HistogramIntersectionKernel::standard();

        // Sparse vectors with different sparsity patterns
        let x = SparseVector::new(vec![0, 2, 5], vec![10.0, 20.0, 5.0]);
        let y = SparseVector::new(vec![1, 2, 4], vec![15.0, 25.0, 8.0]);

        // Only index 2 overlaps: min(20, 25) = 20
        let result = kernel.compute(&x, &y);
        assert_relative_eq!(result, 20.0, epsilon = 1e-10);
    }

    #[test]
    fn test_histogram_intersection_no_overlap() {
        let kernel = HistogramIntersectionKernel::standard();

        // Completely disjoint sparse vectors
        let x = SparseVector::new(vec![0, 1], vec![10.0, 20.0]);
        let y = SparseVector::new(vec![2, 3], vec![15.0, 25.0]);

        // No overlap, intersection = 0
        let result = kernel.compute(&x, &y);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_histogram_intersection_one_empty() {
        let kernel = HistogramIntersectionKernel::standard();

        let x = SparseVector::new(vec![0, 1], vec![10.0, 20.0]);
        let y = SparseVector::new(vec![], vec![]);

        // Empty vector has no overlap
        let result = kernel.compute(&x, &y);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_normalized_histogram_intersection() {
        let kernel = HistogramIntersectionKernel::normalized();

        // Different magnitude histograms
        let x = SparseVector::new(vec![0, 1, 2], vec![10.0, 20.0, 30.0]); // L1 norm = 60
        let y = SparseVector::new(vec![0, 1, 2], vec![5.0, 10.0, 15.0]); // L1 norm = 30

        // Intersection = min(10,5) + min(20,10) + min(30,15) = 5 + 10 + 15 = 30
        // Normalized = 30 / min(60, 30) = 30 / 30 = 1.0
        let result = kernel.compute(&x, &y);
        assert_relative_eq!(result, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_normalized_partial_overlap() {
        let kernel = HistogramIntersectionKernel::normalized();

        let x = SparseVector::new(vec![0, 1, 2], vec![10.0, 20.0, 30.0]); // L1 norm = 60
        let y = SparseVector::new(vec![0, 1, 2], vec![15.0, 10.0, 25.0]); // L1 norm = 50

        // Intersection = min(10,15) + min(20,10) + min(30,25) = 10 + 10 + 25 = 45
        // Normalized = 45 / min(60, 50) = 45 / 50 = 0.9
        let result = kernel.compute(&x, &y);
        assert_relative_eq!(result, 0.9, epsilon = 1e-10);
    }

    #[test]
    fn test_normalized_sparse_overlap() {
        let kernel = HistogramIntersectionKernel::normalized();

        // Sparse vectors with partial overlap
        let x = SparseVector::new(vec![0, 2, 5], vec![10.0, 20.0, 5.0]); // L1 norm = 35
        let y = SparseVector::new(vec![1, 2, 4], vec![15.0, 25.0, 8.0]); // L1 norm = 48

        // Only index 2 overlaps: min(20, 25) = 20
        // Normalized = 20 / min(35, 48) = 20 / 35 ≈ 0.5714
        let result = kernel.compute(&x, &y);
        assert_relative_eq!(result, 20.0 / 35.0, epsilon = 1e-10);
    }

    #[test]
    fn test_l1_norm_computation() {
        let x = SparseVector::new(vec![0, 2, 4], vec![10.0, 20.0, 30.0]);
        let norm = compute_l1_norm(&x);
        assert_eq!(norm, 60.0);

        let empty = SparseVector::new(vec![], vec![]);
        let empty_norm = compute_l1_norm(&empty);
        assert_eq!(empty_norm, 0.0);
    }

    #[test]
    fn test_histogram_intersection_computation() {
        // Test the helper function directly
        let x = SparseVector::new(vec![0, 1, 2], vec![10.0, 20.0, 30.0]);
        let y = SparseVector::new(vec![0, 1, 2], vec![15.0, 10.0, 25.0]);

        let result = compute_histogram_intersection(&x, &y);
        assert_relative_eq!(result, 45.0, epsilon = 1e-10); // 10 + 10 + 25
    }

    #[test]
    fn test_histogram_intersection_empty_vectors() {
        let x = SparseVector::new(vec![], vec![]);
        let y = SparseVector::new(vec![], vec![]);

        let result = compute_histogram_intersection(&x, &y);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_normalized_with_zero_norm() {
        let kernel = HistogramIntersectionKernel::normalized();

        let x = SparseVector::new(vec![], vec![]); // L1 norm = 0
        let y = SparseVector::new(vec![0], vec![10.0]); // L1 norm = 10

        // Should handle zero norm gracefully
        let result = kernel.compute(&x, &y);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_histogram_intersection_properties() {
        let kernel = HistogramIntersectionKernel::standard();

        let x = SparseVector::new(vec![0, 1, 2], vec![10.0, 20.0, 30.0]);
        let y = SparseVector::new(vec![0, 1, 2], vec![15.0, 10.0, 25.0]);

        // Test symmetry: K(x,y) = K(y,x)
        let result_xy = kernel.compute(&x, &y);
        let result_yx = kernel.compute(&y, &x);
        assert_relative_eq!(result_xy, result_yx, epsilon = 1e-10);

        // Test self-similarity for normalized kernel
        let normalized_kernel = HistogramIntersectionKernel::normalized();
        let self_sim = normalized_kernel.compute(&x, &x);
        assert_relative_eq!(self_sim, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kernel_bounds() {
        let kernel = HistogramIntersectionKernel::standard();
        let normalized_kernel = HistogramIntersectionKernel::normalized();

        let x = SparseVector::new(vec![0, 1, 2], vec![10.0, 20.0, 30.0]);
        let y = SparseVector::new(vec![0, 1, 2], vec![15.0, 10.0, 25.0]);

        // Standard kernel should be non-negative
        let result = kernel.compute(&x, &y);
        assert!(result >= 0.0);

        // Normalized kernel should be in [0, 1]
        let normalized_result = normalized_kernel.compute(&x, &y);
        assert!(normalized_result >= 0.0);
        assert!(normalized_result <= 1.0);
    }
}
