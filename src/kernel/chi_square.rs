//! Chi-Square Kernel Implementation
//!
//! The Chi-square kernel is particularly effective for histogram and distribution data.
//! It's widely used in computer vision, bioinformatics, and text analysis applications
//! where data naturally comes in histogram form.
//!
//! The Chi-square kernel is defined as:
//! K(x, y) = exp(-γ * χ²(x, y))
//!
//! Where χ²(x, y) = Σᵢ (xᵢ - yᵢ)² / (xᵢ + yᵢ) for xᵢ + yᵢ ≠ 0
//!
//! This kernel is particularly effective because:
//! - It naturally handles histogram data with different scales
//! - It provides excellent performance on computer vision tasks
//! - It's numerically stable for non-negative features
//! - It captures similarity between distributions effectively

use crate::core::SparseVector;
use crate::kernel::traits::Kernel;

/// Chi-square kernel optimized for histogram and distribution data
#[derive(Debug, Clone)]
pub struct ChiSquareKernel {
    /// Scaling parameter gamma (must be positive)
    pub gamma: f64,
}

impl ChiSquareKernel {
    /// Creates a new Chi-square kernel with the specified gamma parameter
    ///
    /// # Arguments
    /// * `gamma` - Scaling parameter for the chi-square distance (must be positive)
    ///
    /// # Examples
    /// ```
    /// use rsvm::kernel::ChiSquareKernel;
    ///
    /// // Standard chi-square kernel
    /// let kernel = ChiSquareKernel::new(1.0);
    ///
    /// // More sensitive to differences
    /// let sensitive_kernel = ChiSquareKernel::new(10.0);
    /// ```
    pub fn new(gamma: f64) -> Self {
        assert!(gamma > 0.0, "Gamma must be positive");

        Self { gamma }
    }

    /// Creates a chi-square kernel with unit gamma (1.0)
    ///
    /// This is often a good starting point for histogram data.
    ///
    /// # Examples
    /// ```
    /// use rsvm::kernel::ChiSquareKernel;
    ///
    /// let kernel = ChiSquareKernel::unit_gamma();
    /// assert_eq!(kernel.gamma, 1.0);
    /// ```
    pub fn unit_gamma() -> Self {
        Self::new(1.0)
    }

    /// Creates a chi-square kernel with auto-gamma based on feature count
    ///
    /// Uses gamma = 1.0 / n_features as a reasonable default for high-dimensional data.
    ///
    /// # Arguments
    /// * `n_features` - Number of features in the dataset
    ///
    /// # Examples
    /// ```
    /// use rsvm::kernel::ChiSquareKernel;
    ///
    /// // For a 128-dimensional histogram (e.g., color histogram)
    /// let kernel = ChiSquareKernel::with_auto_gamma(128);
    /// assert_eq!(kernel.gamma, 1.0 / 128.0);
    /// ```
    pub fn with_auto_gamma(n_features: usize) -> Self {
        let gamma = 1.0 / n_features as f64;
        Self::new(gamma)
    }

    /// Creates a chi-square kernel optimized for computer vision applications
    ///
    /// Uses gamma = 0.5, which has been empirically shown to work well
    /// for visual feature histograms and image recognition tasks.
    ///
    /// # Examples
    /// ```
    /// use rsvm::kernel::ChiSquareKernel;
    ///
    /// let kernel = ChiSquareKernel::for_computer_vision();
    /// assert_eq!(kernel.gamma, 0.5);
    /// ```
    pub fn for_computer_vision() -> Self {
        Self::new(0.5)
    }

    /// Creates a chi-square kernel optimized for text analysis
    ///
    /// Uses gamma = 2.0, which works well for term frequency histograms
    /// and bag-of-words representations.
    ///
    /// # Examples
    /// ```
    /// use rsvm::kernel::ChiSquareKernel;
    ///
    /// let kernel = ChiSquareKernel::for_text_analysis();
    /// assert_eq!(kernel.gamma, 2.0);
    /// ```
    pub fn for_text_analysis() -> Self {
        Self::new(2.0)
    }
}

impl Kernel for ChiSquareKernel {
    fn compute(&self, x: &SparseVector, y: &SparseVector) -> f64 {
        // Compute chi-square distance efficiently for sparse vectors
        let chi_square_distance = compute_chi_square_distance(x, y);

        // Apply exponential kernel: K(x,y) = exp(-γ * χ²(x,y))
        (-self.gamma * chi_square_distance).exp()
    }
}

/// Efficient chi-square distance computation for sparse vectors
///
/// Computes χ²(x, y) = Σᵢ (xᵢ - yᵢ)² / (xᵢ + yᵢ) for xᵢ + yᵢ ≠ 0
///
/// Uses optimized sparse vector traversal to handle only non-zero elements.
fn compute_chi_square_distance(x: &SparseVector, y: &SparseVector) -> f64 {
    let mut distance = 0.0;
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
            let x_val = x_values[i];
            let y_val = y_values[j];
            let sum = x_val + y_val;

            if sum > 0.0 {
                let diff = x_val - y_val;
                distance += (diff * diff) / sum;
            }

            i += 1;
            j += 1;
        } else if x_indices[i] < y_indices[j] {
            // Only x has non-zero value at this index
            let x_val = x_values[i];
            if x_val > 0.0 {
                // χ² contribution: (x_val - 0)² / (x_val + 0) = x_val
                distance += x_val;
            }
            i += 1;
        } else {
            // Only y has non-zero value at this index
            let y_val = y_values[j];
            if y_val > 0.0 {
                // χ² contribution: (0 - y_val)² / (0 + y_val) = y_val
                distance += y_val;
            }
            j += 1;
        }
    }

    // Handle remaining elements in x
    while i < x_indices.len() {
        let x_val = x_values[i];
        if x_val > 0.0 {
            distance += x_val;
        }
        i += 1;
    }

    // Handle remaining elements in y
    while j < y_indices.len() {
        let y_val = y_values[j];
        if y_val > 0.0 {
            distance += y_val;
        }
        j += 1;
    }

    distance
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_chi_square_kernel_creation() {
        let kernel = ChiSquareKernel::new(2.0);
        assert_eq!(kernel.gamma, 2.0);
    }

    #[test]
    fn test_unit_gamma() {
        let kernel = ChiSquareKernel::unit_gamma();
        assert_eq!(kernel.gamma, 1.0);
    }

    #[test]
    fn test_auto_gamma() {
        let kernel = ChiSquareKernel::with_auto_gamma(100);
        assert_eq!(kernel.gamma, 0.01);
    }

    #[test]
    fn test_computer_vision_gamma() {
        let kernel = ChiSquareKernel::for_computer_vision();
        assert_eq!(kernel.gamma, 0.5);
    }

    #[test]
    fn test_text_analysis_gamma() {
        let kernel = ChiSquareKernel::for_text_analysis();
        assert_eq!(kernel.gamma, 2.0);
    }

    #[test]
    fn test_chi_square_kernel_identical_vectors() {
        let kernel = ChiSquareKernel::new(1.0);

        // Identical histogram vectors
        let x = SparseVector::new(vec![0, 1, 2], vec![10.0, 20.0, 30.0]);
        let y = SparseVector::new(vec![0, 1, 2], vec![10.0, 20.0, 30.0]);

        // Chi-square distance should be 0, so kernel = exp(-γ * 0) = 1.0
        let result = kernel.compute(&x, &y);
        assert_relative_eq!(result, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_chi_square_kernel_different_vectors() {
        let kernel = ChiSquareKernel::new(1.0);

        // Different histogram vectors
        let x = SparseVector::new(vec![0, 1, 2], vec![10.0, 20.0, 30.0]);
        let y = SparseVector::new(vec![0, 1, 2], vec![15.0, 10.0, 35.0]);

        // Calculate expected chi-square distance:
        // For index 0: (10-15)² / (10+15) = 25/25 = 1.0
        // For index 1: (20-10)² / (20+10) = 100/30 = 10/3
        // For index 2: (30-35)² / (30+35) = 25/65 = 5/13
        // Total: 1.0 + 10/3 + 5/13 ≈ 4.718
        let expected_distance = 1.0 + 10.0 / 3.0 + 5.0 / 13.0;
        let expected_kernel = (-expected_distance as f64).exp();

        let result = kernel.compute(&x, &y);
        assert_relative_eq!(result, expected_kernel, epsilon = 1e-10);
    }

    #[test]
    fn test_chi_square_kernel_sparse_vectors() {
        let kernel = ChiSquareKernel::new(1.0);

        // Sparse vectors with different sparsity patterns
        let x = SparseVector::new(vec![0, 2, 5], vec![10.0, 20.0, 5.0]);
        let y = SparseVector::new(vec![1, 2, 4], vec![15.0, 25.0, 8.0]);

        // Only index 2 overlaps: (20-25)² / (20+25) = 25/45 = 5/9
        // Index 0 (x only): 10.0
        // Index 5 (x only): 5.0
        // Index 1 (y only): 15.0
        // Index 4 (y only): 8.0
        // Total: 5/9 + 10 + 5 + 15 + 8 = 38 + 5/9 ≈ 38.556
        let expected_distance = 5.0 / 9.0 + 10.0 + 5.0 + 15.0 + 8.0;
        let expected_kernel = (-expected_distance as f64).exp();

        let result = kernel.compute(&x, &y);
        assert_relative_eq!(result, expected_kernel, epsilon = 1e-10);
    }

    #[test]
    fn test_chi_square_kernel_no_overlap() {
        let kernel = ChiSquareKernel::new(1.0);

        // Completely disjoint sparse vectors
        let x = SparseVector::new(vec![0, 1], vec![10.0, 20.0]);
        let y = SparseVector::new(vec![2, 3], vec![15.0, 25.0]);

        // No overlap, so chi-square distance = sum of all values
        let expected_distance = 10.0 + 20.0 + 15.0 + 25.0;
        let expected_kernel = (-expected_distance as f64).exp();

        let result = kernel.compute(&x, &y);
        assert_relative_eq!(result, expected_kernel, epsilon = 1e-10);
    }

    #[test]
    fn test_chi_square_kernel_with_zeros() {
        let kernel = ChiSquareKernel::new(1.0);

        // One vector has zero in some positions
        let x = SparseVector::new(vec![0, 2], vec![10.0, 20.0]);
        let y = SparseVector::new(vec![0, 1, 2], vec![10.0, 15.0, 25.0]);

        // Index 0: (10-10)² / (10+10) = 0
        // Index 1: y only, contributes 15.0
        // Index 2: (20-25)² / (20+25) = 25/45 = 5/9
        let expected_distance = 0.0 + 15.0 + 5.0 / 9.0;
        let expected_kernel = (-expected_distance as f64).exp();

        let result = kernel.compute(&x, &y);
        assert_relative_eq!(result, expected_kernel, epsilon = 1e-10);
    }

    #[test]
    fn test_chi_square_distance_computation() {
        // Test the helper function directly
        let x = SparseVector::new(vec![0, 1, 2], vec![10.0, 20.0, 30.0]);
        let y = SparseVector::new(vec![0, 1, 2], vec![15.0, 10.0, 35.0]);

        // Expected: (10-15)²/(10+15) + (20-10)²/(20+10) + (30-35)²/(30+35)
        // = 25/25 + 100/30 + 25/65 = 1.0 + 10/3 + 5/13
        let expected = 1.0 + 10.0 / 3.0 + 5.0 / 13.0;

        let result = compute_chi_square_distance(&x, &y);
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_chi_square_distance_empty_vectors() {
        let x = SparseVector::new(vec![], vec![]);
        let y = SparseVector::new(vec![], vec![]);

        let result = compute_chi_square_distance(&x, &y);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_chi_square_kernel_gamma_effect() {
        let x = SparseVector::new(vec![0, 1], vec![10.0, 20.0]);
        let y = SparseVector::new(vec![0, 1], vec![15.0, 25.0]);

        let distance = compute_chi_square_distance(&x, &y);

        // Test different gamma values
        let kernel_1 = ChiSquareKernel::new(1.0);
        let kernel_2 = ChiSquareKernel::new(2.0);

        let result_1 = kernel_1.compute(&x, &y);
        let result_2 = kernel_2.compute(&x, &y);

        assert_relative_eq!(result_1, (-distance).exp(), epsilon = 1e-10);
        assert_relative_eq!(result_2, (-2.0 * distance).exp(), epsilon = 1e-10);

        // Higher gamma should give smaller kernel values for non-identical vectors
        assert!(result_2 < result_1);
    }

    #[test]
    #[should_panic(expected = "Gamma must be positive")]
    fn test_invalid_gamma() {
        ChiSquareKernel::new(-1.0);
    }

    #[test]
    #[should_panic(expected = "Gamma must be positive")]
    fn test_zero_gamma() {
        ChiSquareKernel::new(0.0);
    }
}
