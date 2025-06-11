//! Linear kernel implementation

use crate::core::SparseVector;
use crate::kernel::Kernel;

/// Linear kernel: K(x, y) = x^T * y
///
/// This is the simplest kernel function, computing the dot product between two vectors.
/// For sparse vectors, this is computed efficiently by iterating through non-zero elements.
#[derive(Debug, Clone, Copy, Default)]
pub struct LinearKernel;

impl LinearKernel {
    /// Create a new linear kernel
    pub fn new() -> Self {
        Self
    }
}

impl Kernel for LinearKernel {
    fn compute(&self, x: &SparseVector, y: &SparseVector) -> f64 {
        dot_product_sparse(x, y)
    }
}

/// Compute dot product between two sparse vectors
///
/// Since both vectors have sorted indices, we can compute this efficiently
/// using a merge-like algorithm in O(min(nnz(x), nnz(y))) time.
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
    fn test_linear_kernel_basic() {
        let kernel = LinearKernel::new();

        let x = SparseVector::new(vec![0, 2, 4], vec![1.0, 2.0, 3.0]);
        let y = SparseVector::new(vec![1, 2, 3], vec![1.0, 2.0, 3.0]);

        // Only index 2 overlaps: 2.0 * 2.0 = 4.0
        assert_eq!(kernel.compute(&x, &y), 4.0);
    }

    #[test]
    fn test_linear_kernel_identical() {
        let kernel = LinearKernel::new();

        let x = SparseVector::new(vec![0, 1, 2], vec![1.0, 2.0, 3.0]);

        // x^T * x = 1^2 + 2^2 + 3^2 = 14
        assert_eq!(kernel.compute(&x, &x), 14.0);
    }

    #[test]
    fn test_linear_kernel_no_overlap() {
        let kernel = LinearKernel::new();

        let x = SparseVector::new(vec![0, 2], vec![1.0, 2.0]);
        let y = SparseVector::new(vec![1, 3], vec![1.0, 2.0]);

        // No overlapping indices
        assert_eq!(kernel.compute(&x, &y), 0.0);
    }

    #[test]
    fn test_dot_product_sparse() {
        let x = SparseVector::new(vec![0, 2, 5], vec![1.0, 3.0, 2.0]);
        let y = SparseVector::new(vec![2, 3, 5], vec![2.0, 1.0, 4.0]);

        // Overlapping at indices 2 and 5: 3.0 * 2.0 + 2.0 * 4.0 = 6.0 + 8.0 = 14.0
        assert_eq!(dot_product_sparse(&x, &y), 14.0);
    }

    #[test]
    fn test_dot_product_empty() {
        let x = SparseVector::empty();
        let y = SparseVector::new(vec![0, 1], vec![1.0, 2.0]);

        assert_eq!(dot_product_sparse(&x, &y), 0.0);
        assert_eq!(dot_product_sparse(&y, &x), 0.0);
    }
}
