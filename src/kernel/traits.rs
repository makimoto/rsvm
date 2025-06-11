//! Kernel trait definition

use crate::core::SparseVector;

/// Kernel function trait
///
/// A kernel function K(x, y) must satisfy Mercer's condition to be valid for SVM.
/// This trait provides the interface for different kernel implementations.
pub trait Kernel: Send + Sync {
    /// Compute kernel value K(x, y)
    fn compute(&self, x: &SparseVector, y: &SparseVector) -> f64;

    /// Optional: compute kernel value using precomputed squared norms
    /// This can be more efficient for some kernels (e.g., RBF)
    fn compute_with_norms(
        &self,
        x: &SparseVector,
        y: &SparseVector,
        x_norm_sq: f64,
        y_norm_sq: f64,
    ) -> f64 {
        // Default implementation ignores the norms
        let _ = (x_norm_sq, y_norm_sq);
        self.compute(x, y)
    }
}
