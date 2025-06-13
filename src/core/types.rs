//! Core type definitions for SVM

/// Prediction result containing label and decision value
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Prediction {
    /// Predicted class label (+1 or -1)
    pub label: f64,
    /// Raw decision function value
    pub decision_value: f64,
}

impl Prediction {
    /// Create a new prediction
    pub fn new(label: f64, decision_value: f64) -> Self {
        Self {
            label,
            decision_value,
        }
    }

    /// Get confidence as absolute value of decision value
    pub fn confidence(&self) -> f64 {
        self.decision_value.abs()
    }
}

/// Sparse vector representation with sorted indices
#[derive(Clone, Debug, PartialEq)]
pub struct SparseVector {
    /// Sorted indices of non-zero elements
    pub indices: Vec<usize>,
    /// Values corresponding to indices
    pub values: Vec<f64>,
}

impl SparseVector {
    /// Create a new sparse vector, ensuring indices are sorted
    pub fn new(indices: Vec<usize>, values: Vec<f64>) -> Self {
        assert_eq!(
            indices.len(),
            values.len(),
            "Indices and values must have same length"
        );

        // Sort by indices
        let mut pairs: Vec<_> = indices.into_iter().zip(values).collect();
        pairs.sort_by_key(|&(idx, _)| idx);

        let (indices, values): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
        Self { indices, values }
    }

    /// Create an empty sparse vector
    pub fn empty() -> Self {
        Self {
            indices: Vec::new(),
            values: Vec::new(),
        }
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

    /// Compute L2 norm
    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }

    /// Number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Check if vector is empty
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// Training sample with features and label
#[derive(Clone, Debug)]
pub struct Sample {
    /// Feature vector (sparse representation)
    pub features: SparseVector,
    /// Class label (+1 or -1 for binary classification)
    pub label: f64,
}

impl Sample {
    /// Create a new sample
    pub fn new(features: SparseVector, label: f64) -> Self {
        Self { features, label }
    }
}

/// Result of optimization process
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Lagrange multipliers (alpha values)
    pub alpha: Vec<f64>,
    /// Bias term (b)
    pub b: f64,
    /// Indices of support vectors (where alpha > 0)
    pub support_vectors: Vec<usize>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Final objective value
    pub objective_value: f64,
}

/// Configuration for optimizer
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Regularization parameter (upper bound for alpha)
    pub c: f64,
    /// Tolerance for KKT conditions
    pub epsilon: f64,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Working set size (q in the paper, must be even)
    pub working_set_size: usize,
    /// Kernel cache size in bytes
    pub cache_size: usize,
    /// Enable shrinking heuristic
    pub shrinking: bool,
    /// Number of iterations between shrinking (h in the paper)
    pub shrinking_iterations: usize,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            c: 1.0,
            epsilon: 0.001,
            max_iterations: 10000,
            working_set_size: 2,
            cache_size: 100_000_000, // 100MB
            shrinking: true,         // Enable shrinking by default for better performance
            shrinking_iterations: 100,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_vector_creation() {
        let indices = vec![2, 0, 4];
        let values = vec![2.0, 1.0, 3.0];
        let sv = SparseVector::new(indices, values);

        // Check that indices are sorted
        assert_eq!(sv.indices, vec![0, 2, 4]);
        assert_eq!(sv.values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_sparse_vector_get() {
        let sv = SparseVector::new(vec![1, 3, 5], vec![1.0, 2.0, 3.0]);

        assert_eq!(sv.get(0), 0.0);
        assert_eq!(sv.get(1), 1.0);
        assert_eq!(sv.get(3), 2.0);
        assert_eq!(sv.get(5), 3.0);
        assert_eq!(sv.get(6), 0.0);
    }

    #[test]
    fn test_sparse_vector_norm() {
        let sv = SparseVector::new(vec![0, 1], vec![3.0, 4.0]);
        assert_eq!(sv.norm_squared(), 25.0);
        assert_eq!(sv.norm(), 5.0);
    }

    #[test]
    fn test_prediction() {
        let pred = Prediction::new(1.0, 2.5);
        assert_eq!(pred.label, 1.0);
        assert_eq!(pred.decision_value, 2.5);
        assert_eq!(pred.confidence(), 2.5);

        let neg_pred = Prediction::new(-1.0, -1.8);
        assert_eq!(neg_pred.confidence(), 1.8);
    }

    #[test]
    fn test_sample() {
        let features = SparseVector::new(vec![0, 2], vec![1.0, 3.0]);
        let sample = Sample::new(features.clone(), 1.0);
        assert_eq!(sample.label, 1.0);
        assert_eq!(sample.features, features);
    }

    #[test]
    fn test_optimizer_config_default() {
        let config = OptimizerConfig::default();
        assert_eq!(config.c, 1.0);
        assert_eq!(config.epsilon, 0.001);
        assert_eq!(config.max_iterations, 10000);
        assert_eq!(config.working_set_size, 2);
        assert_eq!(config.cache_size, 100_000_000);
        assert!(config.shrinking);
        assert_eq!(config.shrinking_iterations, 100);
    }

    #[test]
    fn test_sparse_vector_utilities() {
        let sv = SparseVector::new(vec![1, 3], vec![2.0, 4.0]);
        assert_eq!(sv.nnz(), 2);
        assert!(!sv.is_empty());

        let empty = SparseVector::empty();
        assert_eq!(empty.nnz(), 0);
        assert!(empty.is_empty());
    }

    #[test]
    #[should_panic(expected = "Indices and values must have same length")]
    fn test_sparse_vector_length_mismatch() {
        // This should panic due to length mismatch
        SparseVector::new(vec![0, 1], vec![1.0, 2.0, 3.0]);
    }
}
