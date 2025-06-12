//! Utility functions for SVM operations

use crate::core::{Dataset, Sample};

/// Validation and preprocessing utilities
pub mod validation {
    use super::*;

    /// Validate that all labels in a dataset are binary (-1 or +1)
    pub fn validate_binary_labels<D: Dataset>(dataset: &D) -> Result<(), String> {
        let labels = dataset.get_labels();
        for (i, &label) in labels.iter().enumerate() {
            if label != 1.0 && label != -1.0 {
                return Err(format!(
                    "Invalid label {} at index {}: labels must be +1 or -1",
                    label, i
                ));
            }
        }
        Ok(())
    }

    /// Check if dataset labels are balanced (roughly equal +1 and -1 samples)
    pub fn check_label_balance<D: Dataset>(dataset: &D) -> (usize, usize, f64) {
        let labels = dataset.get_labels();
        let positive_count = labels.iter().filter(|&&l| l > 0.0).count();
        let negative_count = labels.len() - positive_count;
        let balance_ratio = if negative_count == 0 {
            f64::INFINITY
        } else {
            positive_count as f64 / negative_count as f64
        };
        (positive_count, negative_count, balance_ratio)
    }

    /// Validate that feature indices are reasonable (not too sparse)
    pub fn validate_feature_sparsity(samples: &[Sample]) -> (usize, f64) {
        if samples.is_empty() {
            return (0, 0.0);
        }

        let max_dim = samples
            .iter()
            .flat_map(|s| s.features.indices.iter())
            .max()
            .map(|&i| i + 1)
            .unwrap_or(0);

        let total_features: usize = samples.iter().map(|s| s.features.indices.len()).sum();

        let avg_sparsity = if max_dim > 0 {
            total_features as f64 / (samples.len() * max_dim) as f64
        } else {
            0.0
        };

        (max_dim, avg_sparsity)
    }
}

/// Statistical utilities for datasets and models
pub mod stats {
    use super::*;
    use std::collections::HashMap;

    /// Calculate basic statistics for sparse vectors in a dataset
    pub fn sparse_vector_stats(samples: &[Sample]) -> SparseVectorStats {
        if samples.is_empty() {
            return SparseVectorStats::default();
        }

        let nnz_values: Vec<usize> = samples.iter().map(|s| s.features.nnz()).collect();

        let total_nnz: usize = nnz_values.iter().sum();
        let mean_nnz = total_nnz as f64 / samples.len() as f64;

        let max_nnz = *nnz_values.iter().max().unwrap_or(&0);
        let min_nnz = *nnz_values.iter().min().unwrap_or(&0);

        // Calculate variance
        let variance = if samples.len() > 1 {
            nnz_values
                .iter()
                .map(|&x| (x as f64 - mean_nnz).powi(2))
                .sum::<f64>()
                / (samples.len() - 1) as f64
        } else {
            0.0
        };

        SparseVectorStats {
            mean_nnz,
            min_nnz,
            max_nnz,
            variance_nnz: variance,
            total_samples: samples.len(),
        }
    }

    /// Feature frequency analysis
    pub fn feature_frequency(samples: &[Sample]) -> HashMap<usize, usize> {
        let mut frequency = HashMap::new();
        for sample in samples {
            for &index in &sample.features.indices {
                *frequency.entry(index).or_insert(0) += 1;
            }
        }
        frequency
    }
}

/// Parallel processing utilities
pub mod parallel {
    use super::*;

    /// Compute kernel matrix row in parallel
    pub fn compute_kernel_row<K: crate::kernel::Kernel + Sync>(
        kernel: &K,
        samples: &[Sample],
        row_index: usize,
        start_col: usize,
    ) -> Vec<f64> {
        let sample_i = &samples[row_index];

        samples[start_col..]
            .iter()
            .map(|sample_j| kernel.compute(&sample_i.features, &sample_j.features))
            .collect()
    }

    /// Batch prediction with parallel processing
    pub fn predict_batch<M: crate::core::SVMModel + Sync>(
        model: &M,
        samples: &[Sample],
    ) -> Vec<crate::core::Prediction> {
        samples
            .iter()
            .map(|sample| model.predict(sample))
            .collect()
    }
}

/// Memory management utilities
pub mod memory {
    /// Estimate memory usage for kernel cache
    pub fn estimate_kernel_cache_memory(n_samples: usize) -> usize {
        // Upper bound: full kernel matrix storage
        // Each entry: 8 bytes (f64) + key overhead (16 bytes) ≈ 24 bytes
        let max_entries = (n_samples * (n_samples + 1)) / 2; // Symmetric matrix
        max_entries * 24
    }

    /// Recommend cache size based on available memory and dataset size
    pub fn recommend_cache_size(n_samples: usize, available_memory_mb: usize) -> usize {
        let available_bytes = available_memory_mb * 1024 * 1024;
        let full_cache_size = estimate_kernel_cache_memory(n_samples);

        // Use at most 50% of available memory for cache
        let max_cache_size = available_bytes / 2;

        full_cache_size.min(max_cache_size)
    }
}

/// Statistics for sparse vector analysis
#[derive(Debug, Clone, Default)]
pub struct SparseVectorStats {
    pub mean_nnz: f64,
    pub min_nnz: usize,
    pub max_nnz: usize,
    pub variance_nnz: f64,
    pub total_samples: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::SparseVector;

    #[test]
    fn test_validate_binary_labels() {
        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![1.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![2.0]), -1.0),
        ];

        // Mock dataset implementation for testing
        struct MockDataset {
            samples: Vec<Sample>,
        }

        impl Dataset for MockDataset {
            fn len(&self) -> usize {
                self.samples.len()
            }
            fn dim(&self) -> usize {
                1
            }
            fn get_sample(&self, i: usize) -> Sample {
                self.samples[i].clone()
            }
            fn get_labels(&self) -> Vec<f64> {
                self.samples.iter().map(|s| s.label).collect()
            }
        }

        let dataset = MockDataset { samples };
        assert!(validation::validate_binary_labels(&dataset).is_ok());
    }

    #[test]
    fn test_sparse_vector_stats() {
        let samples = vec![
            Sample::new(SparseVector::new(vec![0, 1], vec![1.0, 2.0]), 1.0), // 2 nnz
            Sample::new(SparseVector::new(vec![0], vec![1.0]), -1.0),        // 1 nnz
            Sample::new(SparseVector::new(vec![0, 1, 2], vec![1.0, 2.0, 3.0]), 1.0), // 3 nnz
        ];

        let stats = stats::sparse_vector_stats(&samples);
        assert_eq!(stats.total_samples, 3);
        assert_eq!(stats.mean_nnz, 2.0);
        assert_eq!(stats.min_nnz, 1);
        assert_eq!(stats.max_nnz, 3);
        assert!(stats.variance_nnz > 0.0);
    }

    #[test]
    fn test_feature_frequency() {
        let samples = vec![
            Sample::new(SparseVector::new(vec![0, 1], vec![1.0, 2.0]), 1.0),
            Sample::new(SparseVector::new(vec![0, 2], vec![1.0, 3.0]), -1.0),
            Sample::new(SparseVector::new(vec![1, 2], vec![2.0, 3.0]), 1.0),
        ];

        let freq = stats::feature_frequency(&samples);
        assert_eq!(freq[&0], 2); // Feature 0 appears in 2 samples
        assert_eq!(freq[&1], 2); // Feature 1 appears in 2 samples
        assert_eq!(freq[&2], 2); // Feature 2 appears in 2 samples
    }

    #[test]
    fn test_memory_estimation() {
        let memory_100_samples = memory::estimate_kernel_cache_memory(100);
        let memory_1000_samples = memory::estimate_kernel_cache_memory(1000);

        // Memory should scale quadratically
        assert!(memory_1000_samples > memory_100_samples * 50); // Roughly 100x more

        let recommended = memory::recommend_cache_size(100, 100); // 100MB available
        assert!(recommended > 0);
        assert!(recommended <= 50 * 1024 * 1024); // At most 50MB
    }
}
