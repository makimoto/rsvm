//! Utility functions for SVM operations

use crate::core::{Dataset, Sample};

/// Feature scaling utilities  
pub mod scaling {
    use super::*;
    use std::collections::HashMap;

    /// Feature scaling methods
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum ScalingMethod {
        /// Min-Max scaling to [min_val, max_val] range
        MinMax { min_val: f64, max_val: f64 },
        /// Standard (Z-score) normalization: (x - mean) / std
        StandardScore,
        /// Unit scaling: x / max(|x|)  
        UnitScale,
    }

    impl Default for ScalingMethod {
        fn default() -> Self {
            Self::MinMax {
                min_val: -1.0,
                max_val: 1.0,
            }
        }
    }

    /// Feature scaling statistics for a dataset
    #[derive(Debug, Clone)]
    pub struct ScalingParams {
        pub method: ScalingMethod,
        pub feature_stats: HashMap<usize, FeatureStats>,
    }

    /// Statistics for a single feature
    #[derive(Debug, Clone)]
    pub struct FeatureStats {
        pub min: f64,
        pub max: f64,
        pub mean: f64,
        pub std: f64,
        pub count: usize,
    }

    impl ScalingParams {
        /// Compute scaling parameters from training data
        pub fn fit(samples: &[Sample], method: ScalingMethod) -> Self {
            let mut feature_stats = HashMap::new();

            // Collect all feature values
            let mut feature_values: HashMap<usize, Vec<f64>> = HashMap::new();

            for sample in samples {
                for (&feature_idx, &value) in sample
                    .features
                    .indices
                    .iter()
                    .zip(sample.features.values.iter())
                {
                    feature_values.entry(feature_idx).or_default().push(value);
                }
            }

            // Calculate statistics for each feature
            for (feature_idx, values) in feature_values {
                if values.is_empty() {
                    continue;
                }

                let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let mean = values.iter().sum::<f64>() / values.len() as f64;

                let variance = if values.len() > 1 {
                    values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                        / (values.len() - 1) as f64
                } else {
                    0.0
                };
                let std = variance.sqrt();

                feature_stats.insert(
                    feature_idx,
                    FeatureStats {
                        min,
                        max,
                        mean,
                        std,
                        count: values.len(),
                    },
                );
            }

            Self {
                method,
                feature_stats,
            }
        }

        /// Transform a single sample using fitted parameters
        pub fn transform_sample(&self, sample: &Sample) -> Sample {
            let mut scaled_values = Vec::new();
            let mut scaled_indices = Vec::new();

            for (&feature_idx, &value) in sample
                .features
                .indices
                .iter()
                .zip(sample.features.values.iter())
            {
                if let Some(stats) = self.feature_stats.get(&feature_idx) {
                    let scaled_value = self.scale_value(value, stats);

                    // Keep all values - let SparseVector handle near-zero filtering
                    scaled_indices.push(feature_idx);
                    scaled_values.push(scaled_value);
                } else {
                    // Feature not seen in training data - keep original value
                    scaled_indices.push(feature_idx);
                    scaled_values.push(value);
                }
            }

            Sample::new(
                crate::core::SparseVector::new(scaled_indices, scaled_values),
                sample.label,
            )
        }

        /// Transform multiple samples
        pub fn transform_samples(&self, samples: &[Sample]) -> Vec<Sample> {
            samples
                .iter()
                .map(|sample| self.transform_sample(sample))
                .collect()
        }

        /// Scale a single value using the appropriate method
        fn scale_value(&self, value: f64, stats: &FeatureStats) -> f64 {
            match self.method {
                ScalingMethod::MinMax { min_val, max_val } => {
                    if (stats.max - stats.min).abs() < 1e-12 {
                        // Constant feature
                        (min_val + max_val) / 2.0
                    } else {
                        let normalized = (value - stats.min) / (stats.max - stats.min);
                        min_val + normalized * (max_val - min_val)
                    }
                }
                ScalingMethod::StandardScore => {
                    if stats.std < 1e-12 {
                        // Constant feature
                        0.0
                    } else {
                        (value - stats.mean) / stats.std
                    }
                }
                ScalingMethod::UnitScale => {
                    let max_abs = stats.max.abs().max(stats.min.abs());
                    if max_abs < 1e-12 {
                        0.0
                    } else {
                        value / max_abs
                    }
                }
            }
        }
    }

    /// Convenience function: fit and transform in one step
    pub fn fit_transform(
        samples: &[Sample],
        method: ScalingMethod,
    ) -> (Vec<Sample>, ScalingParams) {
        let params = ScalingParams::fit(samples, method);
        let transformed = params.transform_samples(samples);
        (transformed, params)
    }
}

/// Validation and preprocessing utilities
pub mod validation {
    use super::*;

    /// Validate that all labels in a dataset are binary (-1 or +1)
    pub fn validate_binary_labels<D: Dataset>(dataset: &D) -> Result<(), String> {
        let labels = dataset.get_labels();
        for (i, &label) in labels.iter().enumerate() {
            if label != 1.0 && label != -1.0 {
                return Err(format!(
                    "Invalid label {label} at index {i}: labels must be +1 or -1"
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
        samples.iter().map(|sample| model.predict(sample)).collect()
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
    fn test_validate_binary_labels_invalid() {
        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![1.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![2.0]), 0.5), // Invalid label
        ];

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
        let result = validation::validate_binary_labels(&dataset);
        assert!(result.is_err());
        let error_msg = result.unwrap_err();
        assert!(error_msg.contains("Invalid label 0.5 at index 1"));
    }

    #[test]
    fn test_check_label_balance() {
        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![1.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![2.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![3.0]), -1.0),
        ];

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
        let (pos, neg, ratio) = validation::check_label_balance(&dataset);
        assert_eq!(pos, 2);
        assert_eq!(neg, 1);
        assert_eq!(ratio, 2.0);
    }

    #[test]
    fn test_check_label_balance_infinity() {
        // Test case where there are no negative samples
        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![1.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![2.0]), 1.0),
        ];

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
        let (pos, neg, ratio) = validation::check_label_balance(&dataset);
        assert_eq!(pos, 2);
        assert_eq!(neg, 0);
        assert!(ratio.is_infinite());
    }

    #[test]
    fn test_validate_feature_sparsity() {
        let samples = vec![
            Sample::new(SparseVector::new(vec![0, 2], vec![1.0, 2.0]), 1.0),
            Sample::new(SparseVector::new(vec![1], vec![3.0]), -1.0),
            Sample::new(SparseVector::new(vec![0, 1, 2], vec![4.0, 5.0, 6.0]), 1.0),
        ];

        let (max_dim, sparsity) = validation::validate_feature_sparsity(&samples);
        assert_eq!(max_dim, 3); // Max index is 2, so max dimension is 3

        // Total features: 2 + 1 + 3 = 6
        // Total possible: 3 samples * 3 dimensions = 9
        // Sparsity: 6/9 = 0.6666...
        assert!((sparsity - 6.0 / 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_validate_feature_sparsity_empty() {
        let samples = vec![];
        let (max_dim, sparsity) = validation::validate_feature_sparsity(&samples);
        assert_eq!(max_dim, 0);
        assert_eq!(sparsity, 0.0);
    }

    #[test]
    fn test_compute_kernel_row() {
        use crate::kernel::LinearKernel;

        let kernel = LinearKernel::new();
        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![1.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![2.0]), -1.0),
            Sample::new(SparseVector::new(vec![0], vec![3.0]), 1.0),
        ];

        let row = parallel::compute_kernel_row(&kernel, &samples, 0, 1);
        assert_eq!(row.len(), 2); // Should compute for samples 1 and 2
        assert_eq!(row[0], 2.0); // kernel(samples[0], samples[1]) = 1*2 = 2
        assert_eq!(row[1], 3.0); // kernel(samples[0], samples[2]) = 1*3 = 3
    }

    #[test]
    fn test_parallel_predict_batch() {
        use crate::api::SVM;

        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![2.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-2.0]), -1.0),
        ];

        let model = SVM::new().train_samples(&samples).unwrap();
        let test_samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![1.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-1.0]), -1.0),
        ];

        let predictions = parallel::predict_batch(model.inner(), &test_samples);
        assert_eq!(predictions.len(), 2);
        assert_eq!(predictions[0].label, 1.0);
        assert_eq!(predictions[1].label, -1.0);
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

    #[test]
    fn test_scaling_minmax() {
        use crate::utils::scaling::{ScalingMethod, ScalingParams};

        let samples = vec![
            Sample::new(SparseVector::new(vec![0, 1], vec![1.0, 10.0]), 1.0),
            Sample::new(SparseVector::new(vec![0, 1], vec![3.0, 20.0]), -1.0),
            Sample::new(SparseVector::new(vec![0, 1], vec![5.0, 30.0]), 1.0),
        ];

        let params = ScalingParams::fit(
            &samples,
            ScalingMethod::MinMax {
                min_val: 0.0,
                max_val: 1.0,
            },
        );
        let transformed = params.transform_samples(&samples);

        // Check first sample transformation
        let first_transformed = &transformed[0];
        assert_eq!(first_transformed.features.indices, vec![0, 1]);

        // Feature 0: min=1, max=5, so 1.0 -> 0.0
        assert!((first_transformed.features.values[0] - 0.0).abs() < 1e-10);
        // Feature 1: min=10, max=30, so 10.0 -> 0.0
        assert!((first_transformed.features.values[1] - 0.0).abs() < 1e-10);

        // Check last sample transformation
        let last_transformed = &transformed[2];
        // Feature 0: 5.0 -> 1.0
        assert!((last_transformed.features.values[0] - 1.0).abs() < 1e-10);
        // Feature 1: 30.0 -> 1.0
        assert!((last_transformed.features.values[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_scaling_standard_score() {
        use crate::utils::scaling::{ScalingMethod, ScalingParams};

        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![1.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![3.0]), -1.0),
            Sample::new(SparseVector::new(vec![0], vec![5.0]), 1.0),
        ];

        let params = ScalingParams::fit(&samples, ScalingMethod::StandardScore);
        let transformed = params.transform_samples(&samples);

        // Mean should be 3.0, std should be 2.0
        // First value (1.0): (1-3)/2 = -1.0
        let first_value = transformed[0].features.values[0];
        assert!((first_value - (-1.0)).abs() < 1e-10);

        // Second value (3.0): (3-3)/2 = 0.0
        let second_value = transformed[1].features.values[0];
        assert!(second_value.abs() < 1e-10); // Should be ~0

        // Third value (5.0): (5-3)/2 = 1.0
        let third_value = transformed[2].features.values[0];
        assert!((third_value - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_scaling_unit_scale() {
        use crate::utils::scaling::{ScalingMethod, ScalingParams};

        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![-4.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![2.0]), -1.0),
            Sample::new(SparseVector::new(vec![0], vec![8.0]), 1.0),
        ];

        let params = ScalingParams::fit(&samples, ScalingMethod::UnitScale);
        let transformed = params.transform_samples(&samples);

        // Max absolute value is 8.0
        // First: -4.0/8.0 = -0.5
        assert!((transformed[0].features.values[0] - (-0.5)).abs() < 1e-10);
        // Second: 2.0/8.0 = 0.25
        assert!((transformed[1].features.values[0] - 0.25).abs() < 1e-10);
        // Third: 8.0/8.0 = 1.0
        assert!((transformed[2].features.values[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_scaling_constant_feature() {
        use crate::utils::scaling::{ScalingMethod, ScalingParams};

        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![5.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![5.0]), -1.0),
            Sample::new(SparseVector::new(vec![0], vec![5.0]), 1.0),
        ];

        // Test MinMax with constant feature
        let params = ScalingParams::fit(
            &samples,
            ScalingMethod::MinMax {
                min_val: -1.0,
                max_val: 1.0,
            },
        );
        let transformed = params.transform_samples(&samples);

        // Constant feature should map to middle of range
        for sample in &transformed {
            assert!((sample.features.values[0] - 0.0).abs() < 1e-10);
        }

        // Test StandardScore with constant feature
        let params_std = ScalingParams::fit(&samples, ScalingMethod::StandardScore);
        let transformed_std = params_std.transform_samples(&samples);

        // Constant feature should be filtered out (scaled to 0)
        for sample in &transformed_std {
            assert!(
                sample.features.is_empty()
                    || sample.features.values.iter().all(|&v| v.abs() < 1e-10)
            );
        }
    }

    #[test]
    fn test_fit_transform_convenience() {
        use crate::utils::scaling::{fit_transform, ScalingMethod};

        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![1.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![5.0]), -1.0),
        ];

        let (transformed, params) = fit_transform(&samples, ScalingMethod::default());

        assert_eq!(transformed.len(), 2);
        assert_eq!(params.feature_stats.len(), 1);
        assert!(params.feature_stats.contains_key(&0));
    }

    #[test]
    fn test_scaling_sparse_preservation() {
        use crate::utils::scaling::{ScalingMethod, ScalingParams};

        // Test that sparsity is preserved when possible
        let samples = vec![
            Sample::new(SparseVector::new(vec![0, 2], vec![1.0, 3.0]), 1.0), // Missing feature 1
            Sample::new(SparseVector::new(vec![0, 1], vec![2.0, 5.0]), -1.0), // Missing feature 2
        ];

        let params = ScalingParams::fit(
            &samples,
            ScalingMethod::MinMax {
                min_val: 0.0,
                max_val: 1.0,
            },
        );
        let transformed = params.transform_samples(&samples);

        // First sample should still only have features 0 and 2
        assert_eq!(transformed[0].features.indices, vec![0, 2]);
        // Second sample should only have features 0 and 1
        assert_eq!(transformed[1].features.indices, vec![0, 1]);
    }
}
