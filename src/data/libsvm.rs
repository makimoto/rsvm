//! LibSVM format dataset implementation
//!
//! Supports loading datasets in the libsvm format:
//! label index:value index:value ...
//!
//! Example:
//! +1 1:0.5 3:1.2 7:0.8
//! -1 2:0.3 5:2.1

use crate::core::{Dataset, Result, SVMError, Sample, SparseVector};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Dataset implementation for LibSVM format files
#[derive(Debug, Clone)]
pub struct LibSVMDataset {
    samples: Vec<Sample>,
    dimensions: usize,
}

impl LibSVMDataset {
    /// Load a dataset from a LibSVM format file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path).map_err(SVMError::IoError)?;
        let reader = BufReader::new(file);
        Self::from_reader(reader)
    }

    /// Load a dataset from a reader (for testing and flexibility)
    pub fn from_reader<R: BufRead>(reader: R) -> Result<Self> {
        let mut samples = Vec::new();
        let mut max_dimension = 0;

        for (line_num, line) in reader.lines().enumerate() {
            let line = line.map_err(SVMError::IoError)?;
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            match Self::parse_line(line) {
                Ok((sample, max_idx)) => {
                    samples.push(sample);
                    max_dimension = max_dimension.max(max_idx + 1);
                }
                Err(e) => {
                    return Err(SVMError::ParseError(format!(
                        "Error parsing line {}: {}",
                        line_num + 1,
                        e
                    )));
                }
            }
        }

        if samples.is_empty() {
            return Err(SVMError::EmptyDataset);
        }

        Ok(LibSVMDataset {
            samples,
            dimensions: max_dimension,
        })
    }

    /// Parse a single line in libsvm format
    fn parse_line(line: &str) -> Result<(Sample, usize)> {
        let parts: Vec<&str> = line.split_whitespace().collect();

        if parts.is_empty() {
            return Err(SVMError::ParseError("Empty line".to_string()));
        }

        // Parse label
        let label = parts[0]
            .parse::<f64>()
            .map_err(|_| SVMError::ParseError(format!("Invalid label: {}", parts[0])))?;

        // Convert +1/-1 if needed, or validate binary labels
        let label = if label == 1.0 || label == -1.0 {
            label
        } else if label > 0.0 {
            1.0
        } else {
            -1.0
        };

        // Parse feature:value pairs
        let mut indices = Vec::new();
        let mut values = Vec::new();
        let mut max_index = 0;

        for feature_str in &parts[1..] {
            let feature_parts: Vec<&str> = feature_str.split(':').collect();

            if feature_parts.len() != 2 {
                return Err(SVMError::ParseError(format!(
                    "Invalid feature format: {}",
                    feature_str
                )));
            }

            let index = feature_parts[0].parse::<usize>().map_err(|_| {
                SVMError::ParseError(format!("Invalid feature index: {}", feature_parts[0]))
            })?;

            let value = feature_parts[1].parse::<f64>().map_err(|_| {
                SVMError::ParseError(format!("Invalid feature value: {}", feature_parts[1]))
            })?;

            // libsvm uses 1-based indexing, convert to 0-based
            let zero_based_index = if index > 0 {
                index - 1
            } else {
                return Err(SVMError::ParseError(format!(
                    "Feature index must be positive: {}",
                    index
                )));
            };

            indices.push(zero_based_index);
            values.push(value);
            max_index = max_index.max(zero_based_index);
        }

        let features = SparseVector::new(indices, values);
        let sample = Sample::new(features, label);

        Ok((sample, max_index))
    }
}

impl Dataset for LibSVMDataset {
    fn len(&self) -> usize {
        self.samples.len()
    }

    fn dim(&self) -> usize {
        self.dimensions
    }

    fn get_sample(&self, i: usize) -> Sample {
        self.samples[i].clone()
    }

    fn get_labels(&self) -> Vec<f64> {
        self.samples.iter().map(|s| s.label).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_parse_line_basic() {
        let line = "+1 1:0.5 3:1.2";
        let (sample, max_idx) = LibSVMDataset::parse_line(line).unwrap();

        assert_eq!(sample.label, 1.0);
        assert_eq!(sample.features.indices, vec![0, 2]); // 1-based to 0-based
        assert_eq!(sample.features.values, vec![0.5, 1.2]);
        assert_eq!(max_idx, 2);
    }

    #[test]
    fn test_parse_line_negative_label() {
        let line = "-1 2:0.3 5:2.1";
        let (sample, max_idx) = LibSVMDataset::parse_line(line).unwrap();

        assert_eq!(sample.label, -1.0);
        assert_eq!(sample.features.indices, vec![1, 4]); // 1-based to 0-based
        assert_eq!(sample.features.values, vec![0.3, 2.1]);
        assert_eq!(max_idx, 4);
    }

    #[test]
    fn test_parse_line_binary_conversion() {
        // Positive non-unit values should become +1
        let line = "2 1:1.0";
        let (sample, _) = LibSVMDataset::parse_line(line).unwrap();
        assert_eq!(sample.label, 1.0);

        // Negative values should become -1
        let line = "-3 1:1.0";
        let (sample, _) = LibSVMDataset::parse_line(line).unwrap();
        assert_eq!(sample.label, -1.0);
    }

    #[test]
    fn test_parse_line_invalid_format() {
        // Invalid feature format
        let result = LibSVMDataset::parse_line("+1 1");
        assert!(result.is_err());

        // Invalid index
        let result = LibSVMDataset::parse_line("+1 abc:1.0");
        assert!(result.is_err());

        // Invalid value
        let result = LibSVMDataset::parse_line("+1 1:abc");
        assert!(result.is_err());

        // Zero index (libsvm is 1-based)
        let result = LibSVMDataset::parse_line("+1 0:1.0");
        assert!(result.is_err());
    }

    #[test]
    fn test_from_reader_basic() {
        let data = "+1 1:0.5 3:1.2\n-1 2:0.3 5:2.1\n";
        let reader = Cursor::new(data);

        let dataset = LibSVMDataset::from_reader(reader).unwrap();

        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.dim(), 5); // max index is 4 (0-based), so dimension is 5

        let sample1 = dataset.get_sample(0);
        assert_eq!(sample1.label, 1.0);
        assert_eq!(sample1.features.indices, vec![0, 2]);

        let sample2 = dataset.get_sample(1);
        assert_eq!(sample2.label, -1.0);
        assert_eq!(sample2.features.indices, vec![1, 4]);
    }

    #[test]
    fn test_from_reader_empty_lines_and_comments() {
        let data = "# Comment line\n+1 1:0.5\n\n# Another comment\n-1 2:0.3\n";
        let reader = Cursor::new(data);

        let dataset = LibSVMDataset::from_reader(reader).unwrap();

        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.get_labels(), vec![1.0, -1.0]);
    }

    #[test]
    fn test_from_reader_empty_dataset() {
        let data = "# Only comments\n\n";
        let reader = Cursor::new(data);

        let result = LibSVMDataset::from_reader(reader);
        assert!(matches!(result, Err(SVMError::EmptyDataset)));
    }

    #[test]
    fn test_dataset_trait_implementation() {
        let data = "+1 1:0.5 3:1.2\n-1 2:0.3\n";
        let reader = Cursor::new(data);
        let dataset = LibSVMDataset::from_reader(reader).unwrap();

        // Test Dataset trait methods
        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.dim(), 3); // max index is 2, so dimension is 3
        assert!(!dataset.is_empty());

        let labels = dataset.get_labels();
        assert_eq!(labels, vec![1.0, -1.0]);

        let batch = dataset.get_batch(&[0, 1]);
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].label, 1.0);
        assert_eq!(batch[1].label, -1.0);
    }

    #[test]
    fn test_integration_with_smo_solver() {
        use crate::core::OptimizerConfig;
        use crate::kernel::LinearKernel;
        use crate::solver::SMOSolver;
        use std::sync::Arc;

        // Create a simple linearly separable dataset
        let data = "+1 1:2.0\n-1 1:-2.0\n+1 1:1.5\n-1 1:-1.5\n";
        let reader = Cursor::new(data);
        let dataset = LibSVMDataset::from_reader(reader).unwrap();

        // Convert to samples for SMO solver
        let samples: Vec<_> = (0..dataset.len()).map(|i| dataset.get_sample(i)).collect();

        // Test with SMO solver
        let kernel = Arc::new(LinearKernel::new());
        let mut config = OptimizerConfig::default();
        config.max_iterations = 100;
        config.epsilon = 0.001;

        let solver = SMOSolver::new(kernel, config);
        let result = solver.solve(&samples).expect("Should solve successfully");

        // Verify results
        assert_eq!(result.alpha.len(), 4);
        assert!(result.iterations > 0);
        assert!(result.support_vectors.len() > 0);

        // For linearly separable case, alpha values should sum to approximately 0
        // (due to constraint sum(alpha_i * y_i) = 0)
        let alpha_sum: f64 = result
            .alpha
            .iter()
            .zip(samples.iter())
            .map(|(&alpha, sample)| alpha * sample.label)
            .sum();
        assert!(
            (alpha_sum).abs() < 0.1,
            "Alpha constraint violation: {}",
            alpha_sum
        );
    }

    #[test]
    fn test_large_dimension_handling() {
        // Test with large sparse indices
        let data = "+1 1:1.0 1000:2.0 5000:3.0\n-1 2:1.0 500:2.0\n";
        let reader = Cursor::new(data);
        let dataset = LibSVMDataset::from_reader(reader).unwrap();

        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.dim(), 5000); // max index is 4999 (0-based), so dimension is 5000

        // Verify sparse vector efficiency - indices should be sorted
        let sample = dataset.get_sample(0);
        assert_eq!(sample.features.indices, vec![0, 999, 4999]); // 0-based indices
        assert_eq!(sample.features.values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_from_file() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create a temporary file with libsvm data
        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        writeln!(temp_file, "+1 1:0.5 3:1.2").expect("Failed to write");
        writeln!(temp_file, "-1 2:0.3 5:2.1").expect("Failed to write");
        temp_file.flush().expect("Failed to flush");

        // Test loading from file
        let dataset = LibSVMDataset::from_file(temp_file.path()).unwrap();

        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.dim(), 5);
        assert_eq!(dataset.get_labels(), vec![1.0, -1.0]);
    }

    #[test]
    fn test_from_file_io_error() {
        // Test with non-existent file
        let result = LibSVMDataset::from_file("/non/existent/file.libsvm");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), SVMError::IoError(_)));
    }
}
