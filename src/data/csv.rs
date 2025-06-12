//! CSV format dataset implementation
//!
//! Supports loading datasets from CSV files where:
//! - The last column is the label
//! - All other columns are features
//! - First row can be headers (automatically detected)

use crate::core::{Dataset, Sample, SparseVector, Result, SVMError};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Dataset implementation for CSV format files
#[derive(Debug, Clone)]
pub struct CSVDataset {
    samples: Vec<Sample>,
    dimensions: usize,
}

impl CSVDataset {
    /// Load a dataset from a CSV file
    ///
    /// The last column is assumed to be the label.
    /// Headers are automatically detected if present.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path).map_err(SVMError::IoError)?;
        let reader = BufReader::new(file);
        Self::from_reader(reader)
    }

    /// Load a dataset from a reader with optional header specification
    pub fn from_reader<R: BufRead>(reader: R) -> Result<Self> {
        Self::from_reader_with_options(reader, true)
    }

    /// Load a dataset from a reader with explicit header option
    pub fn from_reader_with_options<R: BufRead>(
        mut reader: R, 
        auto_detect_header: bool
    ) -> Result<Self> {
        let mut samples = Vec::new();
        let mut first_line = String::new();
        
        // Read first line to check for headers
        reader.read_line(&mut first_line).map_err(SVMError::IoError)?;
        let first_line = first_line.trim();
        
        if first_line.is_empty() {
            return Err(SVMError::EmptyDataset);
        }

        // Check if first line is a comment
        if first_line.starts_with('#') {
            // Skip comment, continue processing
        } else {
            // Check if first line contains headers
            let has_header = if auto_detect_header {
                Self::is_header_line(&first_line)
            } else {
                false
            };

            // If no header, process the first line as data
            if !has_header {
                if let Some(sample) = Self::parse_data_line(&first_line)? {
                    samples.push(sample);
                }
            }
        }

        // Process remaining lines
        for line in reader.lines() {
            let line = line.map_err(SVMError::IoError)?;
            let line = line.trim();
            
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if let Some(sample) = Self::parse_data_line(&line)? {
                samples.push(sample);
            }
        }

        if samples.is_empty() {
            return Err(SVMError::EmptyDataset);
        }

        // Determine dimensions from all samples
        let dimensions = if !samples.is_empty() {
            // Find the maximum index in any sample to determine dimensions
            samples.iter()
                .flat_map(|s| &s.features.indices)
                .max()
                .map(|&max_idx| max_idx + 1)
                .unwrap_or(0)
                .max(samples.iter()
                    .map(|s| s.features.indices.len())
                    .max()
                    .unwrap_or(0))
        } else {
            0
        };

        Ok(CSVDataset { samples, dimensions })
    }

    /// Check if a line appears to be a header
    fn is_header_line(line: &str) -> bool {
        let fields: Vec<&str> = line.split(',').collect();
        
        // Check if all fields (except possibly the last) fail to parse as numbers
        if fields.len() < 2 {
            return false;
        }

        // Check if most fields are non-numeric (likely headers)
        let non_numeric_count = fields.iter()
            .take(fields.len() - 1)  // Exclude last column (label)
            .filter(|field| field.trim().parse::<f64>().is_err())
            .count();
        
        non_numeric_count > fields.len() / 2
    }

    /// Parse a CSV data line into a Sample
    fn parse_data_line(line: &str) -> Result<Option<Sample>> {
        let fields: Vec<&str> = line.split(',').map(|f| f.trim()).collect();
        
        if fields.len() < 2 {
            return Err(SVMError::ParseError(
                format!("Line has too few fields: {}", line)
            ));
        }

        // Last field is the label
        let label_str = fields[fields.len() - 1];
        let label = label_str.parse::<f64>()
            .map_err(|_| SVMError::ParseError(format!("Invalid label: {}", label_str)))?;
        
        // Convert to binary label if needed
        let label = if label == 1.0 || label == -1.0 {
            label
        } else if label > 0.0 {
            1.0
        } else {
            -1.0
        };

        // Parse features
        let mut indices = Vec::new();
        let mut values = Vec::new();
        
        for (idx, field) in fields.iter().take(fields.len() - 1).enumerate() {
            if let Ok(value) = field.parse::<f64>() {
                if value != 0.0 {  // Only store non-zero values for sparsity
                    indices.push(idx);
                    values.push(value);
                }
            } else {
                return Err(SVMError::ParseError(
                    format!("Invalid feature value at column {}: {}", idx + 1, field)
                ));
            }
        }

        if indices.is_empty() {
            // All features are zero, create an empty sparse vector
            let features = SparseVector::empty();
            Ok(Some(Sample::new(features, label)))
        } else {
            let features = SparseVector::new(indices, values);
            Ok(Some(Sample::new(features, label)))
        }
    }
}

impl Dataset for CSVDataset {
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
    fn test_csv_basic() {
        let data = "1.0,2.0,1\n3.0,4.0,-1\n";
        let reader = Cursor::new(data);
        let dataset = CSVDataset::from_reader(reader).unwrap();
        
        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.dim(), 2);
        
        let sample1 = dataset.get_sample(0);
        assert_eq!(sample1.label, 1.0);
        assert_eq!(sample1.features.indices, vec![0, 1]);
        assert_eq!(sample1.features.values, vec![1.0, 2.0]);
        
        let sample2 = dataset.get_sample(1);
        assert_eq!(sample2.label, -1.0);
        assert_eq!(sample2.features.indices, vec![0, 1]);
        assert_eq!(sample2.features.values, vec![3.0, 4.0]);
    }

    #[test]
    fn test_csv_with_headers() {
        let data = "feature1,feature2,label\n1.0,2.0,1\n3.0,4.0,-1\n";
        let reader = Cursor::new(data);
        let dataset = CSVDataset::from_reader(reader).unwrap();
        
        assert_eq!(dataset.len(), 2);  // Headers should be skipped
        assert_eq!(dataset.get_labels(), vec![1.0, -1.0]);
    }

    #[test]
    fn test_csv_sparse_features() {
        let data = "1.0,0.0,2.0,1\n0.0,3.0,0.0,-1\n";
        let reader = Cursor::new(data);
        let dataset = CSVDataset::from_reader(reader).unwrap();
        
        let sample1 = dataset.get_sample(0);
        assert_eq!(sample1.features.indices, vec![0, 2]);  // Non-zero indices
        assert_eq!(sample1.features.values, vec![1.0, 2.0]);
        
        let sample2 = dataset.get_sample(1);
        assert_eq!(sample2.features.indices, vec![1]);
        assert_eq!(sample2.features.values, vec![3.0]);
    }

    #[test]
    fn test_csv_all_zeros() {
        let data = "0.0,0.0,1\n";
        let reader = Cursor::new(data);
        let dataset = CSVDataset::from_reader(reader).unwrap();
        
        let sample = dataset.get_sample(0);
        assert_eq!(sample.label, 1.0);
        assert!(sample.features.is_empty());
    }

    #[test]
    fn test_csv_label_conversion() {
        let data = "1.0,2.0,0.5\n3.0,4.0,-0.5\n5.0,6.0,0\n";
        let reader = Cursor::new(data);
        let dataset = CSVDataset::from_reader(reader).unwrap();
        
        assert_eq!(dataset.get_labels(), vec![1.0, -1.0, -1.0]);
    }

    #[test]
    fn test_csv_empty_lines_and_comments() {
        let data = "# Comment\n1.0,2.0,1\n\n3.0,4.0,-1\n";
        let reader = Cursor::new(data);
        let dataset = CSVDataset::from_reader(reader).unwrap();
        
        assert_eq!(dataset.len(), 2);
    }

    #[test]
    fn test_csv_invalid_format() {
        // Too few fields
        let data = "1.0\n";
        let reader = Cursor::new(data);
        let result = CSVDataset::from_reader(reader);
        assert!(result.is_err());
        
        // Invalid number
        let data = "1.0,abc,-1\n";
        let reader = Cursor::new(data);
        let result = CSVDataset::from_reader(reader);
        assert!(result.is_err());
    }

    #[test]
    fn test_csv_manual_header_control() {
        let data = "1.0,2.0,1\n3.0,4.0,-1\n";
        let reader = Cursor::new(data);
        
        // Explicitly disable header detection
        let dataset = CSVDataset::from_reader_with_options(reader, false).unwrap();
        assert_eq!(dataset.len(), 2);
    }

    #[test]
    fn test_is_header_line() {
        assert!(CSVDataset::is_header_line("feature1,feature2,label"));
        assert!(CSVDataset::is_header_line("x1,x2,x3,y"));
        assert!(!CSVDataset::is_header_line("1.0,2.0,3.0,1"));
        assert!(!CSVDataset::is_header_line("1"));  // Too few fields
    }

    #[test]
    fn test_integration_with_smo() {
        use crate::solver::SMOSolver;
        use crate::kernel::LinearKernel;
        use crate::core::OptimizerConfig;
        use std::sync::Arc;

        // Simple linearly separable CSV data
        let data = "2.0,0.0,1\n-2.0,0.0,-1\n1.5,0.0,1\n-1.5,0.0,-1\n";
        let reader = Cursor::new(data);
        let dataset = CSVDataset::from_reader(reader).unwrap();
        
        let samples: Vec<_> = (0..dataset.len())
            .map(|i| dataset.get_sample(i))
            .collect();
        
        let kernel = Arc::new(LinearKernel::new());
        let mut config = OptimizerConfig::default();
        config.max_iterations = 100;
        
        let solver = SMOSolver::new(kernel, config);
        let result = solver.solve(&samples).expect("Should solve");
        
        assert!(result.support_vectors.len() > 0);
        assert!(result.iterations > 0);
    }
}