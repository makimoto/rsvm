//! Model serialization and persistence
//!
//! This module provides functionality to save and load trained SVM models
//! for use with the CLI application and other scenarios where model persistence is needed.

use crate::api::TrainedModel;
use crate::core::{Result, SVMError, Sample, SparseVector};
use crate::kernel::LinearKernel;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

/// Serializable representation of a trained SVM model
#[derive(Serialize, Deserialize, Debug)]
pub struct SerializableModel {
    /// Support vectors
    pub support_vectors: Vec<SerializableSample>,
    /// Alpha values times labels (alpha_i * y_i)
    pub alpha_y: Vec<f64>,
    /// Bias term
    pub bias: f64,
    /// Kernel type identifier
    pub kernel_type: String,
    /// Model metadata
    pub metadata: ModelMetadata,
}

/// Serializable sample representation
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SerializableSample {
    /// Feature indices
    pub indices: Vec<usize>,
    /// Feature values
    pub values: Vec<f64>,
    /// Sample label
    pub label: f64,
}

/// Model metadata for tracking and validation
#[derive(Serialize, Deserialize, Debug)]
pub struct ModelMetadata {
    /// Library version used to create the model
    pub library_version: String,
    /// Number of support vectors
    pub n_support_vectors: usize,
    /// Training parameters used
    pub training_params: TrainingParams,
    /// Creation timestamp
    pub created_at: String,
}

/// Training parameters for reference
#[derive(Serialize, Deserialize, Debug)]
pub struct TrainingParams {
    pub c: f64,
    pub epsilon: f64,
    pub max_iterations: usize,
}

impl From<&Sample> for SerializableSample {
    fn from(sample: &Sample) -> Self {
        Self {
            indices: sample.features.indices.clone(),
            values: sample.features.values.clone(),
            label: sample.label,
        }
    }
}

impl From<&SerializableSample> for Sample {
    fn from(s: &SerializableSample) -> Self {
        Sample::new(
            SparseVector::new(s.indices.clone(), s.values.clone()),
            s.label,
        )
    }
}

impl SerializableModel {
    /// Create a serializable model from a trained model
    pub fn from_trained_model(model: &TrainedModel<LinearKernel>) -> Self {
        let info = model.info();
        let inner = model.inner();

        let support_vectors: Vec<SerializableSample> = inner
            .support_vectors()
            .iter()
            .map(SerializableSample::from)
            .collect();

        // Calculate alpha_y = alpha_i * y_i for each support vector
        let alpha_y: Vec<f64> = inner
            .alpha_values()
            .iter()
            .zip(inner.support_vectors().iter())
            .map(|(&alpha, sample)| alpha * sample.label)
            .collect();

        Self {
            support_vectors,
            alpha_y,
            bias: info.bias,
            kernel_type: "linear".to_string(),
            metadata: ModelMetadata {
                library_version: env!("CARGO_PKG_VERSION").to_string(),
                n_support_vectors: info.n_support_vectors,
                training_params: TrainingParams {
                    c: 1.0, // Default - would need to be passed in for real tracking
                    epsilon: 0.001,
                    max_iterations: 1000,
                },
                created_at: chrono::Utc::now().to_rfc3339(),
            },
        }
    }

    /// Save model to file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path).map_err(SVMError::IoError)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)
            .map_err(|e| SVMError::SerializationError(e.to_string()))?;
        Ok(())
    }

    /// Load model from file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path).map_err(SVMError::IoError)?;
        let reader = BufReader::new(file);
        let model = serde_json::from_reader(reader)
            .map_err(|e| SVMError::SerializationError(e.to_string()))?;
        Ok(model)
    }

    /// Convert back to a trained model (for linear kernel only)
    pub fn to_trained_model(&self) -> Result<TrainedModel<LinearKernel>> {
        if self.kernel_type != "linear" {
            return Err(SVMError::InvalidParameter(
                "Only linear kernel models are currently supported".to_string(),
            ));
        }

        let _support_vectors: Vec<Sample> = self.support_vectors.iter().map(Sample::from).collect();

        // Create a trained model using the internal constructor
        // This would require exposing more internals or creating a builder
        // For now, we'll return an error and implement this when needed
        Err(SVMError::InvalidParameter(
            "Model reconstruction not yet implemented".to_string(),
        ))
    }

    /// Print model summary
    pub fn print_summary(&self) {
        println!("=== SVM Model Summary ===");
        println!("Kernel Type: {}", self.kernel_type);
        println!("Support Vectors: {}", self.metadata.n_support_vectors);
        println!("Bias: {:.6}", self.bias);
        println!("Library Version: {}", self.metadata.library_version);
        println!("Created: {}", self.metadata.created_at);
        println!("Training Parameters:");
        println!("  C: {}", self.metadata.training_params.c);
        println!("  Epsilon: {}", self.metadata.training_params.epsilon);
        println!(
            "  Max Iterations: {}",
            self.metadata.training_params.max_iterations
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::SVM;
    use tempfile::NamedTempFile;

    #[test]
    fn test_serializable_sample_conversion() {
        let sample = Sample::new(SparseVector::new(vec![0, 2, 5], vec![1.0, 2.0, 3.0]), 1.0);

        let serializable = SerializableSample::from(&sample);
        assert_eq!(serializable.indices, vec![0, 2, 5]);
        assert_eq!(serializable.values, vec![1.0, 2.0, 3.0]);
        assert_eq!(serializable.label, 1.0);

        let converted_back = Sample::from(&serializable);
        assert_eq!(converted_back.features.indices, sample.features.indices);
        assert_eq!(converted_back.features.values, sample.features.values);
        assert_eq!(converted_back.label, sample.label);
    }

    #[test]
    fn test_model_serialization() -> Result<()> {
        // Create a simple model
        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![2.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-2.0]), -1.0),
        ];

        let model = SVM::new().train_samples(&samples)?;
        let serializable = SerializableModel::from_trained_model(&model);

        // Test saving and loading
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        serializable.save_to_file(temp_file.path())?;

        let loaded = SerializableModel::load_from_file(temp_file.path())?;

        assert_eq!(loaded.kernel_type, "linear");
        assert_eq!(
            loaded.support_vectors.len(),
            serializable.support_vectors.len()
        );
        assert_eq!(loaded.bias, serializable.bias);

        Ok(())
    }

    #[test]
    fn test_save_file_io_error() {
        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![2.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-2.0]), -1.0),
        ];

        let model = SVM::new().train_samples(&samples).unwrap();
        let serializable = SerializableModel::from_trained_model(&model);

        // Try to save to an invalid path
        let result = serializable.save_to_file("/invalid/path/model.json");
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(matches!(e, SVMError::IoError(_)));
        }
    }

    #[test]
    fn test_load_file_io_error() {
        // Try to load from non-existent file
        let result = SerializableModel::load_from_file("/non/existent/file.json");
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(matches!(e, SVMError::IoError(_)));
        }
    }

    #[test]
    fn test_load_invalid_json() {
        use std::io::Write;

        // Create a file with invalid JSON
        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        writeln!(temp_file, "{{ invalid json }}").expect("Failed to write");
        temp_file.flush().expect("Failed to flush");

        let result = SerializableModel::load_from_file(temp_file.path());
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(matches!(e, SVMError::SerializationError(_)));
        }
    }

    #[test]
    fn test_to_trained_model_unsupported_kernel() {
        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![2.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-2.0]), -1.0),
        ];

        let model = SVM::new().train_samples(&samples).unwrap();
        let mut serializable = SerializableModel::from_trained_model(&model);

        // Change kernel type to something unsupported
        serializable.kernel_type = "rbf".to_string();

        let result = serializable.to_trained_model();
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(matches!(e, SVMError::InvalidParameter(_)));
        }
    }

    #[test]
    fn test_to_trained_model_not_implemented() {
        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![2.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-2.0]), -1.0),
        ];

        let model = SVM::new().train_samples(&samples).unwrap();
        let serializable = SerializableModel::from_trained_model(&model);

        // This should return not implemented error
        let result = serializable.to_trained_model();
        assert!(result.is_err());
        if let Err(SVMError::InvalidParameter(msg)) = result {
            assert!(msg.contains("not yet implemented"));
        } else {
            panic!("Expected InvalidParameter error");
        }
    }

    #[test]
    fn test_print_summary() {
        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![2.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-2.0]), -1.0),
        ];

        let model = SVM::new().train_samples(&samples).unwrap();
        let serializable = SerializableModel::from_trained_model(&model);

        // This should not panic and should print to stdout
        serializable.print_summary();
    }

    #[test]
    fn test_metadata_fields() {
        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![2.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-2.0]), -1.0),
        ];

        let model = SVM::new().train_samples(&samples).unwrap();
        let serializable = SerializableModel::from_trained_model(&model);

        // Check metadata fields
        assert_eq!(
            serializable.metadata.library_version,
            env!("CARGO_PKG_VERSION")
        );
        assert_eq!(
            serializable.metadata.n_support_vectors,
            serializable.support_vectors.len()
        );
        assert_eq!(serializable.metadata.training_params.c, 1.0);
        assert_eq!(serializable.metadata.training_params.epsilon, 0.001);
        assert_eq!(serializable.metadata.training_params.max_iterations, 1000);

        // Check that created_at is a valid RFC3339 timestamp
        chrono::DateTime::parse_from_rfc3339(&serializable.metadata.created_at)
            .expect("Should be valid RFC3339 timestamp");
    }
}
