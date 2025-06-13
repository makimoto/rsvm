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

        let support_vectors: Vec<Sample> = self.support_vectors.iter().map(Sample::from).collect();

        // Reconstruct alpha values from alpha_y (alpha_i * y_i)
        let alpha: Vec<f64> = self
            .alpha_y
            .iter()
            .zip(support_vectors.iter())
            .map(|(&alpha_y, sample)| alpha_y * sample.label) // alpha_i = (alpha_i * y_i) * y_i, since y_i^2 = 1
            .collect();

        // Create the trained SVM using the new constructor
        use crate::optimizer::TrainedSVM;
        use std::sync::Arc;
        let inner_model = TrainedSVM::from_components(
            Arc::new(LinearKernel::new()),
            support_vectors,
            alpha,
            self.bias,
        );

        Ok(TrainedModel::from_trained_svm(inner_model))
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
    fn test_to_trained_model_success() {
        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![2.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-2.0]), -1.0),
        ];

        let original_model = SVM::new().train_samples(&samples).unwrap();
        let serializable = SerializableModel::from_trained_model(&original_model);

        // This should now succeed
        let reconstructed_model = serializable
            .to_trained_model()
            .expect("Model reconstruction should succeed");

        // Test that the reconstructed model makes the same predictions
        let test_sample = Sample::new(SparseVector::new(vec![0], vec![1.0]), 1.0);
        let original_pred = original_model.predict(&test_sample);
        let reconstructed_pred = reconstructed_model.predict(&test_sample);

        // Labels should match
        assert_eq!(original_pred.label, reconstructed_pred.label);

        // Decision values should be very close (allowing for small numerical differences)
        assert!((original_pred.decision_value - reconstructed_pred.decision_value).abs() < 1e-10);
    }

    #[test]
    fn test_model_round_trip_reconstruction() {
        let samples = vec![
            Sample::new(SparseVector::new(vec![0, 1], vec![1.0, 2.0]), 1.0),
            Sample::new(SparseVector::new(vec![0, 1], vec![-1.0, -2.0]), -1.0),
            Sample::new(SparseVector::new(vec![0, 1], vec![1.5, 1.0]), 1.0),
        ];

        // Train original model
        let original_model = SVM::new().train_samples(&samples).unwrap();

        // Serialize and save
        let serializable = SerializableModel::from_trained_model(&original_model);
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        serializable.save_to_file(temp_file.path()).unwrap();

        // Load and reconstruct
        let loaded_serializable = SerializableModel::load_from_file(temp_file.path()).unwrap();
        let reconstructed_model = loaded_serializable
            .to_trained_model()
            .expect("Model reconstruction should succeed");

        // Test multiple samples
        let test_samples = vec![
            Sample::new(SparseVector::new(vec![0, 1], vec![0.5, 1.0]), 1.0),
            Sample::new(SparseVector::new(vec![0, 1], vec![-0.5, -1.0]), -1.0),
            Sample::new(SparseVector::new(vec![0, 1], vec![2.0, 0.5]), 1.0),
        ];

        for test_sample in &test_samples {
            let original_pred = original_model.predict(test_sample);
            let reconstructed_pred = reconstructed_model.predict(test_sample);

            assert_eq!(
                original_pred.label, reconstructed_pred.label,
                "Prediction labels should match for sample {:?}",
                test_sample
            );
            assert!(
                (original_pred.decision_value - reconstructed_pred.decision_value).abs() < 1e-10,
                "Decision values should be very close for sample {:?}: {} vs {}",
                test_sample,
                original_pred.decision_value,
                reconstructed_pred.decision_value
            );
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

    #[test]
    fn test_multiple_save_load_cycles() -> Result<()> {
        // Create a model with more complex data to ensure serialization paths are covered
        let samples = vec![
            Sample::new(SparseVector::new(vec![0, 1, 5], vec![1.5, -2.3, 4.7]), 1.0),
            Sample::new(
                SparseVector::new(vec![0, 2, 3], vec![-1.5, 2.3, -4.7]),
                -1.0,
            ),
            Sample::new(SparseVector::new(vec![1, 4], vec![0.5, 1.2]), 1.0),
        ];

        let model = SVM::new().train_samples(&samples)?;
        let serializable = SerializableModel::from_trained_model(&model);

        // Test multiple save/load cycles to ensure all code paths
        for _ in 0..3 {
            let temp_file = NamedTempFile::new().expect("Failed to create temp file");

            // This should cover lines 121-123 in save_to_file
            serializable.save_to_file(temp_file.path())?;

            // This should cover lines 130, 133 in load_from_file
            let loaded = SerializableModel::load_from_file(temp_file.path())?;

            // Verify the loaded model matches
            assert_eq!(loaded.kernel_type, serializable.kernel_type);
            assert_eq!(loaded.bias, serializable.bias);
            assert_eq!(
                loaded.support_vectors.len(),
                serializable.support_vectors.len()
            );
            assert_eq!(loaded.alpha_y.len(), serializable.alpha_y.len());

            // Check individual support vectors
            for (orig, loaded_sv) in serializable
                .support_vectors
                .iter()
                .zip(loaded.support_vectors.iter())
            {
                assert_eq!(orig.indices, loaded_sv.indices);
                assert_eq!(orig.values, loaded_sv.values);
                assert_eq!(orig.label, loaded_sv.label);
            }
        }

        Ok(())
    }

    #[test]
    fn test_round_trip_serialization_edge_cases() -> Result<()> {
        // Test with edge case data that might trigger different serialization paths
        let samples = vec![
            // Large sparse vector with many features
            Sample::new(
                SparseVector::new(
                    vec![0, 10, 100, 1000, 5000],
                    vec![f64::MIN, 0.0, f64::MAX, -f64::INFINITY, f64::INFINITY],
                ),
                1.0,
            ),
            Sample::new(SparseVector::new(vec![1], vec![-1e-10]), -1.0),
        ];

        let model = SVM::new().train_samples(&samples)?;
        let serializable = SerializableModel::from_trained_model(&model);

        // Create temp file and ensure save/load works with edge cases
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");

        // Save (should hit line 121-123)
        serializable.save_to_file(temp_file.path())?;

        // Load (should hit lines 130, 133)
        let loaded = SerializableModel::load_from_file(temp_file.path())?;

        // Verify
        assert_eq!(loaded.kernel_type, "linear");
        assert_eq!(
            loaded.support_vectors.len(),
            serializable.support_vectors.len()
        );

        Ok(())
    }
}
