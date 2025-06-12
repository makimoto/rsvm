//! Integration tests for the rsvm library
//!
//! These tests verify end-to-end functionality across multiple modules
//! and validate real-world usage scenarios.

use rsvm::api::{quick, SVM};
use rsvm::{LibSVMDataset, Sample, SparseVector};
use std::io::Write;
use tempfile::NamedTempFile;

/// Test complete workflow: data loading -> training -> evaluation
#[test]
fn test_complete_workflow_libsvm() {
    // Create test data in LibSVM format
    let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");

    // Classic linearly separable dataset
    writeln!(temp_file, "+1 1:2.0 2:1.0").expect("Failed to write");
    writeln!(temp_file, "+1 1:1.8 2:1.1").expect("Failed to write");
    writeln!(temp_file, "+1 1:2.2 2:0.9").expect("Failed to write");
    writeln!(temp_file, "-1 1:-2.0 2:-1.0").expect("Failed to write");
    writeln!(temp_file, "-1 1:-1.8 2:-1.1").expect("Failed to write");
    writeln!(temp_file, "-1 1:-2.2 2:-0.9").expect("Failed to write");
    temp_file.flush().expect("Failed to flush");

    // Test the complete API workflow
    let model = SVM::new()
        .with_c(1.0)
        .with_epsilon(0.001)
        .with_max_iterations(1000)
        .train_from_file(temp_file.path())
        .expect("Training should succeed");

    // Evaluate on the same data (should get high accuracy)
    let accuracy = model
        .evaluate_from_file(temp_file.path())
        .expect("Evaluation should succeed");

    assert!(
        accuracy >= 0.8,
        "Accuracy should be at least 80% for linearly separable data, got: {}",
        accuracy
    );

    // Test model info
    let info = model.info();
    assert!(info.n_support_vectors > 0, "Should have support vectors");
    assert!(
        info.n_support_vectors <= 6,
        "Should not have more support vectors than samples"
    );

    // Test detailed metrics
    let dataset = LibSVMDataset::from_file(temp_file.path()).expect("Failed to load dataset");
    let metrics = model.evaluate_detailed(&dataset);

    assert!(metrics.accuracy() >= 0.8);
    assert!(metrics.precision() >= 0.8);
    assert!(metrics.recall() >= 0.8);
    assert!(metrics.f1_score() >= 0.8);
}

/// Test CSV workflow with different data characteristics
#[test]
fn test_complete_workflow_csv() {
    // Create test data in CSV format with headers
    let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");

    writeln!(temp_file, "feature1,feature2,feature3,label").expect("Failed to write");
    writeln!(temp_file, "3.0,0.0,1.5,1").expect("Failed to write");
    writeln!(temp_file, "2.8,0.1,1.4,-1").expect("Failed to write");
    writeln!(temp_file, "3.2,-0.1,1.6,1").expect("Failed to write");
    writeln!(temp_file, "-3.0,0.0,-1.5,-1").expect("Failed to write");
    writeln!(temp_file, "-2.8,-0.1,-1.4,-1").expect("Failed to write");
    writeln!(temp_file, "-3.2,0.1,-1.6,1").expect("Failed to write");
    temp_file.flush().expect("Failed to flush");

    // Test CSV loading and training
    let model = quick::train_csv(temp_file.path()).expect("CSV training should succeed");

    let accuracy = model
        .evaluate_from_csv(temp_file.path())
        .expect("CSV evaluation should succeed");

    // This dataset is not perfectly linearly separable, so lower threshold
    assert!(
        accuracy >= 0.5,
        "Accuracy should be reasonable, got: {}",
        accuracy
    );
}

/// Test various parameter configurations
#[test]
fn test_parameter_sensitivity() {
    // Create a simple but challenging dataset
    let samples = vec![
        Sample::new(SparseVector::new(vec![0, 1], vec![1.0, 1.0]), 1.0),
        Sample::new(SparseVector::new(vec![0, 1], vec![1.1, 0.9]), 1.0),
        Sample::new(SparseVector::new(vec![0, 1], vec![-1.0, -1.0]), -1.0),
        Sample::new(SparseVector::new(vec![0, 1], vec![-1.1, -0.9]), -1.0),
        Sample::new(SparseVector::new(vec![0, 1], vec![0.8, 1.2]), 1.0),
        Sample::new(SparseVector::new(vec![0, 1], vec![-0.8, -1.2]), -1.0),
    ];

    // Test different C values
    let c_values = vec![0.1, 1.0, 10.0];

    for &c in &c_values {
        let model = SVM::new()
            .with_c(c)
            .train_samples(&samples)
            .expect(&format!("Training with C={} should succeed", c));

        // All models should classify training data reasonably well
        let correct = samples
            .iter()
            .map(|sample| model.predict(sample))
            .zip(samples.iter())
            .filter(|(pred, sample)| pred.label == sample.label)
            .count();

        let accuracy = correct as f64 / samples.len() as f64;
        assert!(
            accuracy >= 0.5,
            "C={} should give reasonable accuracy, got: {}",
            c,
            accuracy
        );
    }
}

/// Test scalability with larger datasets
#[test]
fn test_scalability() {
    // Generate a larger synthetic dataset
    let mut samples = Vec::new();

    // Generate positive samples around (2, 2)
    for i in 0..50 {
        let noise = (i as f64 % 10.0) / 50.0; // Small noise
        samples.push(Sample::new(
            SparseVector::new(vec![0, 1], vec![2.0 + noise, 2.0 + noise]),
            1.0,
        ));
    }

    // Generate negative samples around (-2, -2)
    for i in 0..50 {
        let noise = (i as f64 % 10.0) / 50.0; // Small noise
        samples.push(Sample::new(
            SparseVector::new(vec![0, 1], vec![-2.0 + noise, -2.0 + noise]),
            -1.0,
        ));
    }

    // Test that training completes in reasonable time and gives good results
    let start = std::time::Instant::now();

    let model = SVM::new()
        .with_c(1.0)
        .with_max_iterations(500) // Limit iterations for speed
        .train_samples(&samples)
        .expect("Training on larger dataset should succeed");

    let duration = start.elapsed();

    // Should complete within reasonable time (generous limit for CI)
    assert!(
        duration.as_secs() < 10,
        "Training should complete within 10 seconds, took: {:?}",
        duration
    );

    // Check that we get reasonable accuracy
    let correct = samples
        .iter()
        .map(|sample| model.predict(sample))
        .zip(samples.iter())
        .filter(|(pred, sample)| pred.label == sample.label)
        .count();

    let accuracy = correct as f64 / samples.len() as f64;
    assert!(
        accuracy >= 0.8,
        "Should achieve good accuracy on synthetic data, got: {}",
        accuracy
    );

    // Check model properties
    let info = model.info();
    assert!(info.n_support_vectors > 0, "Should have support vectors");
    assert!(
        info.n_support_vectors < samples.len(),
        "Should not use all samples as support vectors"
    );
}

/// Test cross-validation functionality
#[test]
fn test_cross_validation() {
    // Create a balanced dataset
    let samples = vec![
        Sample::new(SparseVector::new(vec![0], vec![3.0]), 1.0),
        Sample::new(SparseVector::new(vec![0], vec![2.5]), 1.0),
        Sample::new(SparseVector::new(vec![0], vec![2.8]), 1.0),
        Sample::new(SparseVector::new(vec![0], vec![-3.0]), -1.0),
        Sample::new(SparseVector::new(vec![0], vec![-2.5]), -1.0),
        Sample::new(SparseVector::new(vec![0], vec![-2.8]), -1.0),
        Sample::new(SparseVector::new(vec![0], vec![3.2]), 1.0),
        Sample::new(SparseVector::new(vec![0], vec![-3.2]), -1.0),
    ];

    // Mock dataset for cross-validation
    struct MockDataset {
        samples: Vec<Sample>,
    }

    impl rsvm::Dataset for MockDataset {
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

    // Test cross-validation with different train/test splits
    let train_ratios = vec![0.6, 0.7, 0.8];

    for &ratio in &train_ratios {
        let accuracy = quick::simple_validation(&dataset, ratio, 1.0).expect(&format!(
            "Cross-validation with ratio {} should succeed",
            ratio
        ));

        assert!(
            accuracy >= 0.0 && accuracy <= 1.0,
            "Accuracy should be valid probability"
        );
        // For this simple linearly separable case, we expect reasonable performance
        assert!(
            accuracy >= 0.5,
            "Should get reasonable accuracy with ratio {}, got: {}",
            ratio,
            accuracy
        );
    }
}

/// Test error handling and edge cases
#[test]
fn test_error_handling() {
    // Test empty dataset
    let empty_samples: Vec<Sample> = vec![];
    let result = SVM::new().train_samples(&empty_samples);
    assert!(result.is_err(), "Training on empty dataset should fail");

    // Test invalid file
    let result = SVM::new().train_from_file("/nonexistent/file.libsvm");
    assert!(
        result.is_err(),
        "Training from nonexistent file should fail"
    );

    // Test dataset with all same labels
    let same_label_samples = vec![
        Sample::new(SparseVector::new(vec![0], vec![1.0]), 1.0),
        Sample::new(SparseVector::new(vec![0], vec![2.0]), 1.0),
        Sample::new(SparseVector::new(vec![0], vec![3.0]), 1.0),
    ];

    // This should still work (will just classify everything as positive)
    let result = SVM::new().train_samples(&same_label_samples);
    assert!(result.is_ok(), "Training with same labels should succeed");

    if let Ok(model) = result {
        let test_sample = Sample::new(SparseVector::new(vec![0], vec![1.5]), 1.0);
        let prediction = model.predict(&test_sample);
        assert_eq!(prediction.label, 1.0, "Should predict the only seen label");
    }
}

/// Test compatibility between different modules
#[test]
fn test_module_compatibility() {
    // Test that all major components work together seamlessly

    // 1. Create data using core types
    let samples = vec![
        Sample::new(SparseVector::new(vec![0, 2, 5], vec![1.0, 0.5, 2.0]), 1.0),
        Sample::new(SparseVector::new(vec![1, 3, 4], vec![1.5, 0.8, 1.2]), -1.0),
        Sample::new(SparseVector::new(vec![0, 1, 2], vec![2.0, 1.0, 1.5]), 1.0),
        Sample::new(SparseVector::new(vec![2, 3, 5], vec![0.5, 1.8, 0.8]), -1.0),
    ];

    // 2. Train using optimizer API
    let model = SVM::new()
        .with_c(2.0)
        .with_epsilon(0.01)
        .train_samples(&samples)
        .expect("Training should succeed");

    // 3. Test predictions
    for sample in &samples {
        let prediction = model.predict(sample);
        assert!(
            prediction.label == 1.0 || prediction.label == -1.0,
            "Prediction should be binary"
        );
        assert!(
            prediction.confidence() >= 0.0,
            "Confidence should be non-negative"
        );
    }

    // 4. Test batch predictions
    let predictions = model.predict_batch(&samples);
    assert_eq!(
        predictions.len(),
        samples.len(),
        "Should predict all samples"
    );

    // 5. Test evaluation metrics
    struct MockDataset {
        samples: Vec<Sample>,
    }

    impl rsvm::Dataset for MockDataset {
        fn len(&self) -> usize {
            self.samples.len()
        }
        fn dim(&self) -> usize {
            6
        } // Max index + 1
        fn get_sample(&self, i: usize) -> Sample {
            self.samples[i].clone()
        }
        fn get_labels(&self) -> Vec<f64> {
            self.samples.iter().map(|s| s.label).collect()
        }
    }

    let dataset = MockDataset {
        samples: samples.clone(),
    };
    let metrics = model.evaluate_detailed(&dataset);

    // Verify metrics are reasonable
    let total_samples = metrics.true_positives
        + metrics.true_negatives
        + metrics.false_positives
        + metrics.false_negatives;
    assert_eq!(
        total_samples,
        samples.len(),
        "Metrics should account for all samples"
    );

    // Verify metric calculations
    assert!(metrics.accuracy() >= 0.0 && metrics.accuracy() <= 1.0);
    assert!(metrics.precision() >= 0.0 && metrics.precision() <= 1.0);
    assert!(metrics.recall() >= 0.0 && metrics.recall() <= 1.0);
}

/// Performance benchmark test
#[test]
fn test_performance_benchmark() {
    // Create a moderately sized dataset for performance testing
    let mut samples = Vec::with_capacity(200);

    // Create a 2D spiral-like dataset that's challenging but solvable
    for i in 0..100 {
        let angle = (i as f64) * 0.1;
        let radius = 1.0 + (i as f64) * 0.01;
        let x = radius * angle.cos();
        let y = radius * angle.sin();
        samples.push(Sample::new(SparseVector::new(vec![0, 1], vec![x, y]), 1.0));

        // Add negative class (flipped)
        samples.push(Sample::new(
            SparseVector::new(vec![0, 1], vec![-x, -y]),
            -1.0,
        ));
    }

    // Benchmark training time
    let start = std::time::Instant::now();

    let model = SVM::new()
        .with_c(1.0)
        .with_max_iterations(200) // Reasonable limit
        .train_samples(&samples)
        .expect("Benchmark training should succeed");

    let training_time = start.elapsed();

    // Benchmark prediction time
    let start = std::time::Instant::now();
    let _predictions = model.predict_batch(&samples);
    let prediction_time = start.elapsed();

    // Performance assertions (generous limits for CI environments)
    assert!(
        training_time.as_millis() < 5000,
        "Training should complete within 5 seconds, took: {:?}",
        training_time
    );
    assert!(
        prediction_time.as_millis() < 100,
        "Batch prediction should complete within 100ms, took: {:?}",
        prediction_time
    );

    // Quality assertions (spiral data is challenging for linear SVM)
    let accuracy = model.evaluate(&struct_dataset(&samples));
    assert!(
        accuracy >= 0.6,
        "Should achieve reasonable accuracy on spiral dataset, got: {}",
        accuracy
    );

    println!("Performance benchmark results:");
    println!("  Dataset size: {} samples", samples.len());
    println!("  Training time: {:?}", training_time);
    println!("  Prediction time: {:?}", prediction_time);
    println!("  Accuracy: {:.2}%", accuracy * 100.0);
    println!("  Support vectors: {}", model.info().n_support_vectors);
}

// Helper function to create dataset from samples
fn struct_dataset(samples: &[Sample]) -> impl rsvm::Dataset + '_ {
    struct LocalDataset<'a> {
        samples: &'a [Sample],
    }

    impl<'a> rsvm::Dataset for LocalDataset<'a> {
        fn len(&self) -> usize {
            self.samples.len()
        }
        fn dim(&self) -> usize {
            2
        }
        fn get_sample(&self, i: usize) -> Sample {
            self.samples[i].clone()
        }
        fn get_labels(&self) -> Vec<f64> {
            self.samples.iter().map(|s| s.label).collect()
        }
    }

    LocalDataset { samples }
}
