//! Dataset compatibility and format validation tests
//!
//! Tests for ensuring different data formats work correctly across the pipeline

use rsvm::{api::SVM, CSVDataset, Dataset, LibSVMDataset};
use std::io::Write;
use tempfile::NamedTempFile;

/// Test LibSVM format variations
#[test]
fn test_libsvm_format_variations() {
    // Test various LibSVM format edge cases
    let test_cases = vec![
        // Basic format
        ("+1 1:0.5 3:1.2 7:0.8\n-1 2:0.3 5:2.1\n", "basic format"),
        // With comments and empty lines
        (
            "# This is a comment\n+1 1:0.5 3:1.2\n\n# Another comment\n-1 2:0.3\n",
            "with comments",
        ),
        // Different label formats
        ("1 1:0.5 2:1.0\n-1 1:-0.5 2:-1.0\n", "explicit +/-1 labels"),
        // Sparse indices (non-consecutive)
        (
            "+1 1:1.0 10:2.0 100:3.0\n-1 5:1.5 50:2.5 500:3.5\n",
            "sparse indices",
        ),
        // Single feature
        (
            "+1 1:2.0\n-1 1:-2.0\n+1 1:1.8\n-1 1:-1.8\n",
            "single feature",
        ),
        // Many features
        (
            "+1 1:0.1 2:0.2 3:0.3 4:0.4 5:0.5\n-1 1:-0.1 2:-0.2 3:-0.3 4:-0.4 5:-0.5\n",
            "many features",
        ),
    ];

    for (data, description) in test_cases {
        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        write!(temp_file, "{}", data).expect("Failed to write");
        temp_file.flush().expect("Failed to flush");

        // Test dataset loading
        let dataset = LibSVMDataset::from_file(temp_file.path())
            .expect(&format!("Failed to load LibSVM dataset: {}", description));

        assert!(
            dataset.len() >= 2,
            "Dataset should have at least 2 samples: {}",
            description
        );
        assert!(
            dataset.dim() > 0,
            "Dataset should have dimensions: {}",
            description
        );

        // Test training
        let result = SVM::new().train(&dataset);
        assert!(
            result.is_ok(),
            "Training should succeed for: {}",
            description
        );

        if let Ok(model) = result {
            // Test that we can make predictions
            let sample = dataset.get_sample(0);
            let prediction = model.predict(&sample);
            assert!(
                prediction.label == 1.0 || prediction.label == -1.0,
                "Prediction should be binary for: {}",
                description
            );
        }
    }
}

/// Test CSV format variations
#[test]
fn test_csv_format_variations() {
    let test_cases = vec![
        // With header
        (
            "feature1,feature2,label\n1.0,2.0,1\n-1.0,-2.0,-1\n2.0,1.0,1\n",
            "with header",
        ),
        // Without header
        ("1.0,2.0,1\n-1.0,-2.0,-1\n2.0,1.0,1\n", "without header"),
        // With comments (CSV doesn't support comments like LibSVM)
        (
            "feature1,feature2,label\n1.0,2.0,1\n-1.0,-2.0,-1\n",
            "simple format",
        ),
        // Different separators (still using comma for now)
        (
            "1.0,2.0,3.0,1\n-1.0,-2.0,-3.0,-1\n0.5,1.5,2.5,1\n",
            "multiple features",
        ),
        // Floating point labels
        ("1.0,2.0,1.0\n-1.0,-2.0,-1.0\n", "floating point labels"),
        // Mixed positive/negative values
        (
            "1.5,-0.5,1\n-1.5,0.5,-1\n0.8,-1.2,1\n-0.8,1.2,-1\n",
            "mixed signs",
        ),
    ];

    for (data, description) in test_cases {
        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        write!(temp_file, "{}", data).expect("Failed to write");
        temp_file.flush().expect("Failed to flush");

        // Test dataset loading
        let dataset = CSVDataset::from_file(temp_file.path())
            .expect(&format!("Failed to load CSV dataset: {}", description));

        assert!(
            dataset.len() >= 2,
            "Dataset should have at least 2 samples: {}",
            description
        );
        assert!(
            dataset.dim() > 0,
            "Dataset should have dimensions: {}",
            description
        );

        // Test training
        let result = SVM::new().train(&dataset);
        assert!(
            result.is_ok(),
            "Training should succeed for: {}",
            description
        );
    }
}

/// Test cross-format compatibility
#[test]
fn test_cross_format_compatibility() {
    // Create the same logical dataset in both formats
    let libsvm_data = "+1 1:2.0 2:1.0\n+1 1:1.8 2:1.1\n-1 1:-2.0 2:-1.0\n-1 1:-1.8 2:-1.1\n";
    let csv_data = "feature1,feature2,label\n2.0,1.0,1\n1.8,1.1,1\n-2.0,-1.0,-1\n-1.8,-1.1,-1\n";

    // Create temporary files
    let mut libsvm_file = NamedTempFile::new().expect("Failed to create LibSVM temp file");
    write!(libsvm_file, "{}", libsvm_data).expect("Failed to write LibSVM data");
    libsvm_file.flush().expect("Failed to flush LibSVM file");

    let mut csv_file = NamedTempFile::new().expect("Failed to create CSV temp file");
    write!(csv_file, "{}", csv_data).expect("Failed to write CSV data");
    csv_file.flush().expect("Failed to flush CSV file");

    // Load both datasets
    let libsvm_dataset =
        LibSVMDataset::from_file(libsvm_file.path()).expect("Failed to load LibSVM dataset");
    let csv_dataset = CSVDataset::from_file(csv_file.path()).expect("Failed to load CSV dataset");

    // Verify they have the same structure
    assert_eq!(
        libsvm_dataset.len(),
        csv_dataset.len(),
        "Datasets should have same length"
    );
    assert_eq!(
        libsvm_dataset.dim(),
        csv_dataset.dim(),
        "Datasets should have same dimensions"
    );

    // Verify labels are the same
    let libsvm_labels = libsvm_dataset.get_labels();
    let csv_labels = csv_dataset.get_labels();
    assert_eq!(libsvm_labels, csv_labels, "Labels should be identical");

    // Train models on both datasets
    let libsvm_model = SVM::new()
        .train(&libsvm_dataset)
        .expect("LibSVM training should succeed");
    let csv_model = SVM::new()
        .train(&csv_dataset)
        .expect("CSV training should succeed");

    // Both models should achieve similar performance on the same data
    let libsvm_accuracy = libsvm_model.evaluate(&libsvm_dataset);
    let csv_accuracy = csv_model.evaluate(&csv_dataset);

    // Allow some difference due to potential floating point variations
    let accuracy_diff = (libsvm_accuracy - csv_accuracy).abs();
    assert!(
        accuracy_diff < 0.1,
        "Models should have similar accuracy: LibSVM={}, CSV={}",
        libsvm_accuracy,
        csv_accuracy
    );
}

/// Test large dimension handling
#[test]
fn test_large_dimensions() {
    // Create a dataset with sparse, high-dimensional features
    let mut libsvm_data = String::new();

    // Positive samples with features at high indices
    libsvm_data.push_str("+1 100:1.0 1000:2.0 10000:1.5\n");
    libsvm_data.push_str("+1 150:1.2 1500:1.8 15000:1.3\n");

    // Negative samples
    libsvm_data.push_str("-1 200:1.0 2000:2.0 20000:1.5\n");
    libsvm_data.push_str("-1 250:1.2 2500:1.8 25000:1.3\n");

    let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
    write!(temp_file, "{}", libsvm_data).expect("Failed to write");
    temp_file.flush().expect("Failed to flush");

    // Test loading
    let dataset = LibSVMDataset::from_file(temp_file.path())
        .expect("Failed to load high-dimensional dataset");

    assert_eq!(dataset.len(), 4);
    assert_eq!(dataset.dim(), 25000); // Max index was 24999 (0-based), so dim is 25000

    // Test training (should handle high dimensions efficiently)
    let start = std::time::Instant::now();
    let model = SVM::new()
        .with_max_iterations(100) // Limit for speed
        .train(&dataset)
        .expect("Training on high-dimensional data should succeed");
    let duration = start.elapsed();

    // Should complete reasonably quickly despite high dimensions
    assert!(
        duration.as_secs() < 5,
        "High-dimensional training should be fast due to sparsity"
    );

    // Verify model works
    let sample = dataset.get_sample(0);
    let prediction = model.predict(&sample);
    assert!(prediction.label == 1.0 || prediction.label == -1.0);
}

/// Test malformed data handling
#[test]
fn test_malformed_data_handling() {
    let malformed_cases = vec![
        ("invalid_label 1:1.0\n", "invalid label"),
        ("+1 invalid_feature\n", "invalid feature format"),
        ("+1 0:1.0\n", "zero-based index in LibSVM"),
        ("+1 1:invalid_value\n", "invalid feature value"),
        ("1,invalid_number\n", "invalid CSV number"),
        ("", "empty file"),
    ];

    for (data, description) in malformed_cases {
        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        write!(temp_file, "{}", data).expect("Failed to write");
        temp_file.flush().expect("Failed to flush");

        // LibSVM format should handle errors gracefully
        if description.contains("LibSVM")
            || description.contains("label")
            || description.contains("feature")
            || description.contains("zero-based")
            || description.contains("value")
            || description.contains("empty")
        {
            let result = LibSVMDataset::from_file(temp_file.path());
            assert!(
                result.is_err(),
                "LibSVM should reject malformed data: {}",
                description
            );
        }

        // CSV format should handle errors gracefully
        if description.contains("CSV")
            || description.contains("number")
            || description.contains("empty")
        {
            let result = CSVDataset::from_file(temp_file.path());
            assert!(
                result.is_err(),
                "CSV should reject malformed data: {}",
                description
            );
        }
    }
}

/// Test dataset statistics and validation
#[test]
fn test_dataset_validation() {
    // Create a dataset with various characteristics
    let data = "+1 1:3.0 2:4.0\n+1 1:2.8 2:4.2\n+1 1:3.2 2:3.8\n-1 1:-3.0 2:-4.0\n-1 1:-2.8 2:-4.2\n-1 1:-3.2 2:-3.8\n";

    let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
    write!(temp_file, "{}", data).expect("Failed to write");
    temp_file.flush().expect("Failed to flush");

    let dataset = LibSVMDataset::from_file(temp_file.path()).expect("Failed to load dataset");

    // Test using utility functions
    use rsvm::utils;

    // Validate binary labels
    let validation_result = utils::validation::validate_binary_labels(&dataset);
    assert!(
        validation_result.is_ok(),
        "Dataset should have valid binary labels"
    );

    // Check label balance
    let (pos_count, neg_count, balance_ratio) = utils::validation::check_label_balance(&dataset);
    assert_eq!(pos_count, 3, "Should have 3 positive samples");
    assert_eq!(neg_count, 3, "Should have 3 negative samples");
    assert!(
        (balance_ratio - 1.0).abs() < 0.1,
        "Dataset should be balanced"
    );

    // Test sparse vector statistics
    let samples: Vec<_> = (0..dataset.len()).map(|i| dataset.get_sample(i)).collect();
    let stats = utils::stats::sparse_vector_stats(&samples);

    assert_eq!(stats.total_samples, 6);
    assert_eq!(stats.mean_nnz, 2.0); // Each sample has 2 features
    assert_eq!(stats.min_nnz, 2);
    assert_eq!(stats.max_nnz, 2);
    assert_eq!(stats.variance_nnz, 0.0); // All samples have same number of features

    // Test feature frequency
    let freq = utils::stats::feature_frequency(&samples);
    assert_eq!(freq[&0], 6); // Feature 0 appears in all samples (0-based indexing)
    assert_eq!(freq[&1], 6); // Feature 1 appears in all samples
}

/// Test memory efficiency with cache
#[test]
fn test_memory_efficiency() {
    use rsvm::utils::memory;

    // Test memory estimation functions
    let mem_100 = memory::estimate_kernel_cache_memory(100);
    let mem_1000 = memory::estimate_kernel_cache_memory(1000);

    // Memory should scale roughly quadratically
    assert!(
        mem_1000 > mem_100 * 50,
        "Memory should scale with dataset size squared"
    );

    // Test cache size recommendation
    let recommended = memory::recommend_cache_size(100, 100); // 100 samples, 100MB available
    assert!(recommended > 0, "Should recommend positive cache size");
    assert!(
        recommended <= 50 * 1024 * 1024,
        "Should not exceed 50% of available memory"
    );

    // Test with larger datasets
    let recommended_large = memory::recommend_cache_size(10000, 1000); // 10k samples, 1GB available
    assert!(
        recommended_large > recommended,
        "Larger datasets should get more cache"
    );
}
