//! Integration tests for the CLI application
//!
//! These tests verify that the CLI commands work correctly with real data files.

use std::io::Write;
use std::process::Command;
use tempfile::{NamedTempFile, TempDir};

/// Helper to create test data files
struct TestDataFiles {
    pub libsvm_file: NamedTempFile,
    pub csv_file: NamedTempFile,
    pub test_libsvm_file: NamedTempFile,
    pub _test_csv_file: NamedTempFile,
}

impl TestDataFiles {
    fn new() -> std::io::Result<Self> {
        // Create LibSVM training data
        let mut libsvm_file = NamedTempFile::new()?;
        writeln!(libsvm_file, "+1 1:2.0 2:1.0")?;
        writeln!(libsvm_file, "-1 1:-2.0 2:-1.0")?;
        writeln!(libsvm_file, "+1 1:1.5 2:0.8")?;
        writeln!(libsvm_file, "-1 1:-1.5 2:-0.8")?;
        writeln!(libsvm_file, "+1 1:1.8 2:0.9")?;
        writeln!(libsvm_file, "-1 1:-1.8 2:-0.9")?;
        libsvm_file.flush()?;

        // Create CSV training data
        let mut csv_file = NamedTempFile::with_suffix(".csv")?;
        writeln!(csv_file, "feature1,feature2,label")?;
        writeln!(csv_file, "2.0,1.0,1")?;
        writeln!(csv_file, "-2.0,-1.0,-1")?;
        writeln!(csv_file, "1.5,0.8,1")?;
        writeln!(csv_file, "-1.5,-0.8,-1")?;
        writeln!(csv_file, "1.8,0.9,1")?;
        writeln!(csv_file, "-1.8,-0.9,-1")?;
        csv_file.flush()?;

        // Create LibSVM test data
        let mut test_libsvm_file = NamedTempFile::new()?;
        writeln!(test_libsvm_file, "+1 1:1.6 2:0.7")?;
        writeln!(test_libsvm_file, "-1 1:-1.6 2:-0.7")?;
        test_libsvm_file.flush()?;

        // Create CSV test data
        let mut test_csv_file = NamedTempFile::with_suffix(".csv")?;
        writeln!(test_csv_file, "feature1,feature2,label")?;
        writeln!(test_csv_file, "1.6,0.7,1")?;
        writeln!(test_csv_file, "-1.6,-0.7,-1")?;
        test_csv_file.flush()?;

        Ok(TestDataFiles {
            libsvm_file,
            csv_file,
            test_libsvm_file,
            _test_csv_file: test_csv_file,
        })
    }
}

/// Get the path to the compiled CLI binary
fn get_cli_binary_path() -> String {
    // Try to find the binary in target/debug or target/release
    let debug_path = "target/debug/rsvm";
    let release_path = "target/release/rsvm";

    if std::path::Path::new(debug_path).exists() {
        debug_path.to_string()
    } else if std::path::Path::new(release_path).exists() {
        release_path.to_string()
    } else {
        // Build the binary if it doesn't exist
        let output = Command::new("cargo")
            .args(&["build", "--bin", "rsvm"])
            .output()
            .expect("Failed to build CLI binary");

        if !output.status.success() {
            panic!(
                "Failed to build CLI binary: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        debug_path.to_string()
    }
}

#[test]
fn test_cli_train_command_libsvm() {
    let test_data = TestDataFiles::new().expect("Failed to create test data");
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let model_path = temp_dir.path().join("model.json");

    let output = Command::new(get_cli_binary_path())
        .args(&[
            "train",
            "--data",
            test_data.libsvm_file.path().to_str().unwrap(),
            "--output",
            model_path.to_str().unwrap(),
            "--format",
            "libsvm",
            "-C",
            "1.0",
            "--epsilon",
            "0.001",
            "--max-iterations",
            "100",
        ])
        .output()
        .expect("Failed to run CLI train command");

    assert!(
        output.status.success(),
        "Train command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(model_path.exists(), "Model file was not created");

    // The main success criteria are that the command succeeded and model file was created
    // Output content depends on log level, so we don't require specific messages
}

#[test]
fn test_cli_train_command_csv() {
    let test_data = TestDataFiles::new().expect("Failed to create test data");
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let model_path = temp_dir.path().join("model.json");

    let output = Command::new(get_cli_binary_path())
        .args(&[
            "train",
            "--data",
            test_data.csv_file.path().to_str().unwrap(),
            "--output",
            model_path.to_str().unwrap(),
            "--format",
            "csv",
            "-C",
            "2.0",
        ])
        .output()
        .expect("Failed to run CLI train command");

    assert!(
        output.status.success(),
        "Train command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(model_path.exists(), "Model file was not created");
}

#[test]
fn test_cli_train_auto_format_detection() {
    let test_data = TestDataFiles::new().expect("Failed to create test data");
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let model_path = temp_dir.path().join("model.json");

    // Test with LibSVM file (auto-detection)
    let output = Command::new(get_cli_binary_path())
        .args(&[
            "train",
            "--data",
            test_data.libsvm_file.path().to_str().unwrap(),
            "--output",
            model_path.to_str().unwrap(),
            // No --format specified, should auto-detect
        ])
        .output()
        .expect("Failed to run CLI train command");

    assert!(
        output.status.success(),
        "Auto-detection train command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(model_path.exists(), "Model file was not created");
}

#[test]
fn test_cli_info_command() {
    let test_data = TestDataFiles::new().expect("Failed to create test data");
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let model_path = temp_dir.path().join("model.json");

    // First train a model
    let train_output = Command::new(get_cli_binary_path())
        .args(&[
            "train",
            "--data",
            test_data.libsvm_file.path().to_str().unwrap(),
            "--output",
            model_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to train model");

    assert!(train_output.status.success());

    // Then get info about it
    let info_output = Command::new(get_cli_binary_path())
        .args(&["info", model_path.to_str().unwrap()])
        .output()
        .expect("Failed to run CLI info command");

    assert!(
        info_output.status.success(),
        "Info command failed: {}",
        String::from_utf8_lossy(&info_output.stderr)
    );

    let stdout = String::from_utf8_lossy(&info_output.stdout);
    assert!(stdout.contains("Support Vector Details"));
    assert!(stdout.contains("Alpha*Y values"));
}

#[test]
fn test_cli_predict_command() {
    let test_data = TestDataFiles::new().expect("Failed to create test data");
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let model_path = temp_dir.path().join("model.json");

    // First train a model
    let train_output = Command::new(get_cli_binary_path())
        .args(&[
            "train",
            "--data",
            test_data.libsvm_file.path().to_str().unwrap(),
            "--output",
            model_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to train model");

    assert!(train_output.status.success());

    // Then make predictions
    let predict_output = Command::new(get_cli_binary_path())
        .args(&[
            "predict",
            "--model",
            model_path.to_str().unwrap(),
            "--data",
            test_data.test_libsvm_file.path().to_str().unwrap(),
            "--format",
            "libsvm",
        ])
        .output()
        .expect("Failed to run CLI predict command");

    assert!(
        predict_output.status.success(),
        "Predict command failed: {}",
        String::from_utf8_lossy(&predict_output.stderr)
    );

    let stdout = String::from_utf8_lossy(&predict_output.stdout);
    assert!(stdout.contains("Predictions for") && stdout.contains("samples"));
}

#[test]
fn test_cli_predict_with_confidence() {
    let test_data = TestDataFiles::new().expect("Failed to create test data");
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let model_path = temp_dir.path().join("model.json");

    // Train model
    let train_output = Command::new(get_cli_binary_path())
        .args(&[
            "train",
            "--data",
            test_data.libsvm_file.path().to_str().unwrap(),
            "--output",
            model_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to train model");

    assert!(train_output.status.success());

    // Predict with confidence scores
    let predict_output = Command::new(get_cli_binary_path())
        .args(&[
            "predict",
            "--model",
            model_path.to_str().unwrap(),
            "--data",
            test_data.test_libsvm_file.path().to_str().unwrap(),
            "--confidence",
        ])
        .output()
        .expect("Failed to run CLI predict command");

    assert!(predict_output.status.success());

    let stdout = String::from_utf8_lossy(&predict_output.stdout);
    assert!(stdout.contains("confidence"));
}

#[test]
fn test_cli_evaluate_command() {
    let test_data = TestDataFiles::new().expect("Failed to create test data");
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let model_path = temp_dir.path().join("model.json");

    // Train model
    let train_output = Command::new(get_cli_binary_path())
        .args(&[
            "train",
            "--data",
            test_data.libsvm_file.path().to_str().unwrap(),
            "--output",
            model_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to train model");

    assert!(train_output.status.success());

    // Evaluate model
    let eval_output = Command::new(get_cli_binary_path())
        .args(&[
            "evaluate",
            "--model",
            model_path.to_str().unwrap(),
            "--data",
            test_data.test_libsvm_file.path().to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run CLI evaluate command");

    assert!(
        eval_output.status.success(),
        "Evaluate command failed: {}",
        String::from_utf8_lossy(&eval_output.stderr)
    );

    let stdout = String::from_utf8_lossy(&eval_output.stdout);
    assert!(stdout.contains("Model Evaluation"));
    assert!(stdout.contains("Test Results"));
}

#[test]
fn test_cli_evaluate_detailed() {
    let test_data = TestDataFiles::new().expect("Failed to create test data");
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let model_path = temp_dir.path().join("model.json");

    // Train model
    let train_output = Command::new(get_cli_binary_path())
        .args(&[
            "train",
            "--data",
            test_data.libsvm_file.path().to_str().unwrap(),
            "--output",
            model_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to train model");

    assert!(train_output.status.success());

    // Evaluate with detailed metrics
    let eval_output = Command::new(get_cli_binary_path())
        .args(&[
            "evaluate",
            "--model",
            model_path.to_str().unwrap(),
            "--data",
            test_data.test_libsvm_file.path().to_str().unwrap(),
            "--detailed",
        ])
        .output()
        .expect("Failed to run CLI evaluate command");

    assert!(eval_output.status.success());

    let stdout = String::from_utf8_lossy(&eval_output.stdout);
    assert!(stdout.contains("Detailed Metrics"));
}

#[test]
fn test_cli_quick_eval_command() {
    let test_data = TestDataFiles::new().expect("Failed to create test data");

    let output = Command::new(get_cli_binary_path())
        .args(&[
            "quick",
            "eval",
            test_data.libsvm_file.path().to_str().unwrap(),
            test_data.test_libsvm_file.path().to_str().unwrap(),
            "-C",
            "1.0",
        ])
        .output()
        .expect("Failed to run CLI quick eval command");

    assert!(
        output.status.success(),
        "Quick eval command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Quick Evaluation Results"));
    assert!(stdout.contains("Test accuracy"));
}

#[test]
fn test_cli_quick_cv_command() {
    let test_data = TestDataFiles::new().expect("Failed to create test data");

    let output = Command::new(get_cli_binary_path())
        .args(&[
            "quick",
            "cv",
            test_data.libsvm_file.path().to_str().unwrap(),
            "--ratio",
            "0.8",
            "-C",
            "1.0",
        ])
        .output()
        .expect("Failed to run CLI quick cv command");

    assert!(
        output.status.success(),
        "Quick CV command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Cross-Validation Results"));
    assert!(stdout.contains("CV accuracy"));
}

#[test]
fn test_cli_error_handling_invalid_file() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let model_path = temp_dir.path().join("model.json");

    let output = Command::new(get_cli_binary_path())
        .args(&[
            "train",
            "--data",
            "/nonexistent/file.libsvm",
            "--output",
            model_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run CLI command");

    assert!(
        !output.status.success(),
        "Command should have failed with invalid file"
    );
}

#[test]
fn test_cli_error_handling_invalid_format() {
    let test_data = TestDataFiles::new().expect("Failed to create test data");
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let model_path = temp_dir.path().join("model.json");

    let output = Command::new(get_cli_binary_path())
        .args(&[
            "train",
            "--data",
            test_data.libsvm_file.path().to_str().unwrap(),
            "--output",
            model_path.to_str().unwrap(),
            "--format",
            "invalid_format",
        ])
        .output()
        .expect("Failed to run CLI command");

    assert!(
        !output.status.success(),
        "Command should have failed with invalid format"
    );
}

#[test]
fn test_cli_verbose_and_debug_flags() {
    let test_data = TestDataFiles::new().expect("Failed to create test data");
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let model_path = temp_dir.path().join("model.json");

    // Test verbose flag
    let verbose_output = Command::new(get_cli_binary_path())
        .args(&[
            "-v",
            "train",
            "--data",
            test_data.libsvm_file.path().to_str().unwrap(),
            "--output",
            model_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run CLI command with verbose flag");

    assert!(verbose_output.status.success());

    // Test debug flag
    let debug_output = Command::new(get_cli_binary_path())
        .args(&[
            "-d",
            "train",
            "--data",
            test_data.libsvm_file.path().to_str().unwrap(),
            "--output",
            model_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run CLI command with debug flag");

    assert!(debug_output.status.success());
}

#[test]
fn test_cli_help_output() {
    let output = Command::new(get_cli_binary_path())
        .args(&["--help"])
        .output()
        .expect("Failed to run CLI help command");

    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("A Rust implementation of Support Vector Machine"));
    assert!(stdout.contains("train"));
    assert!(stdout.contains("predict"));
    assert!(stdout.contains("evaluate"));
    assert!(stdout.contains("info"));
    assert!(stdout.contains("quick"));
}

#[test]
fn test_cli_version_output() {
    let output = Command::new(get_cli_binary_path())
        .args(&["--version"])
        .output()
        .expect("Failed to run CLI version command");

    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("rsvm"));
}

#[test]
fn test_cli_train_insufficient_samples() {
    let _test_data = TestDataFiles::new().expect("Failed to create test data");
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let model_path = temp_dir.path().join("model.json");

    // Create file with only 1 sample (insufficient for training)
    let mut single_sample_file = NamedTempFile::new().expect("Failed to create temp file");
    writeln!(single_sample_file, "+1 1:2.0 2:1.0").expect("Failed to write");
    single_sample_file.flush().expect("Failed to flush");

    let output = Command::new(get_cli_binary_path())
        .args(&[
            "train",
            "--data",
            single_sample_file.path().to_str().unwrap(),
            "--output",
            model_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run CLI train command");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("at least 2 samples") || stderr.contains("Dataset must contain"));
}

#[test]
fn test_cli_predict_stdout_only() {
    let test_data = TestDataFiles::new().expect("Failed to create test data");
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let model_path = temp_dir.path().join("model.json");

    // First train a model
    let train_output = Command::new(get_cli_binary_path())
        .args(&[
            "train",
            "--data",
            test_data.libsvm_file.path().to_str().unwrap(),
            "--output",
            model_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to train model");

    assert!(train_output.status.success());

    // Then predict (currently only outputs to stdout, not to file)
    let predict_output = Command::new(get_cli_binary_path())
        .args(&[
            "predict",
            "--model",
            model_path.to_str().unwrap(),
            "--data",
            test_data.test_libsvm_file.path().to_str().unwrap(),
            "--confidence",
        ])
        .output()
        .expect("Failed to run CLI predict command");

    if !predict_output.status.success() {
        eprintln!(
            "Predict command failed with stderr: {}",
            String::from_utf8_lossy(&predict_output.stderr)
        );
        eprintln!(
            "Predict command failed with stdout: {}",
            String::from_utf8_lossy(&predict_output.stdout)
        );
    }

    assert!(predict_output.status.success());

    // Verify output contains predictions (since --output is not yet implemented, output goes to stdout)
    let stdout = String::from_utf8_lossy(&predict_output.stdout);
    assert!(stdout.contains("Predictions for") && stdout.contains("samples"));
    assert!(stdout.contains("confidence"));
}

#[test]
fn test_cli_predict_csv_format() {
    let test_data = TestDataFiles::new().expect("Failed to create test data");
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let model_path = temp_dir.path().join("model.json");

    // Train with CSV data
    let train_output = Command::new(get_cli_binary_path())
        .args(&[
            "train",
            "--data",
            test_data.csv_file.path().to_str().unwrap(),
            "--output",
            model_path.to_str().unwrap(),
            "--format",
            "csv",
        ])
        .output()
        .expect("Failed to train model");

    assert!(train_output.status.success());

    // Predict with CSV format
    let predict_output = Command::new(get_cli_binary_path())
        .args(&[
            "predict",
            "--model",
            model_path.to_str().unwrap(),
            "--data",
            test_data._test_csv_file.path().to_str().unwrap(),
            "--format",
            "csv",
        ])
        .output()
        .expect("Failed to run CLI predict command");

    assert!(predict_output.status.success());
}

#[test]
fn test_cli_evaluate_csv_format() {
    let test_data = TestDataFiles::new().expect("Failed to create test data");
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let model_path = temp_dir.path().join("model.json");

    // Train with CSV data
    let train_output = Command::new(get_cli_binary_path())
        .args(&[
            "train",
            "--data",
            test_data.csv_file.path().to_str().unwrap(),
            "--output",
            model_path.to_str().unwrap(),
            "--format",
            "csv",
        ])
        .output()
        .expect("Failed to train model");

    assert!(train_output.status.success());

    // Evaluate with CSV format
    let eval_output = Command::new(get_cli_binary_path())
        .args(&[
            "evaluate",
            "--model",
            model_path.to_str().unwrap(),
            "--data",
            test_data._test_csv_file.path().to_str().unwrap(),
            "--format",
            "csv",
        ])
        .output()
        .expect("Failed to run CLI evaluate command");

    assert!(eval_output.status.success());
}

#[test]
fn test_cli_quick_cv_with_csv() {
    let test_data = TestDataFiles::new().expect("Failed to create test data");

    let output = Command::new(get_cli_binary_path())
        .args(&[
            "quick",
            "cv",
            test_data.csv_file.path().to_str().unwrap(),
            "--ratio",
            "0.8",
            "-C",
            "1.0",
        ])
        .output()
        .expect("Failed to run CLI quick cv command");

    if !output.status.success() {
        eprintln!(
            "Command failed with stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        eprintln!(
            "Command failed with stdout: {}",
            String::from_utf8_lossy(&output.stdout)
        );
    }

    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Cross-Validation Results"));
}

#[test]
fn test_cli_train_with_unknown_extension() {
    let _test_data = TestDataFiles::new().expect("Failed to create test data");
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let model_path = temp_dir.path().join("model.json");

    // Create a file with unknown extension (should default to LibSVM)
    let mut unknown_file =
        NamedTempFile::with_suffix(".unknown").expect("Failed to create temp file");
    writeln!(unknown_file, "+1 1:2.0").expect("Failed to write");
    writeln!(unknown_file, "-1 1:-2.0").expect("Failed to write");
    unknown_file.flush().expect("Failed to flush");

    let output = Command::new(get_cli_binary_path())
        .args(&[
            "train",
            "--data",
            unknown_file.path().to_str().unwrap(),
            "--output",
            model_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run CLI train command");

    assert!(output.status.success()); // Should default to LibSVM format
    assert!(model_path.exists());
}

#[test]
fn test_cli_info_large_model() {
    let _test_data = TestDataFiles::new().expect("Failed to create test data");
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let model_path = temp_dir.path().join("model.json");

    // Create a larger dataset to potentially get more support vectors
    let mut large_file = NamedTempFile::new().expect("Failed to create temp file");
    for i in 0..20 {
        writeln!(
            large_file,
            "{} 1:{} 2:{} 3:{} 4:{} 5:{} 6:{}",
            if i % 2 == 0 { 1 } else { -1 },
            i as f64 * 0.1,
            i as f64 * 0.2,
            i as f64 * 0.3,
            i as f64 * 0.4,
            i as f64 * 0.5,
            i as f64 * 0.6
        )
        .expect("Failed to write");
    }
    large_file.flush().expect("Failed to flush");

    // Train model
    let train_output = Command::new(get_cli_binary_path())
        .args(&[
            "train",
            "--data",
            large_file.path().to_str().unwrap(),
            "--output",
            model_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to train model");

    assert!(train_output.status.success());

    // Get info about the model
    let info_output = Command::new(get_cli_binary_path())
        .args(&["info", model_path.to_str().unwrap()])
        .output()
        .expect("Failed to run CLI info command");

    assert!(info_output.status.success());

    let stdout = String::from_utf8_lossy(&info_output.stdout);
    assert!(stdout.contains("Support Vector Details"));
    assert!(stdout.contains("Alpha*Y values"));
}

#[test]
fn test_cli_invalid_format_specifications() {
    let test_data = TestDataFiles::new().expect("Failed to create test data");
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let model_path = temp_dir.path().join("model.json");

    // Test training with invalid format
    let output = Command::new(get_cli_binary_path())
        .args(&[
            "train",
            "--data",
            test_data.libsvm_file.path().to_str().unwrap(),
            "--output",
            model_path.to_str().unwrap(),
            "--format",
            "invalid_format",
        ])
        .output()
        .expect("Failed to run CLI command");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Unsupported format") || stderr.contains("invalid"));
}
