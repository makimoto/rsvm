//! Demo showing RBF kernel capabilities on non-linear data

use rsvm::api::SVM;
use rsvm::core::{Sample, SparseVector};
use rsvm::utils::scaling::ScalingMethod;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== RBF Kernel Demo ===");

    // Create XOR-like problem (non-linearly separable)
    let samples = vec![
        // Positive class: opposite corners
        Sample::new(SparseVector::new(vec![0, 1], vec![1.0, 1.0]), 1.0),
        Sample::new(SparseVector::new(vec![0, 1], vec![-1.0, -1.0]), 1.0),
        Sample::new(SparseVector::new(vec![0, 1], vec![0.8, 0.9]), 1.0),
        Sample::new(SparseVector::new(vec![0, 1], vec![-0.9, -0.8]), 1.0),
        // Negative class: other two corners
        Sample::new(SparseVector::new(vec![0, 1], vec![1.0, -1.0]), -1.0),
        Sample::new(SparseVector::new(vec![0, 1], vec![-1.0, 1.0]), -1.0),
        Sample::new(SparseVector::new(vec![0, 1], vec![0.9, -0.8]), -1.0),
        Sample::new(SparseVector::new(vec![0, 1], vec![-0.8, 0.9]), -1.0),
    ];

    println!("Training data points: {}", samples.len());

    // Train Linear SVM
    println!("\n--- Linear SVM ---");
    let model_linear = SVM::new()
        .with_c(10.0)
        .with_feature_scaling(ScalingMethod::MinMax {
            min_val: -1.0,
            max_val: 1.0,
        })
        .train_samples(&samples)?;

    // Train RBF SVM with different gammas
    println!("\n--- RBF SVM (gamma=1.0) ---");
    let model_rbf_1 = SVM::with_rbf(1.0)
        .with_c(10.0)
        .with_feature_scaling(ScalingMethod::MinMax {
            min_val: -1.0,
            max_val: 1.0,
        })
        .train_samples(&samples)?;

    println!("\n--- RBF SVM (gamma=10.0) ---");
    let model_rbf_10 = SVM::with_rbf(10.0)
        .with_c(10.0)
        .with_feature_scaling(ScalingMethod::MinMax {
            min_val: -1.0,
            max_val: 1.0,
        })
        .train_samples(&samples)?;

    println!("\n--- RBF SVM (auto gamma) ---");
    let model_rbf_auto = SVM::with_rbf_auto(2)
        .with_c(10.0)
        .with_feature_scaling(ScalingMethod::MinMax {
            min_val: -1.0,
            max_val: 1.0,
        })
        .train_samples(&samples)?;

    // Evaluate on training data
    println!("\n=== Training Accuracy ===");

    let linear_accuracy = model_linear
        .predict_batch(&samples)
        .iter()
        .zip(samples.iter())
        .filter(|(pred, sample)| pred.label == sample.label)
        .count() as f64
        / samples.len() as f64;

    let rbf_1_accuracy = model_rbf_1
        .predict_batch(&samples)
        .iter()
        .zip(samples.iter())
        .filter(|(pred, sample)| pred.label == sample.label)
        .count() as f64
        / samples.len() as f64;

    let rbf_10_accuracy = model_rbf_10
        .predict_batch(&samples)
        .iter()
        .zip(samples.iter())
        .filter(|(pred, sample)| pred.label == sample.label)
        .count() as f64
        / samples.len() as f64;

    let rbf_auto_accuracy = model_rbf_auto
        .predict_batch(&samples)
        .iter()
        .zip(samples.iter())
        .filter(|(pred, sample)| pred.label == sample.label)
        .count() as f64
        / samples.len() as f64;

    println!(
        "Linear SVM:     {:.1}% ({} support vectors)",
        linear_accuracy * 100.0,
        model_linear.info().n_support_vectors
    );
    println!(
        "RBF γ=1.0:      {:.1}% ({} support vectors)",
        rbf_1_accuracy * 100.0,
        model_rbf_1.info().n_support_vectors
    );
    println!(
        "RBF γ=10.0:     {:.1}% ({} support vectors)",
        rbf_10_accuracy * 100.0,
        model_rbf_10.info().n_support_vectors
    );
    println!(
        "RBF auto γ=0.5: {:.1}% ({} support vectors)",
        rbf_auto_accuracy * 100.0,
        model_rbf_auto.info().n_support_vectors
    );

    // Test predictions on new points
    println!("\n=== Test Predictions ===");
    let test_points = vec![
        (vec![0.0, 0.0], "Center"),
        (vec![0.5, 0.5], "Positive quadrant"),
        (vec![0.5, -0.5], "Mixed quadrant"),
        (vec![-0.5, 0.5], "Mixed quadrant"),
        (vec![-0.5, -0.5], "Negative quadrant"),
    ];

    for (coords, desc) in test_points {
        let test_sample = Sample::new(SparseVector::new(vec![0, 1], coords.clone()), 0.0);

        let linear_pred = model_linear.predict(&test_sample);
        let rbf_1_pred = model_rbf_1.predict(&test_sample);
        let rbf_10_pred = model_rbf_10.predict(&test_sample);
        let rbf_auto_pred = model_rbf_auto.predict(&test_sample);

        println!("\nPoint {:?} ({})", coords, desc);
        println!(
            "  Linear:    {:4.0} (confidence: {:.3})",
            linear_pred.label,
            linear_pred.confidence()
        );
        println!(
            "  RBF γ=1.0: {:4.0} (confidence: {:.3})",
            rbf_1_pred.label,
            rbf_1_pred.confidence()
        );
        println!(
            "  RBF γ=10:  {:4.0} (confidence: {:.3})",
            rbf_10_pred.label,
            rbf_10_pred.confidence()
        );
        println!(
            "  RBF auto:  {:4.0} (confidence: {:.3})",
            rbf_auto_pred.label,
            rbf_auto_pred.confidence()
        );
    }

    println!("\n=== Summary ===");
    println!("RBF kernels should perform better on this XOR-like problem");
    println!("Linear SVM cannot separate XOR patterns (expected ~50% accuracy)");
    println!("RBF SVM with appropriate gamma should achieve higher accuracy");

    Ok(())
}
