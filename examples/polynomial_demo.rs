//! Polynomial Kernel Demonstration
//!
//! This example demonstrates the use of polynomial kernels on different datasets,
//! showing how they can capture non-linear patterns that linear kernels cannot.

use rsvm::api::SVM;
use rsvm::core::{SVMModel, Sample, SparseVector};
use rsvm::kernel::Kernel;
use rsvm::utils::scaling::ScalingMethod;
use rsvm::TrainedModel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Polynomial Kernel Demonstration ===");
    println!();

    // Test 1: Quadratic Separable Data
    println!("📊 Test 1: Quadratic Separable Data");
    test_quadratic_separable()?;
    println!();

    // Test 2: Circular Pattern
    println!("📊 Test 2: Circular Pattern Classification");
    test_circular_pattern()?;
    println!();

    // Test 3: Polynomial Degree Comparison
    println!("📊 Test 3: Polynomial Degree Comparison");
    test_degree_comparison()?;
    println!();

    // Test 4: Feature Scaling Impact
    println!("📊 Test 4: Feature Scaling Impact");
    test_scaling_impact()?;

    Ok(())
}

fn test_quadratic_separable() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Creating quadratic separable dataset...");

    // Create data where y = x² + noise separates classes
    let mut samples = Vec::new();

    // Positive class: points above parabola y = x²
    for i in 0..20 {
        let x = (i as f64 - 10.0) / 5.0; // x in [-2, 2]
        let y = x * x + 0.2 + 0.1 * (i as f64 / 20.0); // slightly above parabola
        samples.push(Sample::new(SparseVector::new(vec![0, 1], vec![x, y]), 1.0));
    }

    // Negative class: points below parabola y = x²
    for i in 0..20 {
        let x = (i as f64 - 10.0) / 5.0;
        let y = x * x - 0.2 - 0.1 * (i as f64 / 20.0); // slightly below parabola
        samples.push(Sample::new(SparseVector::new(vec![0, 1], vec![x, y]), -1.0));
    }

    println!("  Generated {} samples", samples.len());

    // Train linear SVM
    let model_linear = SVM::new()
        .with_c(1.0)
        .with_feature_scaling(ScalingMethod::StandardScore)
        .train_samples(&samples)?;

    // Train quadratic SVM
    let model_quad = SVM::with_quadratic(1.0)
        .with_c(1.0)
        .with_feature_scaling(ScalingMethod::StandardScore)
        .train_samples(&samples)?;

    // Evaluate both models
    let linear_accuracy = evaluate_model(&model_linear, &samples);
    let quad_accuracy = evaluate_model(&model_quad, &samples);

    println!("  Results:");
    println!(
        "    Linear kernel:    {:.1}% accuracy ({} SVs)",
        linear_accuracy * 100.0,
        model_linear.info().n_support_vectors
    );
    println!(
        "    Quadratic kernel: {:.1}% accuracy ({} SVs)",
        quad_accuracy * 100.0,
        model_quad.info().n_support_vectors
    );
    println!(
        "    Improvement:      {:.1} percentage points",
        (quad_accuracy - linear_accuracy) * 100.0
    );

    Ok(())
}

fn test_circular_pattern() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Creating circular pattern dataset...");

    let mut samples = Vec::new();
    let n_per_class = 30;

    // Inner circle (positive class)
    for i in 0..n_per_class {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / n_per_class as f64;
        let radius = 1.0 + 0.2 * (i as f64 / n_per_class as f64 - 0.5); // radius ~1.0
        let x = radius * angle.cos();
        let y = radius * angle.sin();

        samples.push(Sample::new(SparseVector::new(vec![0, 1], vec![x, y]), 1.0));
    }

    // Outer circle (negative class)
    for i in 0..n_per_class {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / n_per_class as f64;
        let radius = 2.0 + 0.2 * (i as f64 / n_per_class as f64 - 0.5); // radius ~2.0
        let x = radius * angle.cos();
        let y = radius * angle.sin();

        samples.push(Sample::new(SparseVector::new(vec![0, 1], vec![x, y]), -1.0));
    }

    println!("  Generated {} samples", samples.len());

    // Train different kernel models
    let model_linear = SVM::new()
        .with_c(10.0)
        .with_feature_scaling(ScalingMethod::MinMax {
            min_val: -1.0,
            max_val: 1.0,
        })
        .train_samples(&samples)?;

    let model_quad = SVM::with_quadratic(1.0)
        .with_c(10.0)
        .with_feature_scaling(ScalingMethod::MinMax {
            min_val: -1.0,
            max_val: 1.0,
        })
        .train_samples(&samples)?;

    let model_cubic = SVM::with_cubic(1.0)
        .with_c(10.0)
        .with_feature_scaling(ScalingMethod::MinMax {
            min_val: -1.0,
            max_val: 1.0,
        })
        .train_samples(&samples)?;

    // Evaluate models
    let linear_acc = evaluate_model(&model_linear, &samples);
    let quad_acc = evaluate_model(&model_quad, &samples);
    let cubic_acc = evaluate_model(&model_cubic, &samples);

    println!("  Results:");
    println!(
        "    Linear:    {:.1}% accuracy ({} SVs)",
        linear_acc * 100.0,
        model_linear.info().n_support_vectors
    );
    println!(
        "    Quadratic: {:.1}% accuracy ({} SVs)",
        quad_acc * 100.0,
        model_quad.info().n_support_vectors
    );
    println!(
        "    Cubic:     {:.1}% accuracy ({} SVs)",
        cubic_acc * 100.0,
        model_cubic.info().n_support_vectors
    );

    // Test on center point (should be positive)
    let center_sample = Sample::new(SparseVector::new(vec![0, 1], vec![0.0, 0.0]), 1.0);
    let linear_pred = model_linear.predict(&center_sample);
    let quad_pred = model_quad.predict(&center_sample);
    let cubic_pred = model_cubic.predict(&center_sample);

    println!("  Center point (0,0) predictions:");
    println!(
        "    Linear:    {} (decision: {:.3})",
        linear_pred.label, linear_pred.decision_value
    );
    println!(
        "    Quadratic: {} (decision: {:.3})",
        quad_pred.label, quad_pred.decision_value
    );
    println!(
        "    Cubic:     {} (decision: {:.3})",
        cubic_pred.label, cubic_pred.decision_value
    );

    Ok(())
}

fn test_degree_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Creating polynomial test dataset...");

    // Create data that benefits from high-degree polynomials
    let mut samples = Vec::new();

    // Create a more complex non-linear pattern
    for i in 0..50 {
        let x = (i as f64 - 25.0) / 12.5; // x in [-2, 2]
        let y_base = x.powi(3) - x; // cubic pattern

        // Positive class: above cubic curve
        let y_pos = y_base + 0.5;
        samples.push(Sample::new(
            SparseVector::new(vec![0, 1], vec![x, y_pos]),
            1.0,
        ));

        // Negative class: below cubic curve
        let y_neg = y_base - 0.5;
        samples.push(Sample::new(
            SparseVector::new(vec![0, 1], vec![x, y_neg]),
            -1.0,
        ));
    }

    println!("  Generated {} samples", samples.len());

    // Test different polynomial degrees
    let degrees = [2, 3, 4, 5]; // Skip degree 1 to avoid type issues
    let mut results = Vec::new();

    // Test linear kernel separately
    let linear_model = SVM::new()
        .with_c(1.0)
        .with_feature_scaling(ScalingMethod::StandardScore)
        .train_samples(&samples)?;
    let linear_accuracy = evaluate_model(&linear_model, &samples);
    let linear_info = linear_model.info();
    results.push((1, linear_accuracy, linear_info.n_support_vectors));

    for &degree in &degrees {
        let model = SVM::with_polynomial(degree, 1.0, 1.0)
            .with_c(1.0)
            .with_feature_scaling(ScalingMethod::StandardScore)
            .train_samples(&samples)?;

        let accuracy = evaluate_model(&model, &samples);
        let info = model.info();

        results.push((degree, accuracy, info.n_support_vectors));
    }

    println!("  Polynomial degree comparison:");
    println!("    Degree | Accuracy | Support Vectors");
    println!("    -------|----------|----------------");
    for (degree, accuracy, svs) in results {
        println!("    {:6} | {:7.1}% | {:14}", degree, accuracy * 100.0, svs);
    }

    Ok(())
}

fn test_scaling_impact() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Creating dataset with different feature scales...");

    let mut samples = Vec::new();

    // Create data with vastly different feature scales
    for i in 0..40 {
        let x1 = (i as f64 - 20.0) / 10.0; // x1 in [-2, 2]
        let x2 = (i as f64 - 20.0) * 100.0; // x2 in [-2000, 2000]

        // Simple linear separation in the scaled space
        let label = if x1 + x2 / 1000.0 > 0.0 { 1.0 } else { -1.0 };

        samples.push(Sample::new(
            SparseVector::new(vec![0, 1], vec![x1, x2]),
            label,
        ));
    }

    println!("  Generated {} samples with mixed scales", samples.len());
    println!("    Feature 1 range: [-2, 2]");
    println!("    Feature 2 range: [-2000, 2000]");

    // Test polynomial with different scaling methods
    let scaling_methods = [
        ("None", None),
        (
            "MinMax",
            Some(ScalingMethod::MinMax {
                min_val: -1.0,
                max_val: 1.0,
            }),
        ),
        ("StandardScore", Some(ScalingMethod::StandardScore)),
        ("UnitScale", Some(ScalingMethod::UnitScale)),
    ];

    println!("  Quadratic kernel with different scaling:");
    println!("    Scaling      | Accuracy | Support Vectors");
    println!("    -------------|----------|----------------");

    for (name, scaling) in scaling_methods {
        let mut svm_builder = SVM::with_quadratic(1.0).with_c(1.0);

        if let Some(scaling_method) = scaling {
            svm_builder = svm_builder.with_feature_scaling(scaling_method);
        }

        let model = svm_builder.train_samples(&samples)?;
        let accuracy = evaluate_model(&model, &samples);
        let info = model.info();

        println!(
            "    {:12} | {:7.1}% | {:14}",
            name,
            accuracy * 100.0,
            info.n_support_vectors
        );
    }

    Ok(())
}

fn evaluate_model<K>(model: &TrainedModel<K>, samples: &[Sample]) -> f64
where
    K: Kernel,
{
    let correct = samples
        .iter()
        .map(|sample| model.predict(sample))
        .zip(samples.iter())
        .filter(|(pred, sample)| pred.label == sample.label)
        .count();

    correct as f64 / samples.len() as f64
}
