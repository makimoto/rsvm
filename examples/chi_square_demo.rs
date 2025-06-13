//! Chi-Square Kernel Demonstration
//!
//! This example demonstrates the use of Chi-square kernels on histogram and distribution data,
//! showing their superior performance over linear kernels for this type of data.
//!
//! The Chi-square kernel is particularly effective for:
//! - Computer vision tasks with histogram features (color histograms, HOG, SIFT)
//! - Text analysis with bag-of-words representations
//! - Bioinformatics with frequency data
//! - Any data naturally represented as probability distributions

use rsvm::api::SVM;
use rsvm::core::{Sample, SparseVector};
use rsvm::kernel::Kernel;
use rsvm::utils::scaling::ScalingMethod;
use rsvm::TrainedModel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Chi-Square Kernel Demonstration ===");
    println!();

    // Test 1: Color Histogram Classification
    println!("📊 Test 1: Color Histogram Classification");
    test_color_histograms()?;
    println!();

    // Test 2: Text Frequency Analysis
    println!("📊 Test 2: Term Frequency Analysis");
    test_text_frequencies()?;
    println!();

    // Test 3: Gamma Parameter Impact
    println!("📊 Test 3: Gamma Parameter Impact");
    test_gamma_impact()?;
    println!();

    // Test 4: Comparison with Other Kernels
    println!("📊 Test 4: Kernel Comparison on Histogram Data");
    test_kernel_comparison()?;

    Ok(())
}

fn test_color_histograms() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Creating color histogram dataset (simulating RGB histograms)...");

    let mut samples = Vec::new();

    // Class 1: Red-dominant histograms (sunset/fire images)
    // High values in red bins, lower in green/blue
    for i in 0..25 {
        let noise = 0.1 * (i as f64 / 25.0 - 0.5);
        let red_hist = vec![40.0 + noise, 35.0, 20.0, 5.0]; // Red bins
        let green_hist = vec![10.0, 15.0 + noise, 20.0, 25.0]; // Green bins
        let blue_hist = vec![5.0, 8.0, 12.0 + noise, 15.0]; // Blue bins

        let mut histogram = Vec::new();
        histogram.extend(red_hist);
        histogram.extend(green_hist);
        histogram.extend(blue_hist);

        let indices: Vec<usize> = (0..histogram.len()).collect();
        samples.push(Sample::new(SparseVector::new(indices, histogram), 1.0));
    }

    // Class 2: Blue-dominant histograms (sky/water images)
    // High values in blue bins, lower in red/green
    for i in 0..25 {
        let noise = 0.1 * (i as f64 / 25.0 - 0.5);
        let red_hist = vec![5.0, 8.0 + noise, 12.0, 15.0]; // Red bins
        let green_hist = vec![10.0 + noise, 15.0, 18.0, 22.0]; // Green bins
        let blue_hist = vec![45.0, 40.0 + noise, 25.0, 10.0]; // Blue bins

        let mut histogram = Vec::new();
        histogram.extend(red_hist);
        histogram.extend(green_hist);
        histogram.extend(blue_hist);

        let indices: Vec<usize> = (0..histogram.len()).collect();
        samples.push(Sample::new(SparseVector::new(indices, histogram), -1.0));
    }

    println!(
        "  Generated {} color histogram samples (12 bins each)",
        samples.len()
    );

    // Train different models
    let model_linear = SVM::new().with_c(1.0).train_samples(&samples)?;

    let model_chi2 = SVM::with_chi_square_cv() // Optimized for computer vision
        .with_c(1.0)
        .train_samples(&samples)?;

    // Evaluate models
    let linear_acc = evaluate_model(&model_linear, &samples);
    let chi2_acc = evaluate_model(&model_chi2, &samples);

    println!("  Results:");
    println!(
        "    Linear kernel:     {:.1}% accuracy ({} SVs)",
        linear_acc * 100.0,
        model_linear.info().n_support_vectors
    );
    println!(
        "    Chi-square kernel: {:.1}% accuracy ({} SVs)",
        chi2_acc * 100.0,
        model_chi2.info().n_support_vectors
    );
    println!(
        "    Improvement:       {:.1} percentage points",
        (chi2_acc - linear_acc) * 100.0
    );

    // Test on ambiguous histogram (mixed colors)
    let mixed_histogram = vec![
        25.0, 25.0, 20.0, 10.0, // Red bins
        20.0, 22.0, 18.0, 15.0, // Green bins
        20.0, 25.0, 22.0, 18.0, // Blue bins
    ];
    let indices: Vec<usize> = (0..mixed_histogram.len()).collect();
    let mixed_sample = Sample::new(SparseVector::new(indices, mixed_histogram), 0.0);

    let linear_pred = model_linear.predict(&mixed_sample);
    let chi2_pred = model_chi2.predict(&mixed_sample);

    println!("  Mixed color histogram prediction:");
    println!(
        "    Linear:     {} (confidence: {:.3})",
        if linear_pred.label > 0.0 {
            "Red-dominant"
        } else {
            "Blue-dominant"
        },
        linear_pred.decision_value.abs()
    );
    println!(
        "    Chi-square: {} (confidence: {:.3})",
        if chi2_pred.label > 0.0 {
            "Red-dominant"
        } else {
            "Blue-dominant"
        },
        chi2_pred.decision_value.abs()
    );

    Ok(())
}

fn test_text_frequencies() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Creating term frequency dataset (simulating document classification)...");

    let mut samples = Vec::new();

    // Class 1: Technical documents (high frequency of technical terms)
    // Terms: [algorithm, data, machine, learning, neural, network, model, training]
    for i in 0..20 {
        let base_freq = 0.1 * (i as f64 / 20.0);
        let tech_frequencies = vec![
            15.0 + base_freq, // algorithm
            20.0 + base_freq, // data
            12.0 + base_freq, // machine
            18.0 + base_freq, // learning
            8.0 + base_freq,  // neural
            10.0 + base_freq, // network
            16.0 + base_freq, // model
            14.0 + base_freq, // training
        ];

        let indices: Vec<usize> = (0..tech_frequencies.len()).collect();
        samples.push(Sample::new(
            SparseVector::new(indices, tech_frequencies),
            1.0,
        ));
    }

    // Class 2: General news documents (different term distribution)
    // Same vocabulary but different frequencies
    for i in 0..20 {
        let base_freq = 0.1 * (i as f64 / 20.0);
        let news_frequencies = vec![
            2.0 + base_freq, // algorithm (rare)
            8.0 + base_freq, // data (common)
            1.0 + base_freq, // machine (rare)
            3.0 + base_freq, // learning (rare)
            0.5 + base_freq, // neural (very rare)
            5.0 + base_freq, // network (moderate)
            4.0 + base_freq, // model (moderate)
            2.0 + base_freq, // training (rare)
        ];

        let indices: Vec<usize> = (0..news_frequencies.len()).collect();
        samples.push(Sample::new(
            SparseVector::new(indices, news_frequencies),
            -1.0,
        ));
    }

    println!(
        "  Generated {} term frequency samples (8 terms each)",
        samples.len()
    );

    // Train different models
    let model_linear = SVM::new().with_c(1.0).train_samples(&samples)?;

    let model_chi2 = SVM::with_chi_square_text() // Optimized for text analysis
        .with_c(1.0)
        .train_samples(&samples)?;

    // Evaluate models
    let linear_acc = evaluate_model(&model_linear, &samples);
    let chi2_acc = evaluate_model(&model_chi2, &samples);

    println!("  Results:");
    println!(
        "    Linear kernel:     {:.1}% accuracy ({} SVs)",
        linear_acc * 100.0,
        model_linear.info().n_support_vectors
    );
    println!(
        "    Chi-square kernel: {:.1}% accuracy ({} SVs)",
        chi2_acc * 100.0,
        model_chi2.info().n_support_vectors
    );
    println!(
        "    Improvement:       {:.1} percentage points",
        (chi2_acc - linear_acc) * 100.0
    );

    Ok(())
}

fn test_gamma_impact() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Creating histogram dataset to test gamma sensitivity...");

    let mut samples = Vec::new();

    // Create two distinct histogram patterns
    for i in 0..30 {
        let noise = 0.05 * (i as f64 / 30.0 - 0.5);

        // Pattern 1: Exponential decay
        let pattern1 = vec![
            50.0 + noise,
            25.0 + noise,
            12.5 + noise,
            6.25 + noise,
            3.0 + noise,
        ];

        // Pattern 2: Bell curve
        let pattern2 = vec![
            5.0 + noise,
            15.0 + noise,
            40.0 + noise,
            15.0 + noise,
            5.0 + noise,
        ];

        let indices: Vec<usize> = (0..5).collect();
        samples.push(Sample::new(
            SparseVector::new(indices.clone(), pattern1),
            1.0,
        ));
        samples.push(Sample::new(SparseVector::new(indices, pattern2), -1.0));
    }

    println!(
        "  Generated {} samples with distinct histogram patterns",
        samples.len()
    );

    // Test different gamma values
    let gamma_values = [0.1, 0.5, 1.0, 2.0, 5.0];
    println!("  Testing gamma sensitivity:");
    println!("    Gamma | Accuracy | Support Vectors");
    println!("    ------|----------|----------------");

    for &gamma in &gamma_values {
        let model = SVM::with_chi_square(gamma)
            .with_c(1.0)
            .train_samples(&samples)?;

        let accuracy = evaluate_model(&model, &samples);
        let info = model.info();

        println!(
            "    {:5.1} | {:7.1}% | {:14}",
            gamma,
            accuracy * 100.0,
            info.n_support_vectors
        );
    }

    Ok(())
}

fn test_kernel_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Creating comprehensive histogram dataset for kernel comparison...");

    let mut samples = Vec::new();

    // Create more complex histogram patterns that should favor Chi-square
    for i in 0..40 {
        let t = i as f64 / 40.0;

        // Class 1: Gamma distribution-like histograms
        let gamma_hist = vec![
            30.0 * (2.0 * t).exp() * (-2.0 * t).exp(),
            25.0 * (1.5 * t).exp() * (-1.5 * t).exp(),
            20.0 * t.exp() * (-t).exp(),
            15.0 * (0.5 * t).exp() * (-0.5 * t).exp(),
            10.0 * (0.1 * t).exp() * (-0.1 * t).exp(),
            5.0,
        ];

        // Class 2: Uniform-like histograms with noise
        let uniform_hist = vec![
            20.0 + 2.0 * (t * 10.0).sin(),
            18.0 + 2.0 * (t * 12.0).sin(),
            22.0 + 2.0 * (t * 8.0).sin(),
            19.0 + 2.0 * (t * 15.0).sin(),
            21.0 + 2.0 * (t * 6.0).sin(),
            20.0 + 2.0 * (t * 9.0).sin(),
        ];

        let indices: Vec<usize> = (0..6).collect();
        samples.push(Sample::new(
            SparseVector::new(indices.clone(), gamma_hist),
            1.0,
        ));
        samples.push(Sample::new(SparseVector::new(indices, uniform_hist), -1.0));
    }

    println!(
        "  Generated {} samples with complex histogram patterns",
        samples.len()
    );

    // Train different kernel models
    let kernels = [
        ("Linear", "linear"),
        ("RBF (γ=1.0)", "rbf"),
        ("Chi-square (γ=1.0)", "chi2"),
        ("Chi-square (CV)", "chi2_cv"),
        ("Polynomial (d=2)", "poly"),
    ];

    println!("  Kernel comparison results:");
    println!("    Kernel              | Accuracy | Support Vectors");
    println!("    --------------------|----------|----------------");

    for (name, kernel_type) in kernels {
        let model: Box<dyn ModelEvaluator> = match kernel_type {
            "linear" => Box::new(
                SVM::new()
                    .with_c(1.0)
                    .with_feature_scaling(ScalingMethod::UnitScale)
                    .train_samples(&samples)?,
            ),
            "rbf" => Box::new(
                SVM::with_rbf(1.0)
                    .with_c(1.0)
                    .with_feature_scaling(ScalingMethod::UnitScale)
                    .train_samples(&samples)?,
            ),
            "chi2" => Box::new(
                SVM::with_chi_square(1.0)
                    .with_c(1.0)
                    .train_samples(&samples)?,
            ),
            "chi2_cv" => Box::new(
                SVM::with_chi_square_cv()
                    .with_c(1.0)
                    .train_samples(&samples)?,
            ),
            "poly" => Box::new(
                SVM::with_quadratic(1.0)
                    .with_c(1.0)
                    .with_feature_scaling(ScalingMethod::UnitScale)
                    .train_samples(&samples)?,
            ),
            _ => unreachable!(),
        };

        let accuracy = model.evaluate(&samples);
        let n_svs = model.n_support_vectors();

        println!("    {:19} | {:7.1}% | {:14}", name, accuracy * 100.0, n_svs);
    }

    Ok(())
}

// Helper trait to unify different kernel types for evaluation
trait ModelEvaluator {
    fn evaluate(&self, samples: &[Sample]) -> f64;
    fn n_support_vectors(&self) -> usize;
}

impl<K: Kernel> ModelEvaluator for TrainedModel<K> {
    fn evaluate(&self, samples: &[Sample]) -> f64 {
        let correct = samples
            .iter()
            .map(|sample| self.predict(sample))
            .zip(samples.iter())
            .filter(|(pred, sample)| pred.label == sample.label)
            .count();

        correct as f64 / samples.len() as f64
    }

    fn n_support_vectors(&self) -> usize {
        self.info().n_support_vectors
    }
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
