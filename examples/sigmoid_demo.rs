//! Sigmoid (Tanh) Kernel Demonstration
//!
//! This example demonstrates the use of Sigmoid kernels for non-linear classification
//! tasks. The Sigmoid kernel is inspired by neural network activation functions and
//! can be effective for problems where RBF/polynomial kernels underperform.
//!
//! Applications demonstrated:
//! - Neural network-inspired classification
//! - Non-linear pattern recognition with bounded outputs
//! - Parameter effects on decision boundaries
//! - Complex decision boundary modeling
//! - Comparison with other kernel types

use rsvm::api::SVM;
use rsvm::core::{Sample, SparseVector};
use rsvm::kernel::Kernel;
use rsvm::TrainedModel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Sigmoid (Tanh) Kernel Demonstration ===");
    println!();

    // Test 1: Neural Network-inspired Classification
    println!("🧠 Test 1: Neural Network-inspired Classification");
    test_neural_network_classification()?;
    println!();

    // Test 2: XOR-like Problem (Non-linearly Separable)
    println!("⚡ Test 2: XOR-like Problem (Non-linearly Separable)");
    test_xor_problem()?;
    println!();

    // Test 3: Parameter Effects on Performance
    println!("⚙️  Test 3: Parameter Effects (Gamma and Bias)");
    test_parameter_effects()?;
    println!();

    // Test 4: Bounded Output Analysis
    println!("📊 Test 4: Bounded Output Analysis (tanh properties)");
    test_bounded_output_analysis()?;
    println!();

    // Test 5: Kernel Comparison on Complex Patterns
    println!("🔬 Test 5: Kernel Comparison on Complex Decision Boundaries");
    test_kernel_comparison()?;

    Ok(())
}

fn test_neural_network_classification() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Creating neural network-inspired classification problem...");

    let mut samples = Vec::new();

    // Class 1: Activation pattern similar to neural network "on" state
    // High positive activations
    for i in 0..20 {
        let noise = 0.1 * ((i as f64 / 20.0) - 0.5);

        // Strong positive signals
        let activation_pattern = vec![
            2.0 + noise, // Strong activation
            1.5 + noise, // Moderate activation
            1.0,         // Base activation
            0.5,         // Weak activation
        ];

        let indices: Vec<usize> = (0..activation_pattern.len()).collect();
        samples.push(Sample::new(
            SparseVector::new(indices, activation_pattern),
            1.0,
        ));
    }

    // Class 2: Suppression pattern similar to neural network "off" state
    // Negative or weak activations
    for i in 0..20 {
        let noise = 0.1 * ((i as f64 / 20.0) - 0.5);

        // Suppressed signals
        let suppression_pattern = vec![
            -1.5 + noise, // Strong suppression
            -1.0 + noise, // Moderate suppression
            -0.5,         // Weak suppression
            0.0,          // No activation
        ];

        let indices: Vec<usize> = (0..suppression_pattern.len()).collect();
        samples.push(Sample::new(
            SparseVector::new(indices, suppression_pattern),
            -1.0,
        ));
    }

    println!(
        "  Generated {} neural network activation samples (4 neurons)",
        samples.len()
    );

    // Train different models with neural network parameters
    let model_sigmoid_neural = SVM::with_sigmoid_neural(4)
        .with_c(1.0)
        .train_samples(&samples)?;

    let model_sigmoid_binary = SVM::with_sigmoid_binary()
        .with_c(1.0)
        .train_samples(&samples)?;

    let model_linear = SVM::new().with_c(1.0).train_samples(&samples)?;

    let model_rbf = SVM::with_rbf(1.0).with_c(1.0).train_samples(&samples)?;

    // Evaluate models
    let neural_acc = evaluate_model(&model_sigmoid_neural, &samples);
    let binary_acc = evaluate_model(&model_sigmoid_binary, &samples);
    let linear_acc = evaluate_model(&model_linear, &samples);
    let rbf_acc = evaluate_model(&model_rbf, &samples);

    println!("  Results on neural network activation patterns:");
    println!(
        "    Sigmoid (neural):     {:.1}% accuracy ({} SVs)",
        neural_acc * 100.0,
        model_sigmoid_neural.info().n_support_vectors
    );
    println!(
        "    Sigmoid (binary):     {:.1}% accuracy ({} SVs)",
        binary_acc * 100.0,
        model_sigmoid_binary.info().n_support_vectors
    );
    println!(
        "    Linear kernel:        {:.1}% accuracy ({} SVs)",
        linear_acc * 100.0,
        model_linear.info().n_support_vectors
    );
    println!(
        "    RBF kernel:           {:.1}% accuracy ({} SVs)",
        rbf_acc * 100.0,
        model_rbf.info().n_support_vectors
    );
    println!(
        "    Neural sigmoid gain:  {:.1} percentage points over linear",
        (neural_acc - linear_acc) * 100.0
    );

    // Test on ambiguous activation (mixed signals)
    let mixed_pattern = vec![0.5, -0.2, 0.8, -0.1];
    let indices: Vec<usize> = (0..mixed_pattern.len()).collect();
    let mixed_sample = Sample::new(SparseVector::new(indices, mixed_pattern), 0.0);

    let neural_pred = model_sigmoid_neural.predict(&mixed_sample);
    println!(
        "  Mixed activation prediction: {} (confidence: {:.3})",
        if neural_pred.label > 0.0 {
            "Activate"
        } else {
            "Suppress"
        },
        neural_pred.decision_value.abs()
    );

    Ok(())
}

fn test_xor_problem() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Creating XOR-like problem (classic non-linear challenge)...");

    // XOR problem: same signs -> negative, different signs -> positive
    let samples = vec![
        // Both positive -> Class -1
        Sample::new(SparseVector::new(vec![0, 1], vec![1.0, 1.0]), -1.0),
        Sample::new(SparseVector::new(vec![0, 1], vec![1.2, 0.8]), -1.0),
        Sample::new(SparseVector::new(vec![0, 1], vec![0.9, 1.1]), -1.0),
        // Both negative -> Class -1
        Sample::new(SparseVector::new(vec![0, 1], vec![-1.0, -1.0]), -1.0),
        Sample::new(SparseVector::new(vec![0, 1], vec![-1.2, -0.8]), -1.0),
        Sample::new(SparseVector::new(vec![0, 1], vec![-0.9, -1.1]), -1.0),
        // Positive/Negative -> Class +1
        Sample::new(SparseVector::new(vec![0, 1], vec![1.0, -1.0]), 1.0),
        Sample::new(SparseVector::new(vec![0, 1], vec![1.1, -0.9]), 1.0),
        Sample::new(SparseVector::new(vec![0, 1], vec![0.8, -1.2]), 1.0),
        // Negative/Positive -> Class +1
        Sample::new(SparseVector::new(vec![0, 1], vec![-1.0, 1.0]), 1.0),
        Sample::new(SparseVector::new(vec![0, 1], vec![-1.1, 0.9]), 1.0),
        Sample::new(SparseVector::new(vec![0, 1], vec![-0.8, 1.2]), 1.0),
    ];

    println!("  Generated {} XOR pattern samples", samples.len());

    // Train different kernels on XOR problem
    let model_linear = SVM::new().with_c(10.0).train_samples(&samples)?;

    let model_sigmoid_default = SVM::with_sigmoid_default()
        .with_c(10.0)
        .train_samples(&samples)?;

    let model_sigmoid_zero = SVM::with_sigmoid_zero_bias(0.1)
        .with_c(10.0)
        .train_samples(&samples)?;

    let model_rbf = SVM::with_rbf(2.0).with_c(10.0).train_samples(&samples)?;

    let model_poly = SVM::with_quadratic(1.0)
        .with_c(10.0)
        .train_samples(&samples)?;

    // Evaluate models
    let linear_acc = evaluate_model(&model_linear, &samples);
    let sigmoid_default_acc = evaluate_model(&model_sigmoid_default, &samples);
    let sigmoid_zero_acc = evaluate_model(&model_sigmoid_zero, &samples);
    let rbf_acc = evaluate_model(&model_rbf, &samples);
    let poly_acc = evaluate_model(&model_poly, &samples);

    println!("  Results on XOR-like problem:");
    println!(
        "    Linear kernel:        {:.1}% accuracy ({} SVs) - Expected to struggle",
        linear_acc * 100.0,
        model_linear.info().n_support_vectors
    );
    println!(
        "    Sigmoid (default):    {:.1}% accuracy ({} SVs)",
        sigmoid_default_acc * 100.0,
        model_sigmoid_default.info().n_support_vectors
    );
    println!(
        "    Sigmoid (zero bias):  {:.1}% accuracy ({} SVs)",
        sigmoid_zero_acc * 100.0,
        model_sigmoid_zero.info().n_support_vectors
    );
    println!(
        "    RBF kernel:           {:.1}% accuracy ({} SVs)",
        rbf_acc * 100.0,
        model_rbf.info().n_support_vectors
    );
    println!(
        "    Polynomial kernel:    {:.1}% accuracy ({} SVs)",
        poly_acc * 100.0,
        model_poly.info().n_support_vectors
    );

    // Test prediction on origin (most ambiguous point)
    let origin_sample = Sample::new(SparseVector::new(vec![0, 1], vec![0.0, 0.0]), 0.0);
    let sigmoid_pred = model_sigmoid_zero.predict(&origin_sample);
    println!(
        "  Origin prediction: {} (decision: {:.3})",
        if sigmoid_pred.label > 0.0 {
            "Different signs"
        } else {
            "Same signs"
        },
        sigmoid_pred.decision_value
    );

    Ok(())
}

fn test_parameter_effects() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Analyzing sigmoid parameter effects...");

    // Create a simple dataset to study parameter effects
    let samples = vec![
        Sample::new(SparseVector::new(vec![0, 1], vec![2.0, 1.0]), 1.0),
        Sample::new(SparseVector::new(vec![0, 1], vec![1.0, 2.0]), 1.0),
        Sample::new(SparseVector::new(vec![0, 1], vec![-2.0, -1.0]), -1.0),
        Sample::new(SparseVector::new(vec![0, 1], vec![-1.0, -2.0]), -1.0),
        Sample::new(SparseVector::new(vec![0, 1], vec![1.5, 1.5]), 1.0),
        Sample::new(SparseVector::new(vec![0, 1], vec![-1.5, -1.5]), -1.0),
    ];

    println!(
        "  Generated {} samples for parameter analysis",
        samples.len()
    );

    // Test different gamma values (steepness of sigmoid)
    let gammas = [0.01, 0.1, 0.5, 1.0];
    println!("  Gamma effects (bias = -1.0):");
    println!("    Gamma  | Accuracy | Support Vectors");
    println!("    -------|----------|----------------");

    for &gamma in &gammas {
        let model = SVM::with_sigmoid(gamma, -1.0)
            .with_c(10.0)
            .train_samples(&samples)?;

        let accuracy = evaluate_model(&model, &samples);
        println!(
            "    {:6} | {:7.1}% | {:14}",
            gamma,
            accuracy * 100.0,
            model.info().n_support_vectors
        );
    }

    // Test different bias values (shift of sigmoid)
    let biases = [-2.0, -1.0, 0.0, 1.0];
    println!("\n  Bias effects (gamma = 0.1):");
    println!("    Bias   | Accuracy | Support Vectors");
    println!("    -------|----------|----------------");

    for &bias in &biases {
        let model = SVM::with_sigmoid(0.1, bias)
            .with_c(10.0)
            .train_samples(&samples)?;

        let accuracy = evaluate_model(&model, &samples);
        println!(
            "    {:6} | {:7.1}% | {:14}",
            bias,
            accuracy * 100.0,
            model.info().n_support_vectors
        );
    }

    // Demonstrate decision value ranges
    let test_points = vec![
        (vec![3.0, 3.0], "Strong positive"),
        (vec![0.5, 0.5], "Weak positive"),
        (vec![0.0, 0.0], "Origin"),
        (vec![-0.5, -0.5], "Weak negative"),
        (vec![-3.0, -3.0], "Strong negative"),
    ];

    let model_sigmoid = SVM::with_sigmoid(0.1, -1.0)
        .with_c(10.0)
        .train_samples(&samples)?;

    println!("\n  Decision values (bounded by tanh):");
    for (point, description) in test_points {
        let indices: Vec<usize> = (0..point.len()).collect();
        let sample = Sample::new(SparseVector::new(indices, point.clone()), 0.0);
        let pred = model_sigmoid.predict(&sample);
        println!(
            "    {:14}: decision = {:6.3} (in [-1,1])",
            description, pred.decision_value
        );
    }

    Ok(())
}

fn test_bounded_output_analysis() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Analyzing bounded output properties of tanh kernel...");

    // Create dataset with extreme values to test bounds
    let mut samples = Vec::new();

    // Class 1: Large positive values
    for scale in [1.0, 10.0, 100.0] {
        samples.push(Sample::new(
            SparseVector::new(vec![0, 1], vec![scale, scale]),
            1.0,
        ));
    }

    // Class 2: Large negative values
    for scale in [1.0, 10.0, 100.0] {
        samples.push(Sample::new(
            SparseVector::new(vec![0, 1], vec![-scale, -scale]),
            -1.0,
        ));
    }

    println!("  Generated {} samples with extreme values", samples.len());

    // Compare sigmoid with unbounded kernels
    let model_sigmoid = SVM::with_sigmoid(0.01, 0.0)
        .with_c(1.0)
        .train_samples(&samples)?;

    let model_linear = SVM::new().with_c(1.0).train_samples(&samples)?;

    let model_poly = SVM::with_quadratic(0.01)
        .with_c(1.0)
        .train_samples(&samples)?;

    println!("  Kernel output ranges on extreme values:");

    // Test on increasingly extreme values
    let test_scales = [1.0, 5.0, 10.0, 50.0, 100.0];

    for &scale in &test_scales {
        let extreme_positive = Sample::new(SparseVector::new(vec![0, 1], vec![scale, scale]), 0.0);

        let sigmoid_pred = model_sigmoid.predict(&extreme_positive);
        let linear_pred = model_linear.predict(&extreme_positive);
        let poly_pred = model_poly.predict(&extreme_positive);

        println!(
            "    Scale {:5.0}: Sigmoid={:6.3} | Linear={:8.1} | Poly={:8.1}",
            scale,
            sigmoid_pred.decision_value,
            linear_pred.decision_value,
            poly_pred.decision_value
        );
    }

    println!("\n  Note: Sigmoid values stay bounded in [-1,1] while others grow unbounded");

    Ok(())
}

fn test_kernel_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Creating comprehensive dataset for kernel comparison...");

    let mut samples = Vec::new();

    // Create complex non-linear patterns
    for i in 0..25 {
        let t = (i as f64) / 25.0 * 2.0 * std::f64::consts::PI;

        // Class 1: Spiral pattern (outer)
        let r1 = 2.0 + 0.5 * t;
        let x1 = r1 * t.cos();
        let y1 = r1 * t.sin();
        samples.push(Sample::new(
            SparseVector::new(vec![0, 1], vec![x1, y1]),
            1.0,
        ));

        // Class 2: Spiral pattern (inner)
        let r2 = 1.0 + 0.3 * t;
        let x2 = r2 * (t + std::f64::consts::PI).cos();
        let y2 = r2 * (t + std::f64::consts::PI).sin();
        samples.push(Sample::new(
            SparseVector::new(vec![0, 1], vec![x2, y2]),
            -1.0,
        ));
    }

    println!("  Generated {} samples with spiral patterns", samples.len());

    // Test comprehensive kernel comparison
    let kernels = [
        ("Linear", "linear"),
        ("RBF (γ=0.1)", "rbf_01"),
        ("RBF (γ=1.0)", "rbf_10"),
        ("Polynomial (d=2)", "poly"),
        ("Polynomial (d=3)", "poly3"),
        ("Sigmoid (default)", "sigmoid_default"),
        ("Sigmoid (neural)", "sigmoid_neural"),
        ("Sigmoid (zero bias)", "sigmoid_zero"),
    ];

    println!("  Comprehensive kernel comparison on spiral patterns:");
    println!("    Kernel                | Accuracy | Support Vectors");
    println!("    ----------------------|----------|----------------");

    for (name, kernel_type) in kernels {
        let model: Box<dyn ModelEvaluator> = match kernel_type {
            "linear" => Box::new(SVM::new().with_c(1.0).train_samples(&samples)?),
            "rbf_01" => Box::new(SVM::with_rbf(0.1).with_c(1.0).train_samples(&samples)?),
            "rbf_10" => Box::new(SVM::with_rbf(1.0).with_c(1.0).train_samples(&samples)?),
            "poly" => Box::new(
                SVM::with_quadratic(0.1)
                    .with_c(1.0)
                    .train_samples(&samples)?,
            ),
            "poly3" => Box::new(SVM::with_cubic(0.1).with_c(1.0).train_samples(&samples)?),
            "sigmoid_default" => Box::new(
                SVM::with_sigmoid_default()
                    .with_c(1.0)
                    .train_samples(&samples)?,
            ),
            "sigmoid_neural" => Box::new(
                SVM::with_sigmoid_neural(2)
                    .with_c(1.0)
                    .train_samples(&samples)?,
            ),
            "sigmoid_zero" => Box::new(
                SVM::with_sigmoid_zero_bias(0.1)
                    .with_c(1.0)
                    .train_samples(&samples)?,
            ),
            _ => unreachable!(),
        };

        let accuracy = model.evaluate(&samples);
        let n_svs = model.n_support_vectors();

        println!("    {:21} | {:7.1}% | {:14}", name, accuracy * 100.0, n_svs);
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
