//! Comprehensive Kernel Performance Evaluation
//! Tests both Linear and RBF kernels on multiple datasets and synthetic problems

use rsvm::api::SVM;
use rsvm::core::{Sample, SparseVector};
use rsvm::utils::scaling::ScalingMethod;
use std::collections::HashMap;
use std::time::Instant;

#[derive(Debug, Clone)]
struct TestResult {
    accuracy: f64,
    training_time_ms: u64,
    support_vectors: usize,
    test_name: String,
}

#[derive(Debug, Clone)]
struct TestConfig {
    name: String,
    c: f64,
    scaling: Option<ScalingMethod>,
    gamma: Option<f64>, // None for linear kernel
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== RSVM Comprehensive Kernel Performance Evaluation ===");
    println!(
        "Started at: {}",
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
    );
    println!();

    let mut all_results = HashMap::new();

    // Test configurations
    let test_configs = generate_test_configs();

    // Run tests on different problem types
    println!("🔬 Testing on Synthetic XOR Problem (Non-linear)");
    let xor_results = test_on_xor_problem(&test_configs)?;
    all_results.insert("XOR_Problem".to_string(), xor_results);

    println!("\n🔬 Testing on Synthetic Concentric Circles (Non-linear)");
    let circles_results = test_on_concentric_circles(&test_configs)?;
    all_results.insert("Concentric_Circles".to_string(), circles_results);

    println!("\n🔬 Testing on Linearly Separable Problem");
    let linear_results = test_on_linear_problem(&test_configs)?;
    all_results.insert("Linear_Separable".to_string(), linear_results);

    println!("\n🔬 Testing on High-Dimensional Sparse Problem");
    let sparse_results = test_on_sparse_problem(&test_configs)?;
    all_results.insert("High_Dim_Sparse".to_string(), sparse_results);

    // Generate comprehensive report
    generate_report(&all_results)?;

    println!("\n✅ Evaluation completed successfully!");
    println!("📊 Report saved to: comprehensive_kernel_evaluation_report.md");

    Ok(())
}

fn generate_test_configs() -> Vec<TestConfig> {
    let mut configs = Vec::new();

    // Linear kernel tests
    for &c in &[0.1, 1.0, 10.0] {
        for scaling in &[
            None,
            Some(ScalingMethod::MinMax {
                min_val: -1.0,
                max_val: 1.0,
            }),
            Some(ScalingMethod::StandardScore),
        ] {
            configs.push(TestConfig {
                name: format!("Linear_C{}_Scale{}", c, scaling_name(scaling)),
                c,
                scaling: scaling.clone(),
                gamma: None,
            });
        }
    }

    // RBF kernel tests
    for &c in &[0.1, 1.0, 10.0] {
        for &gamma in &[0.1, 1.0, 10.0] {
            for scaling in &[
                None,
                Some(ScalingMethod::MinMax {
                    min_val: -1.0,
                    max_val: 1.0,
                }),
                Some(ScalingMethod::StandardScore),
            ] {
                configs.push(TestConfig {
                    name: format!("RBF_C{}_G{}_Scale{}", c, gamma, scaling_name(scaling)),
                    c,
                    scaling: scaling.clone(),
                    gamma: Some(gamma),
                });
            }
        }
    }

    configs
}

fn scaling_name(scaling: &Option<ScalingMethod>) -> &str {
    match scaling {
        None => "None",
        Some(ScalingMethod::MinMax { .. }) => "MinMax",
        Some(ScalingMethod::StandardScore) => "StdScore",
        Some(ScalingMethod::UnitScale) => "Unit",
    }
}

fn test_on_xor_problem(
    configs: &[TestConfig],
) -> Result<Vec<TestResult>, Box<dyn std::error::Error>> {
    println!("  Generating XOR-like problem (4 quadrants)...");

    // Create XOR-like dataset with more samples
    let mut samples = Vec::new();
    let n_per_quadrant = 50;

    // Positive class: (positive, positive) and (negative, negative) quadrants
    for i in 0..n_per_quadrant {
        let noise_x = (i as f64 / n_per_quadrant as f64) * 0.3 - 0.15;
        let noise_y = ((i * 7) % n_per_quadrant) as f64 / n_per_quadrant as f64 * 0.3 - 0.15;

        // Positive quadrant
        samples.push(Sample::new(
            SparseVector::new(vec![0, 1], vec![1.0 + noise_x, 1.0 + noise_y]),
            1.0,
        ));

        // Negative quadrant
        samples.push(Sample::new(
            SparseVector::new(vec![0, 1], vec![-1.0 + noise_x, -1.0 + noise_y]),
            1.0,
        ));
    }

    // Negative class: (positive, negative) and (negative, positive) quadrants
    for i in 0..n_per_quadrant {
        let noise_x = (i as f64 / n_per_quadrant as f64) * 0.3 - 0.15;
        let noise_y = ((i * 11) % n_per_quadrant) as f64 / n_per_quadrant as f64 * 0.3 - 0.15;

        // Mixed quadrants
        samples.push(Sample::new(
            SparseVector::new(vec![0, 1], vec![1.0 + noise_x, -1.0 + noise_y]),
            -1.0,
        ));

        samples.push(Sample::new(
            SparseVector::new(vec![0, 1], vec![-1.0 + noise_x, 1.0 + noise_y]),
            -1.0,
        ));
    }

    println!("  Generated {} samples for XOR problem", samples.len());
    run_tests_on_samples(&samples, configs, "XOR")
}

fn test_on_concentric_circles(
    configs: &[TestConfig],
) -> Result<Vec<TestResult>, Box<dyn std::error::Error>> {
    println!("  Generating concentric circles problem...");

    let mut samples = Vec::new();
    let n_per_circle = 75;

    // Inner circle (positive class)
    for i in 0..n_per_circle {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / n_per_circle as f64;
        let radius = 0.5 + 0.1 * (i as f64 / n_per_circle as f64 - 0.5);
        let x = radius * angle.cos();
        let y = radius * angle.sin();

        samples.push(Sample::new(SparseVector::new(vec![0, 1], vec![x, y]), 1.0));
    }

    // Outer circle (negative class)
    for i in 0..n_per_circle {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / n_per_circle as f64;
        let radius = 1.5 + 0.2 * (i as f64 / n_per_circle as f64 - 0.5);
        let x = radius * angle.cos();
        let y = radius * angle.sin();

        samples.push(Sample::new(SparseVector::new(vec![0, 1], vec![x, y]), -1.0));
    }

    println!(
        "  Generated {} samples for concentric circles",
        samples.len()
    );
    run_tests_on_samples(&samples, configs, "Circles")
}

fn test_on_linear_problem(
    configs: &[TestConfig],
) -> Result<Vec<TestResult>, Box<dyn std::error::Error>> {
    println!("  Generating linearly separable problem...");

    let mut samples = Vec::new();
    let n_per_class = 100;

    // Class 1: points above the line y = x + 0.5
    for i in 0..n_per_class {
        let x = (i as f64 / n_per_class as f64) * 4.0 - 2.0; // x in [-2, 2]
        let y = x + 0.5 + 0.5 + 0.2 * (i as f64 / n_per_class as f64); // y > x + 0.5

        samples.push(Sample::new(SparseVector::new(vec![0, 1], vec![x, y]), 1.0));
    }

    // Class -1: points below the line y = x + 0.5
    for i in 0..n_per_class {
        let x = (i as f64 / n_per_class as f64) * 4.0 - 2.0;
        let y = x + 0.5 - 0.5 - 0.2 * (i as f64 / n_per_class as f64); // y < x + 0.5

        samples.push(Sample::new(SparseVector::new(vec![0, 1], vec![x, y]), -1.0));
    }

    println!("  Generated {} samples for linear problem", samples.len());
    run_tests_on_samples(&samples, configs, "Linear")
}

fn test_on_sparse_problem(
    configs: &[TestConfig],
) -> Result<Vec<TestResult>, Box<dyn std::error::Error>> {
    println!("  Generating high-dimensional sparse problem...");

    let mut samples = Vec::new();
    let n_per_class = 80;
    let dimensions = 50;

    // Class 1: sparse vectors with features in first half of dimensions
    for i in 0..n_per_class {
        let mut indices = Vec::new();
        let mut values = Vec::new();

        // Randomly select 5-8 features from first half
        let n_features = 5 + (i % 4);
        for j in 0..n_features {
            let idx = (i * 7 + j * 3) % (dimensions / 2);
            let val = 1.0 + 0.1 * (j as f64);
            indices.push(idx);
            values.push(val);
        }

        indices.sort();
        samples.push(Sample::new(SparseVector::new(indices, values), 1.0));
    }

    // Class -1: sparse vectors with features in second half of dimensions
    for i in 0..n_per_class {
        let mut indices = Vec::new();
        let mut values = Vec::new();

        let n_features = 5 + (i % 4);
        for j in 0..n_features {
            let idx = dimensions / 2 + (i * 11 + j * 5) % (dimensions / 2);
            let val = 1.0 + 0.1 * (j as f64);
            indices.push(idx);
            values.push(val);
        }

        indices.sort();
        samples.push(Sample::new(SparseVector::new(indices, values), -1.0));
    }

    println!("  Generated {} samples for sparse problem", samples.len());
    run_tests_on_samples(&samples, configs, "Sparse")
}

fn run_tests_on_samples(
    samples: &[Sample],
    configs: &[TestConfig],
    problem_type: &str,
) -> Result<Vec<TestResult>, Box<dyn std::error::Error>> {
    let mut results = Vec::new();
    let train_size = (samples.len() * 4) / 5; // 80/20 split

    let train_samples = &samples[..train_size];
    let test_samples = &samples[train_size..];

    println!(
        "  Split: {} train, {} test",
        train_samples.len(),
        test_samples.len()
    );

    for (i, config) in configs.iter().enumerate() {
        print!(
            "  [{:2}/{:2}] Testing {:<30}",
            i + 1,
            configs.len(),
            config.name
        );

        let start_time = Instant::now();

        let (accuracy, support_vectors) = match config.gamma {
            None => {
                // Linear kernel
                let mut svm_builder = SVM::new().with_c(config.c);
                if let Some(ref scaling) = config.scaling {
                    svm_builder = svm_builder.with_feature_scaling(scaling.clone());
                }
                match svm_builder.train_samples(train_samples) {
                    Ok(model) => {
                        let correct = test_samples
                            .iter()
                            .map(|sample| model.predict(sample))
                            .zip(test_samples.iter())
                            .filter(|(pred, sample)| pred.label == sample.label)
                            .count();
                        let accuracy = correct as f64 / test_samples.len() as f64;
                        let info = model.info();
                        (Ok(accuracy), info.n_support_vectors)
                    }
                    Err(e) => (Err(e), 0),
                }
            }
            Some(gamma) => {
                // RBF kernel
                let mut svm_builder = SVM::with_rbf(gamma).with_c(config.c);
                if let Some(ref scaling) = config.scaling {
                    svm_builder = svm_builder.with_feature_scaling(scaling.clone());
                }
                match svm_builder.train_samples(train_samples) {
                    Ok(model) => {
                        let correct = test_samples
                            .iter()
                            .map(|sample| model.predict(sample))
                            .zip(test_samples.iter())
                            .filter(|(pred, sample)| pred.label == sample.label)
                            .count();
                        let accuracy = correct as f64 / test_samples.len() as f64;
                        let info = model.info();
                        (Ok(accuracy), info.n_support_vectors)
                    }
                    Err(e) => (Err(e), 0),
                }
            }
        };

        let training_time = start_time.elapsed();

        match accuracy {
            Ok(acc) => {
                results.push(TestResult {
                    accuracy: acc,
                    training_time_ms: training_time.as_millis() as u64,
                    support_vectors,
                    test_name: format!("{}_{}", problem_type, config.name),
                });

                println!(
                    " → {:.1}% ({} ms, {} SVs)",
                    acc * 100.0,
                    training_time.as_millis(),
                    support_vectors
                );
            }
            Err(e) => {
                println!(" → ERROR: {}", e);
                results.push(TestResult {
                    accuracy: 0.0,
                    training_time_ms: 0,
                    support_vectors: 0,
                    test_name: format!("{}_{}", problem_type, config.name),
                });
            }
        }
    }

    Ok(results)
}

fn generate_report(
    all_results: &HashMap<String, Vec<TestResult>>,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create("comprehensive_kernel_evaluation_report.md")?;

    writeln!(file, "# Comprehensive Kernel Performance Evaluation Report")?;
    writeln!(file)?;
    writeln!(file, "## Executive Summary")?;
    writeln!(file)?;
    writeln!(file, "This report compares the performance of Linear and RBF kernels across different problem types:")?;
    writeln!(file, "- **XOR Problem**: Non-linearly separable pattern")?;
    writeln!(
        file,
        "- **Concentric Circles**: Radial non-linear separation"
    )?;
    writeln!(
        file,
        "- **Linear Separable**: Control test for linear patterns"
    )?;
    writeln!(
        file,
        "- **High-Dimensional Sparse**: Real-world-like sparse feature problem"
    )?;
    writeln!(file)?;

    for (problem_name, results) in all_results {
        writeln!(file, "## {} Results", problem_name.replace('_', " "))?;
        writeln!(file)?;
        writeln!(
            file,
            "| Configuration | Accuracy (%) | Training Time (ms) | Support Vectors |"
        )?;
        writeln!(
            file,
            "|---------------|-------------:|------------------:|----------------:|"
        )?;

        let mut sorted_results = results.clone();
        sorted_results.sort_by(|a, b| b.accuracy.partial_cmp(&a.accuracy).unwrap());

        for result in &sorted_results {
            let config_name = result
                .test_name
                .split('_')
                .skip(1)
                .collect::<Vec<_>>()
                .join("_");
            writeln!(
                file,
                "| {} | {:.1} | {} | {} |",
                config_name,
                result.accuracy * 100.0,
                result.training_time_ms,
                result.support_vectors
            )?;
        }
        writeln!(file)?;

        // Find best linear and RBF results
        let best_linear = sorted_results
            .iter()
            .find(|r| r.test_name.contains("Linear_"));
        let best_rbf = sorted_results.iter().find(|r| r.test_name.contains("RBF_"));

        writeln!(file, "### Analysis for {}", problem_name.replace('_', " "))?;
        if let (Some(linear), Some(rbf)) = (best_linear, best_rbf) {
            writeln!(
                file,
                "- **Best Linear**: {:.1}% accuracy",
                linear.accuracy * 100.0
            )?;
            writeln!(
                file,
                "- **Best RBF**: {:.1}% accuracy",
                rbf.accuracy * 100.0
            )?;
            writeln!(
                file,
                "- **RBF Advantage**: {:.1} percentage points",
                (rbf.accuracy - linear.accuracy) * 100.0
            )?;
        }
        writeln!(file)?;
    }

    writeln!(file, "## Overall Analysis")?;
    writeln!(file)?;
    writeln!(file, "### Kernel Performance Summary")?;

    for (problem_name, results) in all_results {
        let linear_results: Vec<_> = results
            .iter()
            .filter(|r| r.test_name.contains("Linear_"))
            .collect();
        let rbf_results: Vec<_> = results
            .iter()
            .filter(|r| r.test_name.contains("RBF_"))
            .collect();

        if !linear_results.is_empty() && !rbf_results.is_empty() {
            let best_linear_acc = linear_results
                .iter()
                .map(|r| r.accuracy)
                .fold(0.0, f64::max);
            let best_rbf_acc = rbf_results.iter().map(|r| r.accuracy).fold(0.0, f64::max);

            writeln!(file, "**{}:**", problem_name.replace('_', " "))?;
            writeln!(
                file,
                "- Linear kernel best: {:.1}%",
                best_linear_acc * 100.0
            )?;
            writeln!(file, "- RBF kernel best: {:.1}%", best_rbf_acc * 100.0)?;
            writeln!(
                file,
                "- Winner: {}",
                if best_rbf_acc > best_linear_acc {
                    "RBF"
                } else {
                    "Linear"
                }
            )?;
            writeln!(file)?;
        }
    }

    writeln!(file, "### Key Insights")?;
    writeln!(file)?;
    writeln!(
        file,
        "1. **Non-linear Problems**: RBF kernels excel on XOR and concentric circle patterns"
    )?;
    writeln!(file, "2. **Linear Problems**: Linear kernels are sufficient and faster for linearly separable data")?;
    writeln!(
        file,
        "3. **High-Dimensional Sparse**: Performance depends on feature distribution and overlap"
    )?;
    writeln!(
        file,
        "4. **Parameter Sensitivity**: Both C and gamma significantly impact RBF performance"
    )?;
    writeln!(
        file,
        "5. **Feature Scaling**: Generally improves performance, especially for RBF kernels"
    )?;
    writeln!(file)?;

    writeln!(file, "### Recommendations")?;
    writeln!(file)?;
    writeln!(
        file,
        "1. **Start with Linear**: For initial experiments and baseline performance"
    )?;
    writeln!(
        file,
        "2. **Use RBF for Non-linear Patterns**: When linear models show poor performance"
    )?;
    writeln!(
        file,
        "3. **Always Scale Features**: Especially critical for RBF kernels"
    )?;
    writeln!(
        file,
        "4. **Tune Hyperparameters**: Use cross-validation for C and gamma selection"
    )?;
    writeln!(
        file,
        "5. **Consider Training Time**: Linear kernels are generally faster"
    )?;
    writeln!(file)?;

    writeln!(file, "---")?;
    writeln!(
        file,
        "*Report generated by RSVM Comprehensive Kernel Evaluation*"
    )?;
    writeln!(
        file,
        "*Date: {}*",
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
    )?;

    Ok(())
}
