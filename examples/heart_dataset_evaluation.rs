//! Comprehensive Heart Dataset Evaluation
//! Tests both Linear and RBF kernels on the heart disease dataset

use rsvm::api::SVM;
use rsvm::core::Dataset;
use rsvm::data::LibSVMDataset;
use rsvm::utils::scaling::ScalingMethod;
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
    println!("=== RSVM Heart Dataset Performance Evaluation ===");
    println!("Using heart.libsvm dataset for comprehensive kernel comparison");
    println!();

    // Load heart dataset
    let dataset = LibSVMDataset::from_file("heart.libsvm")?;
    println!("📊 Dataset loaded:");
    println!("  - Samples: {}", dataset.len());
    println!("  - Dimensions: {}", dataset.dim());

    // Check class distribution
    let labels = dataset.get_labels();
    let positive = labels.iter().filter(|&&l| l > 0.0).count();
    let negative = labels.len() - positive;
    println!(
        "  - Class distribution: {} positive, {} negative",
        positive, negative
    );
    println!();

    // Test configurations
    let test_configs = generate_test_configs();
    println!("🧪 Testing {} configurations...", test_configs.len());

    // Split dataset
    let samples: Vec<_> = (0..dataset.len()).map(|i| dataset.get_sample(i)).collect();
    let train_size = (samples.len() * 4) / 5; // 80/20 split
    let train_samples = &samples[..train_size];
    let test_samples = &samples[train_size..];

    println!(
        "📈 Train/Test split: {} / {}",
        train_samples.len(),
        test_samples.len()
    );
    println!();

    // Run tests
    let mut all_results = Vec::new();

    for (i, config) in test_configs.iter().enumerate() {
        print!(
            "[{:2}/{:2}] Testing {:<40}",
            i + 1,
            test_configs.len(),
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
                all_results.push(TestResult {
                    accuracy: acc,
                    training_time_ms: training_time.as_millis() as u64,
                    support_vectors,
                    test_name: config.name.clone(),
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
                all_results.push(TestResult {
                    accuracy: 0.0,
                    training_time_ms: 0,
                    support_vectors: 0,
                    test_name: config.name.clone(),
                });
            }
        }
    }

    // Generate report
    generate_report(&all_results, &dataset)?;

    println!("\n✅ Evaluation completed!");
    println!("📊 Report saved to: heart_dataset_evaluation_report.md");

    Ok(())
}

fn generate_test_configs() -> Vec<TestConfig> {
    let mut configs = Vec::new();

    // Linear kernel tests - key configurations
    for &c in &[0.1, 1.0, 10.0] {
        for (scaling_name, scaling) in &[
            ("None", None),
            (
                "MinMax",
                Some(ScalingMethod::MinMax {
                    min_val: -1.0,
                    max_val: 1.0,
                }),
            ),
            ("StandardScore", Some(ScalingMethod::StandardScore)),
        ] {
            configs.push(TestConfig {
                name: format!("Linear_C{}_Scale{}", c, scaling_name),
                c,
                scaling: scaling.clone(),
                gamma: None,
            });
        }
    }

    // RBF kernel tests - focused on promising configurations
    for &c in &[0.1, 1.0] {
        for &gamma in &[0.001, 0.01, 0.1] {
            for (scaling_name, scaling) in &[
                (
                    "MinMax",
                    Some(ScalingMethod::MinMax {
                        min_val: -1.0,
                        max_val: 1.0,
                    }),
                ),
                ("StandardScore", Some(ScalingMethod::StandardScore)),
            ] {
                configs.push(TestConfig {
                    name: format!("RBF_C{}_G{}_Scale{}", c, gamma, scaling_name),
                    c,
                    scaling: scaling.clone(),
                    gamma: Some(gamma),
                });
            }
        }
    }

    configs
}

fn generate_report(
    results: &[TestResult],
    dataset: &LibSVMDataset,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create("heart_dataset_evaluation_report.md")?;

    writeln!(file, "# Heart Dataset Performance Evaluation Report")?;
    writeln!(file)?;
    writeln!(file, "## Dataset Information")?;
    writeln!(file, "- **Dataset**: heart.libsvm")?;
    writeln!(file, "- **Samples**: {}", dataset.len())?;
    writeln!(file, "- **Dimensions**: {}", dataset.dim())?;

    let labels = dataset.get_labels();
    let positive = labels.iter().filter(|&&l| l > 0.0).count();
    let negative = labels.len() - positive;
    writeln!(
        file,
        "- **Classes**: {} positive, {} negative",
        positive, negative
    )?;
    writeln!(file, "- **Split**: 80% train, 20% test")?;
    writeln!(file)?;

    writeln!(file, "## Complete Results")?;
    writeln!(file)?;
    writeln!(
        file,
        "| Configuration | Accuracy (%) | Training Time (ms) | Support Vectors |"
    )?;
    writeln!(
        file,
        "|---------------|-------------:|------------------:|----------------:|"
    )?;

    let mut sorted_results = results.to_vec();
    sorted_results.sort_by(|a, b| b.accuracy.partial_cmp(&a.accuracy).unwrap());

    for result in &sorted_results {
        if result.accuracy > 0.0 {
            writeln!(
                file,
                "| {} | {:.1} | {} | {} |",
                result.test_name,
                result.accuracy * 100.0,
                result.training_time_ms,
                result.support_vectors
            )?;
        }
    }
    writeln!(file)?;

    // Analysis sections
    writeln!(file, "## Performance Analysis")?;
    writeln!(file)?;

    // Find best configurations
    let best_overall = &sorted_results[0];
    writeln!(file, "### Best Overall Configuration")?;
    writeln!(file, "- **Configuration**: {}", best_overall.test_name)?;
    writeln!(
        file,
        "- **Accuracy**: {:.1}%",
        best_overall.accuracy * 100.0
    )?;
    writeln!(
        file,
        "- **Training Time**: {} ms",
        best_overall.training_time_ms
    )?;
    writeln!(
        file,
        "- **Support Vectors**: {}",
        best_overall.support_vectors
    )?;
    writeln!(file)?;

    // Linear vs RBF comparison
    let linear_results: Vec<_> = sorted_results
        .iter()
        .filter(|r| r.test_name.starts_with("Linear_"))
        .collect();
    let rbf_results: Vec<_> = sorted_results
        .iter()
        .filter(|r| r.test_name.starts_with("RBF_"))
        .collect();

    writeln!(file, "### Linear vs RBF Kernel Comparison")?;
    if !linear_results.is_empty() && !rbf_results.is_empty() {
        let best_linear = &linear_results[0];
        let best_rbf = &rbf_results[0];

        writeln!(file, "**Best Linear Kernel:**")?;
        writeln!(file, "- Configuration: {}", best_linear.test_name)?;
        writeln!(file, "- Accuracy: {:.1}%", best_linear.accuracy * 100.0)?;
        writeln!(file, "- Training Time: {} ms", best_linear.training_time_ms)?;
        writeln!(file)?;

        writeln!(file, "**Best RBF Kernel:**")?;
        writeln!(file, "- Configuration: {}", best_rbf.test_name)?;
        writeln!(file, "- Accuracy: {:.1}%", best_rbf.accuracy * 100.0)?;
        writeln!(file, "- Training Time: {} ms", best_rbf.training_time_ms)?;
        writeln!(file)?;

        let accuracy_diff = (best_rbf.accuracy - best_linear.accuracy) * 100.0;
        writeln!(
            file,
            "**RBF Advantage**: {:.1} percentage points",
            accuracy_diff
        )?;

        if accuracy_diff > 2.0 {
            writeln!(
                file,
                "🎯 **Recommendation**: RBF kernel shows significant improvement"
            )?;
        } else if accuracy_diff > -2.0 {
            writeln!(
                file,
                "⚖️ **Recommendation**: Both kernels perform similarly, consider training time"
            )?;
        } else {
            writeln!(
                file,
                "🚀 **Recommendation**: Linear kernel is more efficient for this dataset"
            )?;
        }
        writeln!(file)?;
    }

    // C parameter analysis
    writeln!(file, "### C Parameter Impact")?;
    for &c in &[0.1, 1.0, 10.0, 100.0] {
        let c_results: Vec<_> = sorted_results
            .iter()
            .filter(|r| r.test_name.contains(&format!("C{}_", c)))
            .collect();
        if !c_results.is_empty() {
            let avg_acc =
                c_results.iter().map(|r| r.accuracy).sum::<f64>() / c_results.len() as f64;
            let best_acc = c_results.iter().map(|r| r.accuracy).fold(0.0, f64::max);
            writeln!(
                file,
                "- **C={}**: Average {:.1}%, Best {:.1}%",
                c,
                avg_acc * 100.0,
                best_acc * 100.0
            )?;
        }
    }
    writeln!(file)?;

    // Scaling method analysis
    writeln!(file, "### Scaling Method Impact")?;
    for scaling in &["None", "MinMax", "StandardScore", "UnitScale"] {
        let scaling_results: Vec<_> = sorted_results
            .iter()
            .filter(|r| r.test_name.contains(&format!("Scale{}", scaling)))
            .collect();
        if !scaling_results.is_empty() {
            let avg_acc = scaling_results.iter().map(|r| r.accuracy).sum::<f64>()
                / scaling_results.len() as f64;
            let best_acc = scaling_results
                .iter()
                .map(|r| r.accuracy)
                .fold(0.0, f64::max);
            writeln!(
                file,
                "- **{}**: Average {:.1}%, Best {:.1}%",
                scaling,
                avg_acc * 100.0,
                best_acc * 100.0
            )?;
        }
    }
    writeln!(file)?;

    // RBF gamma analysis
    if !rbf_results.is_empty() {
        writeln!(file, "### RBF Gamma Parameter Impact")?;
        for &gamma in &[0.001, 0.01, 0.1, 1.0, 10.0] {
            let gamma_results: Vec<_> = rbf_results
                .iter()
                .filter(|r| r.test_name.contains(&format!("G{}_", gamma)))
                .collect();
            if !gamma_results.is_empty() {
                let avg_acc = gamma_results.iter().map(|r| r.accuracy).sum::<f64>()
                    / gamma_results.len() as f64;
                let best_acc = gamma_results.iter().map(|r| r.accuracy).fold(0.0, f64::max);
                writeln!(
                    file,
                    "- **γ={}**: Average {:.1}%, Best {:.1}%",
                    gamma,
                    avg_acc * 100.0,
                    best_acc * 100.0
                )?;
            }
        }
        writeln!(file)?;
    }

    writeln!(file, "## Key Insights")?;
    writeln!(file)?;
    writeln!(
        file,
        "1. **Best Performance**: {:.1}% accuracy achieved",
        best_overall.accuracy * 100.0
    )?;

    if !linear_results.is_empty() && !rbf_results.is_empty() {
        let linear_avg =
            linear_results.iter().map(|r| r.accuracy).sum::<f64>() / linear_results.len() as f64;
        let rbf_avg =
            rbf_results.iter().map(|r| r.accuracy).sum::<f64>() / rbf_results.len() as f64;

        if rbf_avg > linear_avg + 0.02 {
            writeln!(
                file,
                "2. **Kernel Choice**: RBF kernels significantly outperform linear"
            )?;
        } else if linear_avg > rbf_avg + 0.02 {
            writeln!(
                file,
                "2. **Kernel Choice**: Linear kernels are sufficient for this dataset"
            )?;
        } else {
            writeln!(
                file,
                "2. **Kernel Choice**: Both Linear and RBF kernels perform similarly"
            )?;
        }
    }

    // Find fastest configuration with good accuracy
    let good_results: Vec<_> = sorted_results
        .iter()
        .filter(|r| r.accuracy >= best_overall.accuracy - 0.05)
        .collect();
    if let Some(fastest) = good_results.iter().min_by_key(|r| r.training_time_ms) {
        writeln!(
            file,
            "3. **Efficiency**: {} achieves {:.1}% in {} ms",
            fastest.test_name,
            fastest.accuracy * 100.0,
            fastest.training_time_ms
        )?;
    }

    writeln!(
        file,
        "4. **Parameter Sensitivity**: Performance varies significantly with C and γ values"
    )?;
    writeln!(file)?;

    writeln!(file, "## Recommendations")?;
    writeln!(file)?;
    writeln!(file, "**For Production Use:**")?;
    writeln!(file, "- **Primary Choice**: {}", best_overall.test_name)?;
    writeln!(
        file,
        "- **Expected Accuracy**: {:.1}%",
        best_overall.accuracy * 100.0
    )?;

    if let Some(fast_good) = good_results.iter().min_by_key(|r| r.training_time_ms) {
        if fast_good.test_name != best_overall.test_name {
            writeln!(
                file,
                "- **Fast Alternative**: {} ({:.1}%, {} ms)",
                fast_good.test_name,
                fast_good.accuracy * 100.0,
                fast_good.training_time_ms
            )?;
        }
    }

    writeln!(file)?;
    writeln!(file, "**For New Similar Datasets:**")?;
    writeln!(file, "1. Start with the best configuration found")?;
    writeln!(file, "2. Use cross-validation for final parameter tuning")?;
    writeln!(
        file,
        "3. Consider feature scaling, especially for RBF kernels"
    )?;
    writeln!(file, "4. Monitor support vector ratio for overfitting")?;
    writeln!(file)?;

    writeln!(file, "---")?;
    writeln!(file, "*Generated by RSVM Heart Dataset Evaluation*")?;

    Ok(())
}
