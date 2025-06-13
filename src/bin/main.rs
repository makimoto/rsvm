//! RSVM Command Line Interface
//!
//! A command-line interface for training, evaluating, and using SVM models
//! with LibSVM and CSV data formats.

use clap::{Args, Parser, Subcommand, ValueEnum};
use env_logger::Env;
use log::{error, info, warn};
use rsvm::api::{quick, SVM};
use rsvm::core::{Result, WorkingSetStrategy};
use rsvm::persistence::SerializableModel;
use rsvm::utils::scaling::ScalingMethod;
use rsvm::{CSVDataset, Dataset, LibSVMDataset};
use std::path::{Path, PathBuf};
use std::process;

#[derive(Parser)]
#[command(name = "rsvm")]
#[command(about = "A Rust implementation of Support Vector Machine")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(author = "RSVM Contributors")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose output
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Enable debug output
    #[arg(short, long, global = true)]
    debug: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a new SVM model
    Train(TrainArgs),
    /// Make predictions using a trained model
    Predict(PredictArgs),
    /// Evaluate a model on test data
    Evaluate(EvaluateArgs),
    /// Display model information
    Info(InfoArgs),
    /// Quick operations without model saving
    Quick(QuickArgs),
}

#[derive(Args)]
struct TrainArgs {
    /// Training data file (LibSVM or CSV format)
    #[arg(long)]
    data: PathBuf,

    /// Output model file
    #[arg(short, long)]
    output: PathBuf,

    /// Data format: auto, libsvm, or csv
    #[arg(short, long, default_value = "auto")]
    format: String,

    /// Regularization parameter C
    #[arg(short = 'C', long, default_value = "1.0")]
    c: f64,

    /// Convergence tolerance
    #[arg(short, long, default_value = "0.001")]
    epsilon: f64,

    /// Maximum iterations
    #[arg(short, long, default_value = "1000")]
    max_iterations: usize,

    /// Kernel cache size in MB
    #[arg(long, default_value = "100")]
    cache_size: usize,

    /// Working set selection strategy
    #[arg(long, default_value = "smo-heuristic")]
    working_set_strategy: CliWorkingSetStrategy,

    /// Feature scaling method
    #[arg(long)]
    feature_scaling: Option<CliScalingMethod>,
}

#[derive(ValueEnum, Clone, Debug)]
enum CliWorkingSetStrategy {
    /// SMO heuristic: max |E_i - E_j| (fast, default)
    #[value(name = "smo-heuristic")]
    SMOHeuristic,
    /// Steepest descent: max KKT violation (SVMlight style, more rigorous)
    #[value(name = "steepest-descent")]
    SteepestDescent,
    /// Random selection (for debugging/comparison)
    #[value(name = "random")]
    Random,
}

#[derive(ValueEnum, Clone, Debug)]
enum CliScalingMethod {
    /// Min-Max scaling to [-1, 1] range
    #[value(name = "minmax")]
    MinMax,
    /// Standard score (Z-score) normalization
    #[value(name = "standard")]
    StandardScore,
    /// Unit scaling by maximum absolute value
    #[value(name = "unit")]
    UnitScale,
}

impl From<CliScalingMethod> for ScalingMethod {
    fn from(cli_method: CliScalingMethod) -> Self {
        match cli_method {
            CliScalingMethod::MinMax => ScalingMethod::MinMax {
                min_val: -1.0,
                max_val: 1.0,
            },
            CliScalingMethod::StandardScore => ScalingMethod::StandardScore,
            CliScalingMethod::UnitScale => ScalingMethod::UnitScale,
        }
    }
}

impl From<CliWorkingSetStrategy> for WorkingSetStrategy {
    fn from(cli_strategy: CliWorkingSetStrategy) -> Self {
        match cli_strategy {
            CliWorkingSetStrategy::SMOHeuristic => WorkingSetStrategy::SMOHeuristic,
            CliWorkingSetStrategy::SteepestDescent => WorkingSetStrategy::SteepestDescent,
            CliWorkingSetStrategy::Random => WorkingSetStrategy::Random,
        }
    }
}

#[derive(Args)]
struct PredictArgs {
    /// Trained model file
    #[arg(short, long)]
    model: PathBuf,

    /// Input data file
    #[arg(long)]
    data: PathBuf,

    /// Output predictions file (optional, prints to stdout if not specified)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Data format: auto, libsvm, or csv
    #[arg(short, long, default_value = "auto")]
    format: String,

    /// Show confidence scores
    #[arg(long)]
    confidence: bool,
}

#[derive(Args)]
struct EvaluateArgs {
    /// Trained model file
    #[arg(short, long)]
    model: PathBuf,

    /// Test data file
    #[arg(long)]
    data: PathBuf,

    /// Data format: auto, libsvm, or csv
    #[arg(short, long, default_value = "auto")]
    format: String,

    /// Show detailed metrics
    #[arg(long)]
    detailed: bool,
}

#[derive(Args)]
struct InfoArgs {
    /// Model file
    model: PathBuf,
}

#[derive(Args)]
struct QuickArgs {
    #[command(subcommand)]
    operation: QuickOperation,
}

#[derive(Subcommand)]
enum QuickOperation {
    /// Quick train and evaluate with train/test split
    Eval {
        /// Training data file
        train: PathBuf,
        /// Test data file  
        test: PathBuf,
        /// Regularization parameter C
        #[arg(short = 'C', long, default_value = "1.0")]
        c: f64,
        /// Feature scaling method
        #[arg(long)]
        feature_scaling: Option<CliScalingMethod>,
    },
    /// Cross-validation on a single dataset
    Cv {
        /// Data file
        data: PathBuf,
        /// Training ratio (0.0-1.0)
        #[arg(short, long, default_value = "0.8")]
        ratio: f64,
        /// Regularization parameter C
        #[arg(short = 'C', long, default_value = "1.0")]
        c: f64,
        /// Working set selection strategy
        #[arg(long, default_value = "smo-heuristic")]
        working_set_strategy: CliWorkingSetStrategy,
        /// Feature scaling method
        #[arg(long)]
        feature_scaling: Option<CliScalingMethod>,
    },
}

fn main() {
    let cli = Cli::parse();

    // Initialize logging
    let log_level = if cli.debug {
        "debug"
    } else if cli.verbose {
        "info"
    } else {
        "warn"
    };

    env_logger::Builder::from_env(Env::default().default_filter_or(log_level)).init();

    let result = match cli.command {
        Commands::Train(args) => train_command(args),
        Commands::Predict(args) => predict_command(args),
        Commands::Evaluate(args) => evaluate_command(args),
        Commands::Info(args) => info_command(args),
        Commands::Quick(args) => quick_command(args),
    };

    if let Err(e) = result {
        error!("Error: {e}");
        process::exit(1);
    }
}

fn train_command(args: TrainArgs) -> Result<()> {
    info!("Training SVM model...");
    info!("Data file: {:?}", args.data);
    info!(
        "Parameters: C={}, epsilon={}, max_iter={}",
        args.c, args.epsilon, args.max_iterations
    );

    // Determine format
    let format = if args.format == "auto" {
        detect_format(&args.data)
    } else {
        args.format.clone()
    };

    info!("Loading dataset as {format} format");

    // Process different formats separately to avoid trait object issues
    match format.as_str() {
        "libsvm" => {
            let dataset = LibSVMDataset::from_file(&args.data)?;
            train_with_dataset(&args, dataset)
        }
        "csv" => {
            let dataset = CSVDataset::from_file(&args.data)?;
            train_with_dataset(&args, dataset)
        }
        _ => Err(rsvm::core::SVMError::InvalidParameter(format!(
            "Unsupported format: {format}. Use 'libsvm' or 'csv'"
        ))),
    }
}

fn train_with_dataset<D: Dataset>(args: &TrainArgs, dataset: D) -> Result<()> {
    info!(
        "Loaded {} samples with {} dimensions",
        dataset.len(),
        dataset.dim()
    );

    // Validate dataset
    if dataset.len() < 2 {
        return Err(rsvm::core::SVMError::InvalidDataset(
            "Dataset must contain at least 2 samples".to_string(),
        ));
    }

    // Train model
    let mut svm_builder = SVM::new()
        .with_c(args.c)
        .with_epsilon(args.epsilon)
        .with_max_iterations(args.max_iterations)
        .with_cache_size(args.cache_size * 1024 * 1024) // Convert MB to bytes
        .with_working_set_strategy(args.working_set_strategy.clone().into());

    // Add feature scaling if specified
    if let Some(scaling_method) = &args.feature_scaling {
        info!("Using feature scaling: {scaling_method:?}");
        svm_builder = svm_builder.with_feature_scaling(scaling_method.clone().into());
    }

    let model = svm_builder.train(&dataset)?;

    info!("Training completed successfully");

    let info = model.info();
    info!("Support vectors: {}", info.n_support_vectors);
    info!("Bias: {:.6}", info.bias);

    // Save model
    let serializable = SerializableModel::from_trained_model(&model);
    serializable.save_to_file(&args.output)?;
    info!("Model saved to: {:?}", args.output);

    // Quick evaluation on training data
    let accuracy = model.evaluate(&dataset);
    info!("Training accuracy: {:.2}%", accuracy * 100.0);

    Ok(())
}

fn predict_command(args: PredictArgs) -> Result<()> {
    info!("Loading model from: {:?}", args.model);
    let serializable_model = SerializableModel::load_from_file(&args.model)?;
    let model = serializable_model.to_trained_model()?;

    info!("Loading prediction data from: {:?}", args.data);

    let format = if args.format == "auto" {
        detect_format(&args.data)
    } else {
        args.format.clone()
    };

    info!(
        "Making predictions using model with {} support vectors",
        serializable_model.metadata.n_support_vectors
    );

    let predictions = match format.as_str() {
        "libsvm" => {
            let dataset = LibSVMDataset::from_file(&args.data)?;
            model.predict_dataset(&dataset)
        }
        "csv" => {
            let dataset = CSVDataset::from_file(&args.data)?;
            model.predict_dataset(&dataset)
        }
        _ => {
            return Err(rsvm::core::SVMError::InvalidParameter(format!(
                "Unsupported format: {format}"
            )))
        }
    };

    // Output results
    if let Some(output_path) = args.output {
        // Write to file
        use std::fs::File;
        use std::io::{BufWriter, Write};

        let file = File::create(&output_path).map_err(rsvm::core::SVMError::IoError)?;
        let mut writer = BufWriter::new(file);

        writeln!(writer, "# Predictions for {} samples", predictions.len())
            .map_err(rsvm::core::SVMError::IoError)?;
        writeln!(
            writer,
            "# Format: sample_index predicted_label{}",
            if args.confidence { " confidence" } else { "" }
        )
        .map_err(rsvm::core::SVMError::IoError)?;

        for (i, pred) in predictions.iter().enumerate() {
            if args.confidence {
                writeln!(writer, "{} {:.0} {:.6}", i, pred.label, pred.confidence())
                    .map_err(rsvm::core::SVMError::IoError)?;
            } else {
                writeln!(writer, "{} {:.0}", i, pred.label)
                    .map_err(rsvm::core::SVMError::IoError)?;
            }
        }

        info!("Predictions saved to: {output_path:?}");
    } else {
        // Print to stdout
        println!("# Predictions for {} samples", predictions.len());
        println!(
            "# Format: sample_index predicted_label{}",
            if args.confidence { " confidence" } else { "" }
        );

        for (i, pred) in predictions.iter().enumerate() {
            if args.confidence {
                println!("{} {:.0} {:.6}", i, pred.label, pred.confidence());
            } else {
                println!("{} {:.0}", i, pred.label);
            }
        }
    }

    Ok(())
}

fn evaluate_command(args: EvaluateArgs) -> Result<()> {
    info!("Loading model from: {:?}", args.model);
    let serializable_model = SerializableModel::load_from_file(&args.model)?;
    let model = serializable_model.to_trained_model()?;

    info!("Loading test data from: {:?}", args.data);

    let format = if args.format == "auto" {
        detect_format(&args.data)
    } else {
        args.format.clone()
    };

    info!(
        "Evaluating model with {} support vectors",
        serializable_model.metadata.n_support_vectors
    );

    let (accuracy, detailed_metrics) = match format.as_str() {
        "libsvm" => {
            let dataset = LibSVMDataset::from_file(&args.data)?;
            let accuracy = model.evaluate(&dataset);
            let detailed = if args.detailed {
                Some(model.evaluate_detailed(&dataset))
            } else {
                None
            };
            (accuracy, detailed)
        }
        "csv" => {
            let dataset = CSVDataset::from_file(&args.data)?;
            let accuracy = model.evaluate(&dataset);
            let detailed = if args.detailed {
                Some(model.evaluate_detailed(&dataset))
            } else {
                None
            };
            (accuracy, detailed)
        }
        _ => {
            return Err(rsvm::core::SVMError::InvalidParameter(format!(
                "Unsupported format: {format}"
            )))
        }
    };

    // Show evaluation results
    println!("=== Model Evaluation ===");
    serializable_model.print_summary();

    println!("\nTest Results:");
    println!("  Accuracy: {:.2}%", accuracy * 100.0);

    if let Some(metrics) = detailed_metrics {
        println!("\nDetailed Metrics:");
        println!("  True Positives:  {}", metrics.true_positives);
        println!("  True Negatives:  {}", metrics.true_negatives);
        println!("  False Positives: {}", metrics.false_positives);
        println!("  False Negatives: {}", metrics.false_negatives);
        println!("  Precision:       {:.4}", metrics.precision());
        println!("  Recall:          {:.4}", metrics.recall());
        println!("  F1 Score:        {:.4}", metrics.f1_score());
        println!("  Specificity:     {:.4}", metrics.specificity());
    }

    Ok(())
}

fn info_command(args: InfoArgs) -> Result<()> {
    info!("Loading model from: {:?}", args.model);
    let serializable_model = SerializableModel::load_from_file(&args.model)?;

    serializable_model.print_summary();

    println!("\nSupport Vector Details:");
    println!("  Total: {}", serializable_model.support_vectors.len());

    if !serializable_model.support_vectors.is_empty() {
        let first_sv = &serializable_model.support_vectors[0];
        println!("  First SV dimensions: {}", first_sv.indices.len());
        println!(
            "  First SV indices: {:?}",
            &first_sv.indices[..first_sv.indices.len().min(5)]
        );

        if first_sv.indices.len() > 5 {
            println!("    ... ({} more)", first_sv.indices.len() - 5);
        }
    }

    println!("\nAlpha*Y values:");
    let alpha_y = &serializable_model.alpha_y;
    let n_show = alpha_y.len().min(10);
    for (i, &alpha_y_val) in alpha_y.iter().enumerate().take(n_show) {
        println!("  α{i}*y{i}: {alpha_y_val:.6}");
    }
    if alpha_y.len() > n_show {
        println!("  ... ({} more)", alpha_y.len() - n_show);
    }

    Ok(())
}

fn quick_command(args: QuickArgs) -> Result<()> {
    match args.operation {
        QuickOperation::Eval {
            train,
            test,
            c,
            feature_scaling,
        } => {
            info!("Quick evaluation: train on {train:?}, test on {test:?}");

            let scaling_method = feature_scaling.clone().map(|s| s.into());
            let accuracy = quick::evaluate_split_with_params(&train, &test, c, scaling_method)?;

            println!("=== Quick Evaluation Results ===");
            println!("Training file: {train:?}");
            println!("Test file: {test:?}");
            println!("C parameter: {c}");
            if let Some(ref scaling) = feature_scaling {
                println!("Feature scaling: {scaling:?}");
            }
            println!("Test accuracy: {:.2}%", accuracy * 100.0);

            Ok(())
        }
        QuickOperation::Cv {
            data,
            ratio,
            c,
            working_set_strategy,
            feature_scaling,
        } => {
            info!("Cross-validation on {data:?} with ratio {ratio}");

            let format = detect_format(&data);
            let strategy = working_set_strategy.clone().into();
            let scaling_method = feature_scaling.clone().map(|s| s.into());

            let accuracy = match format.as_str() {
                "libsvm" => {
                    let dataset = LibSVMDataset::from_file(&data)?;
                    quick::simple_validation_with_strategy_and_scaling(
                        &dataset,
                        ratio,
                        c,
                        strategy,
                        scaling_method,
                    )?
                }
                "csv" => {
                    let dataset = CSVDataset::from_file(&data)?;
                    quick::simple_validation_with_strategy_and_scaling(
                        &dataset,
                        ratio,
                        c,
                        strategy,
                        scaling_method,
                    )?
                }
                _ => {
                    return Err(rsvm::core::SVMError::InvalidParameter(format!(
                        "Unsupported format: {format}"
                    )))
                }
            };

            println!("=== Cross-Validation Results ===");
            println!("Data file: {data:?}");
            println!("Train/test ratio: {ratio:.1}/{:.1}", 1.0 - ratio);
            println!("C parameter: {c}");
            println!("Working set strategy: {strategy:?}");
            if let Some(ref scaling) = feature_scaling {
                println!("Feature scaling: {scaling:?}");
            }
            println!("CV accuracy: {:.2}%", accuracy * 100.0);

            Ok(())
        }
    }
}

fn detect_format(path: &Path) -> String {
    if let Some(ext) = path.extension() {
        match ext.to_str() {
            Some("csv") => "csv".to_string(),
            Some("libsvm") | Some("svm") => "libsvm".to_string(),
            _ => {
                // Try to detect by content or use libsvm as default
                warn!("Unknown file extension, assuming LibSVM format");
                "libsvm".to_string()
            }
        }
    } else {
        warn!("No file extension, assuming LibSVM format");
        "libsvm".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_detection() {
        assert_eq!(detect_format(&PathBuf::from("test.csv")), "csv");
        assert_eq!(detect_format(&PathBuf::from("test.libsvm")), "libsvm");
        assert_eq!(detect_format(&PathBuf::from("test.svm")), "libsvm");
        assert_eq!(detect_format(&PathBuf::from("test")), "libsvm");
    }
}
