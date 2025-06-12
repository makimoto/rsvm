//! RSVM Command Line Interface
//!
//! A command-line interface for training, evaluating, and using SVM models
//! with LibSVM and CSV data formats.

use clap::{Args, Parser, Subcommand};
use env_logger::Env;
use log::{error, info, warn};
use rsvm::api::{quick, SVM};
use rsvm::core::Result;
use rsvm::persistence::SerializableModel;
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
    #[arg(short, long)]
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
}

#[derive(Args)]
struct PredictArgs {
    /// Trained model file
    #[arg(short, long)]
    model: PathBuf,

    /// Input data file
    #[arg(short, long)]
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
    #[arg(short, long)]
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
        error!("Error: {}", e);
        process::exit(1);
    }
}

fn train_command(args: TrainArgs) -> Result<()> {
    info!("Training SVM model...");
    info!("Data file: {:?}", args.data);
    info!("Parameters: C={}, epsilon={}, max_iter={}", args.c, args.epsilon, args.max_iterations);

    // Determine format
    let format = if args.format == "auto" {
        detect_format(&args.data)
    } else {
        args.format.clone()
    };

    info!("Loading dataset as {} format", format);

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
            "Unsupported format: {}. Use 'libsvm' or 'csv'",
            format
        ))),
    }
}

fn train_with_dataset<D: Dataset>(args: &TrainArgs, dataset: D) -> Result<()> {
    info!("Loaded {} samples with {} dimensions", dataset.len(), dataset.dim());

    // Validate dataset
    if dataset.len() < 2 {
        return Err(rsvm::core::SVMError::InvalidDataset(
            "Dataset must contain at least 2 samples".to_string(),
        ));
    }

    // Train model
    let model = SVM::new()
        .with_c(args.c)
        .with_epsilon(args.epsilon)
        .with_max_iterations(args.max_iterations)
        .with_cache_size(args.cache_size * 1024 * 1024) // Convert MB to bytes
        .train(&dataset)?;

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

    info!("Loading prediction data from: {:?}", args.data);
    
    let format = if args.format == "auto" {
        detect_format(&args.data)
    } else {
        args.format.clone()
    };

    let dataset_len = match format.as_str() {
        "libsvm" => {
            let dataset = LibSVMDataset::from_file(&args.data)?;
            dataset.len()
        }
        "csv" => {
            let dataset = CSVDataset::from_file(&args.data)?;
            dataset.len()
        }
        _ => return Err(rsvm::core::SVMError::InvalidParameter(format!(
            "Unsupported format: {}",
            format
        ))),
    };

    // For now, we can't reconstruct the full model, so we'll show what we would do
    warn!("Model reconstruction not yet fully implemented");
    warn!("Showing predictions that would be made with {} support vectors", 
          serializable_model.metadata.n_support_vectors);

    // Print prediction format that would be output
    println!("# Predictions for {} samples", dataset_len);
    println!("# Format: sample_index predicted_label{}", 
             if args.confidence { " confidence" } else { "" });

    // Show sample predictions (dummy implementation)
    for i in 0..dataset_len.min(10) {
        // For demo purposes, make a dummy prediction
        let dummy_prediction = if i % 2 == 0 { 1.0 } else { -1.0 };
        let dummy_confidence = 0.8;

        if args.confidence {
            println!("{} {:.0} {:.3}", i, dummy_prediction, dummy_confidence);
        } else {
            println!("{} {:.0}", i, dummy_prediction);
        }
    }

    if dataset_len > 10 {
        println!("... ({} more samples)", dataset_len - 10);
    }

    Ok(())
}

fn evaluate_command(args: EvaluateArgs) -> Result<()> {
    info!("Loading model from: {:?}", args.model);
    let serializable_model = SerializableModel::load_from_file(&args.model)?;

    info!("Loading test data from: {:?}", args.data);
    
    let format = if args.format == "auto" {
        detect_format(&args.data)
    } else {
        args.format.clone()
    };

    let (dataset_len, dataset_dim, labels) = match format.as_str() {
        "libsvm" => {
            let dataset = LibSVMDataset::from_file(&args.data)?;
            (dataset.len(), dataset.dim(), dataset.get_labels())
        }
        "csv" => {
            let dataset = CSVDataset::from_file(&args.data)?;
            (dataset.len(), dataset.dim(), dataset.get_labels())
        }
        _ => return Err(rsvm::core::SVMError::InvalidParameter(format!(
            "Unsupported format: {}",
            format
        ))),
    };

    warn!("Model reconstruction not yet fully implemented");
    info!("Test dataset: {} samples, {} dimensions", dataset_len, dataset_dim);

    // Show what evaluation would look like
    println!("=== Model Evaluation ===");
    serializable_model.print_summary();
    println!("\nTest Dataset:");
    println!("  Samples: {}", dataset_len);
    println!("  Dimensions: {}", dataset_dim);

    // For demo, compute some basic dataset statistics
    let pos_count = labels.iter().filter(|&&l| l > 0.0).count();
    let neg_count = labels.len() - pos_count;

    println!("  Positive samples: {}", pos_count);
    println!("  Negative samples: {}", neg_count);
    println!("  Balance ratio: {:.2}", pos_count as f64 / neg_count as f64);

    if args.detailed {
        println!("\nDetailed metrics would be computed here once model reconstruction is implemented");
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
        println!("  First SV indices: {:?}", &first_sv.indices[..first_sv.indices.len().min(5)]);
        
        if first_sv.indices.len() > 5 {
            println!("    ... ({} more)", first_sv.indices.len() - 5);
        }
    }

    println!("\nAlpha*Y values:");
    let alpha_y = &serializable_model.alpha_y;
    let n_show = alpha_y.len().min(10);
    for (i, &alpha_y_val) in alpha_y.iter().enumerate().take(n_show) {
        println!("  α{}*y{}: {:.6}", i, i, alpha_y_val);
    }
    if alpha_y.len() > n_show {
        println!("  ... ({} more)", alpha_y.len() - n_show);
    }

    Ok(())
}

fn quick_command(args: QuickArgs) -> Result<()> {
    match args.operation {
        QuickOperation::Eval { train, test, c } => {
            info!("Quick evaluation: train on {:?}, test on {:?}", train, test);
            
            let accuracy = quick::evaluate_split(&train, &test)?;
            
            println!("=== Quick Evaluation Results ===");
            println!("Training file: {:?}", train);
            println!("Test file: {:?}", test);
            println!("C parameter: {}", c);
            println!("Test accuracy: {:.2}%", accuracy * 100.0);
            
            Ok(())
        }
        QuickOperation::Cv { data, ratio, c } => {
            info!("Cross-validation on {:?} with ratio {}", data, ratio);
            
            let format = detect_format(&data);
            let accuracy = match format.as_str() {
                "libsvm" => {
                    let dataset = LibSVMDataset::from_file(&data)?;
                    quick::simple_validation(&dataset, ratio, c)?
                }
                "csv" => {
                    let dataset = CSVDataset::from_file(&data)?;
                    quick::simple_validation(&dataset, ratio, c)?
                }
                _ => return Err(rsvm::core::SVMError::InvalidParameter(format!(
                    "Unsupported format: {}",
                    format
                ))),
            };
            
            println!("=== Cross-Validation Results ===");
            println!("Data file: {:?}", data);
            println!("Train/test ratio: {:.1}/{:.1}", ratio, 1.0 - ratio);
            println!("C parameter: {}", c);
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