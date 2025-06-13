# RSVM Tutorial: From Installation to Model Training

This tutorial provides a step-by-step guide to using the RSVM (Rust Support Vector Machine) library, from installation through training and evaluating models on real datasets.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [CLI Usage](#cli-usage)
4. [Working with Data Formats](#working-with-data-formats)
5. [Training Your First Model](#training-your-first-model)
6. [Model Evaluation](#model-evaluation)
7. [Advanced Configuration](#advanced-configuration)
8. [Real-World Example](#real-world-example)
9. [Performance Tips](#performance-tips)
10. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Rust 1.70 or later
- Cargo package manager

### Option 1: Add as Dependency

Add RSVM to your `Cargo.toml`:

```toml
[dependencies]
rsvm = "0.1.0"
```

### Option 2: Build from Source

```bash
git clone https://github.com/your-org/rsvm.git
cd rsvm
cargo build --release
```

To build the CLI binary:

```bash
cargo build --bin rsvm --release
```

### Verify Installation

```bash
cargo test
```

All tests should pass, confirming the installation is successful.

Verify CLI functionality:

```bash
./target/release/rsvm --help
```

## Quick Start

### Simple Example

Create a minimal SVM training example:

```rust
use rsvm::api::SVM;
use rsvm::{Sample, SparseVector};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create training data with different scales
    let samples = vec![
        Sample::new(SparseVector::new(vec![0, 1], vec![200.0, 10.0]), 1.0),
        Sample::new(SparseVector::new(vec![0, 1], vec![180.0, 11.0]), 1.0),
        Sample::new(SparseVector::new(vec![0, 1], vec![-200.0, -10.0]), -1.0),
        Sample::new(SparseVector::new(vec![0, 1], vec![-180.0, -11.0]), -1.0),
    ];

    // Train the model with feature scaling
    let model = SVM::new()
        .with_c(1.0)
        .with_feature_scaling(rsvm::utils::scaling::ScalingMethod::MinMax { 
            min_val: -1.0, max_val: 1.0 
        })
        .train_samples(&samples)?;

    // Make a prediction (no need to scale - done automatically)
    let test_sample = Sample::new(SparseVector::new(vec![0, 1], vec![150.0, 8.0]), 1.0);
    let prediction = model.predict(&test_sample);
    
    println!("Predicted label: {}", prediction.label);
    println!("Confidence: {:.3}", prediction.confidence());

    Ok(())
}
```

## CLI Usage

RSVM includes a comprehensive command-line interface for easy model training, prediction, and evaluation without writing code.

### Basic CLI Commands

```bash
# Show help
rsvm --help

# Show help for specific commands
rsvm train --help
rsvm predict --help
rsvm evaluate --help
rsvm info --help
```

### Model Training

```bash
# Basic training (LibSVM format)
rsvm train --data training_data.libsvm --output my_model.json

# Training with CSV data
rsvm train --data training_data.csv --output my_model.json --format csv

# Training with custom parameters
rsvm train --data training_data.libsvm --output my_model.json \
    -C 10.0 --epsilon 0.0001 --max-iterations 2000

# Training with feature scaling
rsvm train --data training_data.libsvm --output my_model.json \
    --feature-scaling minmax

# Verbose training output
rsvm train --data training_data.libsvm --output my_model.json --verbose
```

### Model Information

```bash
# Display detailed model information
rsvm info my_model.json
```

Example output:
```
=== SVM Model Summary ===
Kernel Type: linear
Support Vectors: 23
Bias: 0.123456
Library Version: 0.1.0
Created: 2025-06-12T10:30:00+00:00
Training Parameters:
  C: 1
  Epsilon: 0.001
  Max Iterations: 1000
```

### Quick Operations

```bash
# Train/test split evaluation
rsvm quick eval training_data.libsvm test_data.libsvm

# Train/test split with feature scaling
rsvm quick eval training_data.libsvm test_data.libsvm --feature-scaling minmax

# Cross-validation
rsvm quick cv data.libsvm --ratio 0.8

# Cross-validation with feature scaling
rsvm quick cv data.libsvm --ratio 0.8 --feature-scaling standard

# Quick evaluation with custom parameters and scaling
rsvm quick cv data.libsvm --ratio 0.8 -C 5.0 --feature-scaling minmax
```

### Practical CLI Workflow

Complete workflow using LibSVM data:

```bash
# 1. Create sample data
cat > sample_data.libsvm << EOF
+1 1:2.0 2:1.0
+1 1:1.8 2:1.1
+1 1:2.2 2:0.9
-1 1:-2.0 2:-1.0
-1 1:-1.8 2:-1.1
-1 1:-2.2 2:-0.9
EOF

# 2. Create test data
cat > test_data.libsvm << EOF
+1 1:1.9 2:0.95
-1 1:-1.9 2:-0.95
EOF

# 3. Train model
rsvm train --data sample_data.libsvm --output trained_model.json --verbose

# 4. Check model info
rsvm info trained_model.json

# 5. Make predictions
rsvm predict --model trained_model.json --data test_data.libsvm --confidence

# 6. Evaluate model
rsvm evaluate --model trained_model.json --data test_data.libsvm --detailed

# 7. Cross-validate on original data
rsvm quick cv sample_data.libsvm --ratio 0.8

# 8. Cross-validate with feature scaling for better accuracy
rsvm quick cv sample_data.libsvm --ratio 0.8 --feature-scaling standard
```

### Parameter Tuning

```bash
# Batch evaluation with different C values
for c in 0.1 1.0 10.0 100.0; do
  echo "Testing C=$c"
  rsvm quick cv data.libsvm -C $c --ratio 0.8
done

# Parameter tuning with feature scaling
for c in 0.1 1.0 10.0 100.0; do
  echo "Testing C=$c with StandardScore scaling"
  rsvm quick cv data.libsvm -C $c --ratio 0.8 --feature-scaling standard
done
```

### Model Prediction and Evaluation

```bash
# Make predictions using a trained model
rsvm predict --model my_model.json --data test_data.libsvm

# Make predictions with confidence scores
rsvm predict --model my_model.json --data test_data.libsvm --confidence

# Save predictions to file
rsvm predict --model my_model.json --data test_data.libsvm --output predictions.txt

# Evaluate model performance
rsvm evaluate --model my_model.json --data test_data.libsvm

# Detailed evaluation with precision, recall, F1-score
rsvm evaluate --model my_model.json --data test_data.libsvm --detailed
```

Example detailed evaluation output:
```
=== Model Evaluation ===
=== SVM Model Summary ===
Kernel Type: linear
Support Vectors: 2
Bias: -0.000000

Test Results:
  Accuracy: 100.00%

Detailed Metrics:
  True Positives:  1
  True Negatives:  1
  False Positives: 0
  False Negatives: 0
  Precision:       1.0000
  Recall:          1.0000
  F1 Score:        1.0000
  Specificity:     1.0000
```

For comprehensive CLI examples, see [CLI_EXAMPLES.md](CLI_EXAMPLES.md).

## Working with Data Formats

RSVM supports two popular data formats: LibSVM and CSV.

### LibSVM Format

LibSVM format is widely used in machine learning:

```
+1 1:2.0 2:1.0 3:0.5
-1 1:-2.0 2:-1.0 3:-0.5
+1 2:1.5 4:2.0
```

Format: `<label> <index>:<value> <index>:<value> ...`

- Labels: +1 (positive class) or -1 (negative class)
- Indices: 1-based feature indices
- Values: feature values (only non-zero values need to be specified)

#### Creating LibSVM Data

```rust
// data.libsvm
use std::fs::File;
use std::io::Write;

let mut file = File::create("data.libsvm")?;
writeln!(file, "+1 1:2.0 2:1.0")?;
writeln!(file, "-1 1:-2.0 2:-1.0")?;
writeln!(file, "+1 1:1.8 2:1.1")?;
writeln!(file, "-1 1:-1.8 2:-1.1")?;
```

#### Loading LibSVM Data

In code:
```rust
use rsvm::api::SVM;

let model = SVM::new()
    .with_c(1.0)
    .train_from_file("data.libsvm")?;
```

With CLI:
```bash
rsvm train --data data.libsvm --output model.json
```

### CSV Format

CSV format is convenient for data from spreadsheets or databases:

```csv
feature1,feature2,label
2.0,1.0,1
-2.0,-1.0,-1
1.8,1.1,1
-1.8,-1.1,-1
```

- Last column: label (automatically converted to +1/-1)
- Other columns: features
- Headers: automatically detected

#### Loading CSV Data

In code:
```rust
use rsvm::api::SVM;

let model = SVM::new()
    .with_c(1.0)
    .train_from_csv("data.csv")?;
```

With CLI:
```bash
rsvm train --data data.csv --output model.json --format csv
```

## Training Your First Model

There are two ways to start training models with RSVM: using the CLI tool or programmatically with the API.

### CLI Quick Training

Start with the CLI for quick experimentation:

```bash
# Create sample data
cat > first_training.libsvm << EOF
+1 1:2.0 2:2.0
+1 1:2.1 2:1.9
+1 1:1.9 2:2.1
-1 1:-2.0 2:-2.0
-1 1:-2.1 2:-1.9
-1 1:-1.9 2:-2.1
EOF

# Train model
rsvm train --data first_training.libsvm --output first_model.json --verbose

# Check model info
rsvm info first_model.json

# Cross-validate
rsvm quick cv first_training.libsvm --ratio 0.8
```

### Programmatic Training

#### Step 1: Prepare Your Data

Let's create a simple linearly separable dataset:

```rust
use std::fs::File;
use std::io::Write;

fn create_sample_data() -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create("tutorial_data.libsvm")?;
    
    // Positive class: points around (2, 2)
    writeln!(file, "+1 1:2.0 2:2.0")?;
    writeln!(file, "+1 1:2.1 2:1.9")?;
    writeln!(file, "+1 1:1.9 2:2.1")?;
    writeln!(file, "+1 1:2.2 2:1.8")?;
    
    // Negative class: points around (-2, -2)
    writeln!(file, "-1 1:-2.0 2:-2.0")?;
    writeln!(file, "-1 1:-2.1 2:-1.9")?;
    writeln!(file, "-1 1:-1.9 2:-2.1")?;
    writeln!(file, "-1 1:-2.2 2:-1.8")?;
    
    Ok(())
}
```

#### Step 2: Train the Model

```rust
use rsvm::api::SVM;

fn train_model() -> Result<(), Box<dyn std::error::Error>> {
    // Create sample data
    create_sample_data()?;
    
    // Train with custom parameters
    let model = SVM::new()
        .with_c(1.0)                    // Regularization parameter
        .with_epsilon(0.001)            // Convergence tolerance
        .with_max_iterations(1000)      // Maximum iterations
        .train_from_file("tutorial_data.libsvm")?;
    
    println!("Model trained successfully!");
    println!("Support vectors: {}", model.info().n_support_vectors);
    println!("Bias: {:.3}", model.info().bias);
    
    Ok(())
}
```

#### Step 3: Test the Model

```rust
use rsvm::{Sample, SparseVector};

fn test_model(model: &rsvm::api::TrainedModel<rsvm::LinearKernel>) {
    let test_cases = vec![
        (vec![1.5, 1.5], "Should be positive"),
        (vec![-1.5, -1.5], "Should be negative"),
        (vec![0.0, 0.0], "Boundary case"),
    ];
    
    for (coords, description) in test_cases {
        let test_sample = Sample::new(
            SparseVector::new(vec![0, 1], coords),
            0.0  // Unknown label
        );
        
        let prediction = model.predict(&test_sample);
        println!("{}: {} (confidence: {:.3})", 
                description, 
                prediction.label, 
                prediction.confidence());
    }
}
```

## Model Evaluation

### CLI Evaluation

Quick evaluation using CLI:
```bash
# Cross-validation
rsvm quick cv tutorial_data.libsvm --ratio 0.8

# Separate train/test evaluation
rsvm quick eval train_data.libsvm test_data.libsvm

# Parameter tuning
rsvm quick cv tutorial_data.libsvm --ratio 0.8 -C 5.0
```

### Programmatic Evaluation

#### Basic Accuracy

```rust
// Evaluate on training data (for verification)
let accuracy = model.evaluate_from_file("tutorial_data.libsvm")?;
println!("Training accuracy: {:.1}%", accuracy * 100.0);
```

#### Detailed Metrics

```rust
use rsvm::LibSVMDataset;

let dataset = LibSVMDataset::from_file("tutorial_data.libsvm")?;
let metrics = model.evaluate_detailed(&dataset);

println!("Accuracy: {:.3}", metrics.accuracy());
println!("Precision: {:.3}", metrics.precision());
println!("Recall: {:.3}", metrics.recall());
println!("F1 Score: {:.3}", metrics.f1_score());
println!("Specificity: {:.3}", metrics.specificity());
```

#### Cross-Validation

```rust
use rsvm::api::quick;
use rsvm::utils::scaling::ScalingMethod;

// Simple train/test split validation
let accuracy = quick::simple_validation(&dataset, 0.8, 1.0)?;
println!("Cross-validation accuracy: {:.1}%", accuracy * 100.0);

// Cross-validation with feature scaling
let accuracy_scaled = quick::simple_validation_with_strategy_and_scaling(
    &dataset, 
    0.8, 
    1.0,
    rsvm::core::WorkingSetStrategy::SMOHeuristic,
    Some(ScalingMethod::StandardScore)
)?;
println!("CV accuracy with scaling: {:.1}%", accuracy_scaled * 100.0);
```

## Advanced Configuration

### Parameter Tuning

```rust
// Try different C values
let c_values = vec![0.1, 1.0, 10.0, 100.0];

for &c in &c_values {
    let model = SVM::new()
        .with_c(c)
        .train_from_file("tutorial_data.libsvm")?;
    
    let accuracy = model.evaluate_from_file("tutorial_data.libsvm")?;
    println!("C = {}: Accuracy = {:.1}%", c, accuracy * 100.0);
}
```

### Memory Optimization

```rust
// For large datasets, increase cache size
let model = SVM::new()
    .with_c(1.0)
    .with_cache_size(100 * 1024 * 1024)  // 100MB cache
    .train_from_file("large_dataset.libsvm")?;
```

### Feature Scaling Configuration

```rust
use rsvm::utils::scaling::ScalingMethod;

// Combined configuration with scaling
let model = SVM::new()
    .with_c(10.0)
    .with_feature_scaling(ScalingMethod::StandardScore)
    .with_max_iterations(2000)
    .train_from_file("raw_data.libsvm")?;

// Custom range scaling
let model = SVM::new()
    .with_feature_scaling(ScalingMethod::MinMax { min_val: 0.0, max_val: 1.0 })
    .train_from_file("data.libsvm")?;
```

### Custom Kernels

```rust
use rsvm::kernel::LinearKernel;

// Using linear kernel explicitly
let model = SVM::with_kernel(LinearKernel::new())
    .with_c(1.0)
    .train_from_file("data.libsvm")?;
```

## Real-World Example

Let's work with a more realistic dataset - the classic Iris dataset adapted for binary classification.

### Preparing Iris Data

```rust
use std::fs::File;
use std::io::Write;

fn create_iris_data() -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create("iris_binary.csv")?;
    
    // Header
    writeln!(file, "sepal_length,sepal_width,petal_length,petal_width,class")?;
    
    // Setosa (class 1) vs others (class -1)
    // Setosa samples
    writeln!(file, "5.1,3.5,1.4,0.2,1")?;
    writeln!(file, "4.9,3.0,1.4,0.2,1")?;
    writeln!(file, "4.7,3.2,1.3,0.2,1")?;
    writeln!(file, "4.6,3.1,1.5,0.2,1")?;
    writeln!(file, "5.0,3.6,1.4,0.2,1")?;
    
    // Non-Setosa samples (Versicolor/Virginica)
    writeln!(file, "7.0,3.2,4.7,1.4,-1")?;
    writeln!(file, "6.4,3.2,4.5,1.5,-1")?;
    writeln!(file, "6.9,3.1,4.9,1.5,-1")?;
    writeln!(file, "5.5,2.3,4.0,1.3,-1")?;
    writeln!(file, "6.5,2.8,4.6,1.5,-1")?;
    
    Ok(())
}

fn iris_classification_example() -> Result<(), Box<dyn std::error::Error>> {
    create_iris_data()?;
    
    // Train the model
    let model = SVM::new()
        .with_c(10.0)  // Higher C for this dataset
        .train_from_csv("iris_binary.csv")?;
    
    // Evaluate
    let accuracy = model.evaluate_from_csv("iris_binary.csv")?;
    println!("Iris classification accuracy: {:.1}%", accuracy * 100.0);
    
    // Test with new samples
    let test_setosa = Sample::new(
        SparseVector::new(vec![0, 1, 2, 3], vec![5.2, 3.4, 1.4, 0.2]),
        0.0
    );
    
    let test_versicolor = Sample::new(
        SparseVector::new(vec![0, 1, 2, 3], vec![6.8, 3.0, 4.8, 1.4]),
        0.0
    );
    
    println!("Test Setosa prediction: {}", model.predict(&test_setosa).label);
    println!("Test Versicolor prediction: {}", model.predict(&test_versicolor).label);
    
    Ok(())
}
```

## Performance Tips

### 1. Feature Scaling

RSVM provides built-in feature scaling to improve numerical stability and convergence. Three scaling methods are available:

```rust
use rsvm::utils::scaling::ScalingMethod;

// Min-Max scaling to [-1, 1] range (default)
let model = SVM::new()
    .with_feature_scaling(ScalingMethod::MinMax { min_val: -1.0, max_val: 1.0 })
    .train_samples(&samples)?;

// Standard score (Z-score) normalization
let model = SVM::new()
    .with_feature_scaling(ScalingMethod::StandardScore)
    .train_samples(&samples)?;

// Unit scaling by maximum absolute value
let model = SVM::new()
    .with_feature_scaling(ScalingMethod::UnitScale)
    .train_samples(&samples)?;
```

**CLI Usage**:
```bash
# Min-Max scaling
rsvm train --data data.libsvm --output model.json --feature-scaling minmax

# Standard score normalization  
rsvm train --data data.libsvm --output model.json --feature-scaling standard

# Unit scaling
rsvm train --data data.libsvm --output model.json --feature-scaling unit
```

**Important**: When using scaling, the scaling parameters are automatically stored with the model and applied to prediction data. You don't need to manually scale test data.

### 2. Sparse Data

```rust
// Use sparse vectors efficiently
let sparse_sample = Sample::new(
    SparseVector::new(
        vec![5, 100, 1000],      // Only specify non-zero indices
        vec![1.5, 2.0, 0.8]      // Corresponding values
    ),
    1.0
);
```

### 3. Batch Operations

```rust
// Use batch prediction for multiple samples
let predictions = model.predict_batch(&test_samples);
```

### 4. Memory Management

```rust
// Estimate memory usage for large datasets
use rsvm::utils::memory;

let n_samples = 10000;
let estimated_memory = memory::estimate_kernel_cache_memory(n_samples);
println!("Estimated memory usage: {} MB", estimated_memory / (1024 * 1024));

// Set appropriate cache size
let cache_size = memory::recommend_cache_size(n_samples, 1000 * 1024 * 1024); // 1GB available
```

## Troubleshooting

### Common Issues

**1. Poor Accuracy**
```rust
// Try different C values
let c_values = vec![0.01, 0.1, 1.0, 10.0, 100.0];
// Check data quality and balance
// Consider feature scaling
```

**2. Slow Training**
```rust
// Reduce max_iterations for initial testing
let model = SVM::new()
    .with_max_iterations(100)  // Lower for testing
    .train_samples(&samples)?;

// Increase cache size for large datasets
let model = SVM::new()
    .with_cache_size(500 * 1024 * 1024)  // 500MB
    .train_samples(&samples)?;
```

**3. Memory Issues**
```rust
// For very large datasets, reduce cache size
let model = SVM::new()
    .with_cache_size(50 * 1024 * 1024)   // 50MB
    .train_samples(&samples)?;
```

**4. Convergence Issues**
```rust
// Adjust epsilon and max_iterations
let model = SVM::new()
    .with_epsilon(0.01)        // Less strict convergence
    .with_max_iterations(2000) // More iterations
    .train_samples(&samples)?;
```

### Error Messages

- `EmptyDataset`: Check that your data file exists and contains valid samples
- `ParseError`: Verify data format (LibSVM indices are 1-based, CSV last column is label)
- `InvalidParameter`: Check that C > 0, epsilon > 0, max_iterations > 0

### Debugging Tips

```rust
// Enable debug output
let info = model.info();
println!("Support vectors: {}", info.n_support_vectors);
println!("Support vector indices: {:?}", info.support_vector_indices);

// Check data loading
let dataset = LibSVMDataset::from_file("data.libsvm")?;
println!("Loaded {} samples with {} dimensions", dataset.len(), dataset.dim());
```

## Complete Example Program

Here's a complete example that demonstrates the entire workflow:

```rust
use rsvm::api::{SVM, quick};
use rsvm::{LibSVMDataset, Sample, SparseVector};
use std::fs::File;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("RSVM Tutorial Example");
    println!("=====================");
    
    // 1. Create sample data
    create_tutorial_dataset()?;
    
    // 2. Train model
    println!("\n1. Training model...");
    let model = SVM::new()
        .with_c(1.0)
        .with_epsilon(0.001)
        .train_from_file("tutorial_dataset.libsvm")?;
    
    println!("   ✓ Model trained successfully");
    println!("   Support vectors: {}", model.info().n_support_vectors);
    
    // 3. Evaluate model
    println!("\n2. Evaluating model...");
    let accuracy = model.evaluate_from_file("tutorial_dataset.libsvm")?;
    println!("   Accuracy: {:.1}%", accuracy * 100.0);
    
    // 4. Detailed metrics
    let dataset = LibSVMDataset::from_file("tutorial_dataset.libsvm")?;
    let metrics = model.evaluate_detailed(&dataset);
    println!("   Precision: {:.3}", metrics.precision());
    println!("   Recall: {:.3}", metrics.recall());
    println!("   F1 Score: {:.3}", metrics.f1_score());
    
    // 5. Test predictions
    println!("\n3. Testing predictions...");
    test_predictions(&model);
    
    // 6. Cross-validation
    println!("\n4. Cross-validation...");
    let cv_accuracy = quick::simple_validation(&dataset, 0.8, 1.0)?;
    println!("   Cross-validation accuracy: {:.1}%", cv_accuracy * 100.0);
    
    println!("\n✓ Tutorial completed successfully!");
    
    Ok(())
}

fn create_tutorial_dataset() -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create("tutorial_dataset.libsvm")?;
    
    // Create a 2D dataset that's linearly separable
    let positive_samples = vec![
        (2.0, 2.0), (2.1, 1.9), (1.9, 2.1), (2.2, 1.8),
        (1.8, 2.2), (2.3, 2.0), (2.0, 2.3), (1.7, 1.9),
    ];
    
    let negative_samples = vec![
        (-2.0, -2.0), (-2.1, -1.9), (-1.9, -2.1), (-2.2, -1.8),
        (-1.8, -2.2), (-2.3, -2.0), (-2.0, -2.3), (-1.7, -1.9),
    ];
    
    for (x, y) in positive_samples {
        writeln!(file, "+1 1:{} 2:{}", x, y)?;
    }
    
    for (x, y) in negative_samples {
        writeln!(file, "-1 1:{} 2:{}", x, y)?;
    }
    
    Ok(())
}

fn test_predictions(model: &rsvm::api::TrainedModel<rsvm::LinearKernel>) {
    let test_cases = vec![
        ((1.5, 1.5), "Positive region"),
        ((-1.5, -1.5), "Negative region"),
        ((0.5, 0.5), "Near boundary (positive side)"),
        ((-0.5, -0.5), "Near boundary (negative side)"),
        ((0.0, 0.0), "Origin"),
    ];
    
    for ((x, y), description) in test_cases {
        let test_sample = Sample::new(
            SparseVector::new(vec![0, 1], vec![x, y]),
            0.0
        );
        
        let prediction = model.predict(&test_sample);
        println!("   {} ({}, {}): {} (confidence: {:.3})", 
                description, x, y, 
                if prediction.label > 0.0 { "+" } else { "-" },
                prediction.confidence());
    }
}
```

## Summary

This tutorial covers the complete RSVM workflow from installation to advanced usage. Key takeaways:

- **CLI Tool**: Use for quick experimentation and production workflows
- **API**: Provides full programmatic control with builder pattern
- **Data Formats**: LibSVM and CSV both supported with auto-detection  
- **Performance**: Linear memory usage, efficient sparse operations
- **Evaluation**: Comprehensive metrics and cross-validation support

The RSVM library provides a production-ready SMO-based SVM implementation that's both user-friendly and technically sound. For advanced features like shrinking optimization and model reconstruction, see the implementation roadmap in [DESIGN.md](DESIGN.md).

For more examples and technical details:
- [CLI_EXAMPLES.md](CLI_EXAMPLES.md) - Comprehensive CLI usage examples
- [DESIGN.md](DESIGN.md) - Technical implementation details and paper comparison