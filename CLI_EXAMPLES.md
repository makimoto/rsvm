# RSVM CLI Examples

This document provides practical examples of using the RSVM command-line interface for common machine learning tasks.

## Table of Contents

1. [Basic Training](#basic-training)
2. [Model Evaluation](#model-evaluation)
3. [Making Predictions](#making-predictions)
4. [Quick Operations](#quick-operations)
5. [Working with Different Data Formats](#working-with-different-data-formats)
6. [Parameter Tuning](#parameter-tuning)
7. [Batch Processing](#batch-processing)

## Basic Training

### Simple Training

Train an SVM model with default parameters:

```bash
# Train on LibSVM format data
rsvm train --data data.libsvm --output model.json

# Train on CSV data with headers
rsvm train --data data.csv --output model.json --format csv

# Auto-detect format (recommended)
rsvm train --data data.libsvm --output model.json --format auto
```

### Training with Custom Parameters

```bash
# Higher regularization parameter
rsvm train --data data.libsvm --output model.json -C 10.0

# More strict convergence
rsvm train --data data.libsvm --output model.json --epsilon 0.0001

# Limit iterations for faster training
rsvm train --data data.libsvm --output model.json --max-iterations 500

# Increase cache for better performance
rsvm train --data data.libsvm --output model.json --cache-size 200

# Train with automatic feature scaling
rsvm train --data data.libsvm --output model.json --feature-scaling minmax

# Train with standard score normalization
rsvm train --data data.libsvm --output model.json --feature-scaling standard
```

### Verbose Training

Get detailed information during training:

```bash
rsvm train --data data.libsvm --output model.json --verbose
```

Example output:
```
[INFO] Training SVM model...
[INFO] Data file: "data.libsvm"
[INFO] Parameters: C=1, epsilon=0.001, max_iter=1000
[INFO] Loading dataset as libsvm format
[INFO] Loaded 150 samples with 4 dimensions
[INFO] Training completed successfully
[INFO] Support vectors: 23
[INFO] Bias: 0.123456
[INFO] Model saved to: "model.json"
[INFO] Training accuracy: 98.67%
```

## Model Evaluation

### Basic Model Information

```bash
# Display model summary
rsvm info model.json
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

### Evaluate on Test Data

```bash
# Basic evaluation
rsvm evaluate --model model.json --data test.libsvm

# Detailed metrics
rsvm evaluate --model model.json --data test.libsvm --detailed

# Specify data format explicitly
rsvm evaluate --model model.json --data test.csv --format csv
```

## Making Predictions

### Basic Predictions

```bash
# Predict labels only
rsvm predict --model model.json --data new_data.libsvm

# Include confidence scores
rsvm predict --model model.json --data new_data.libsvm --confidence

# Save predictions to file
rsvm predict --model model.json --data new_data.libsvm --output predictions.txt
```

Example prediction output:
```
# Predictions for 10 samples
# Format: sample_index predicted_label confidence
0 1 0.850
1 -1 0.750
2 1 0.920
...
```

## Quick Operations

### Train/Test Split Evaluation

Quickly evaluate performance with separate train and test files:

```bash
rsvm quick eval train.libsvm test.libsvm

# With custom C parameter
rsvm quick eval train.libsvm test.libsvm -C 5.0

# With feature scaling
rsvm quick eval train.libsvm test.libsvm --feature-scaling minmax
```

### Cross-Validation

Perform cross-validation on a single dataset:

```bash
# 80/20 split (default)
rsvm quick cv data.libsvm

# Custom split ratio
rsvm quick cv data.libsvm --ratio 0.7

# Different C parameter
rsvm quick cv data.libsvm --ratio 0.8 -C 2.0

# Cross-validation with feature scaling
rsvm quick cv data.libsvm --ratio 0.8 --feature-scaling standard

# Combine multiple options
rsvm quick cv data.libsvm --ratio 0.8 -C 5.0 --feature-scaling minmax
```

## Working with Different Data Formats

### LibSVM Format

```bash
# Train on LibSVM data
rsvm train --data iris.libsvm --output iris_model.json

# Example LibSVM file (iris.libsvm):
# +1 1:5.1 2:3.5 3:1.4 4:0.2
# +1 1:4.9 2:3.0 3:1.4 4:0.2
# -1 1:7.0 2:3.2 3:4.7 4:1.4
```

### CSV Format

```bash
# Train on CSV data with headers
rsvm train --data iris.csv --output iris_model.json --format csv

# Example CSV file (iris.csv):
# sepal_length,sepal_width,petal_length,petal_width,species
# 5.1,3.5,1.4,0.2,1
# 4.9,3.0,1.4,0.2,1
# 7.0,3.2,4.7,1.4,-1
```

### Auto-Detection

```bash
# Let RSVM detect the format automatically
rsvm train --data dataset.libsvm --output model.json  # Detects LibSVM
rsvm train --data dataset.csv --output model.json     # Detects CSV
rsvm train --data dataset.svm --output model.json     # Detects LibSVM
```

## Parameter Tuning

### C Parameter Exploration

```bash
# Test different C values
for c in 0.1 1.0 10.0 100.0; do
    echo "Testing C=$c"
    rsvm quick cv data.libsvm -C $c --ratio 0.8
done

# Test C values with feature scaling
for c in 0.1 1.0 10.0 100.0; do
    echo "Testing C=$c with StandardScore scaling"
    rsvm quick cv data.libsvm -C $c --ratio 0.8 --feature-scaling standard
done
```

### Convergence Tuning

```bash
# Strict convergence for final model
rsvm train --data data.libsvm --output final_model.json --epsilon 0.0001

# Fast training for experimentation
rsvm train --data data.libsvm --output test_model.json --epsilon 0.01 --max-iterations 100
```

## Batch Processing

### Process Multiple Datasets

```bash
#!/bin/bash
# Train models for multiple datasets

datasets=("dataset1.libsvm" "dataset2.libsvm" "dataset3.libsvm")
c_values=(0.1 1.0 10.0)

for dataset in "${datasets[@]}"; do
    for c in "${c_values[@]}"; do
        output="model_$(basename $dataset .libsvm)_C${c}.json"
        echo "Training $dataset with C=$c -> $output"
        rsvm train --data "$dataset" --output "$output" -C "$c" --verbose
    done
done
```

### Cross-Validation Grid Search

```bash
#!/bin/bash
# Grid search for best parameters

dataset="data.libsvm"
c_values=(0.01 0.1 1.0 10.0 100.0)
ratios=(0.7 0.8 0.9)

echo "C,Ratio,Accuracy" > results.csv

for c in "${c_values[@]}"; do
    for ratio in "${ratios[@]}"; do
        accuracy=$(rsvm quick cv "$dataset" -C "$c" --ratio "$ratio" | grep "CV accuracy" | cut -d: -f2 | tr -d ' %')
        echo "$c,$ratio,$accuracy" >> results.csv
    done
done
```

### Batch Prediction

```bash
#!/bin/bash
# Apply trained model to multiple test files

model="trained_model.json"
test_files=("test1.libsvm" "test2.libsvm" "test3.libsvm")

for test_file in "${test_files[@]}"; do
    output="predictions_$(basename $test_file .libsvm).txt"
    echo "Predicting $test_file -> $output"
    rsvm predict --model "$model" --data "$test_file" --output "$output" --confidence
done
```

## Advanced Examples

### Pipeline with Data Preprocessing

```bash
#!/bin/bash
# Complete ML pipeline

echo "Step 1: Data validation"
rsvm quick cv raw_data.csv --ratio 0.1  # Quick sanity check

echo "Step 2: Parameter tuning"
best_c=$(rsvm quick cv raw_data.csv -C 0.1 --ratio 0.8 | grep "CV accuracy" | cut -d: -f2)
# (In practice, you'd test multiple C values and pick the best)

echo "Step 3: Final training"
rsvm train --data raw_data.csv --output final_model.json -C 1.0 --verbose

echo "Step 4: Model validation"
rsvm evaluate --model final_model.json --data validation.csv --detailed

echo "Step 5: Production predictions"
rsvm predict --model final_model.json --data production_data.csv --output production_predictions.txt --confidence
```

### Model Comparison

```bash
#!/bin/bash
# Compare models with different parameters

data="dataset.libsvm"
test_data="test.libsvm"

echo "Model,C,Support_Vectors,Test_Accuracy" > comparison.csv

for c in 0.1 1.0 10.0; do
    model="model_C${c}.json"
    
    # Train model
    rsvm train --data "$data" --output "$model" -C "$c" --verbose
    
    # Extract support vector count
    sv_count=$(rsvm info "$model" | grep "Support Vectors:" | cut -d: -f2 | tr -d ' ')
    
    # Evaluate on test data
    # Note: This would work once model reconstruction is implemented
    # For now, this is a placeholder showing the intended workflow
    test_acc="N/A"
    
    echo "Model_C${c},$c,$sv_count,$test_acc" >> comparison.csv
done
```

## Troubleshooting

### Common Issues and Solutions

#### Format Detection Problems
```bash
# Explicitly specify format if auto-detection fails
rsvm train --data unclear_file.txt --output model.json --format libsvm
```

#### Memory Issues
```bash
# Reduce cache size for large datasets
rsvm train --data large_data.libsvm --output model.json --cache-size 50
```

#### Convergence Issues
```bash
# Relax convergence criteria
rsvm train --data difficult_data.libsvm --output model.json --epsilon 0.01 --max-iterations 2000
```

#### Debug Mode
```bash
# Enable debug output for troubleshooting
rsvm train --data data.libsvm --output model.json --debug
```

This CLI provides a complete interface for SVM training and evaluation. The examples above demonstrate the flexibility and power of the RSVM command-line tool for both interactive use and automated pipelines.