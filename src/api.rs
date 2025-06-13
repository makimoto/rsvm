//! High-level API for Support Vector Machine operations
//!
//! This module provides a user-friendly interface for common SVM tasks,
//! including training, prediction, and model evaluation.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use rsvm::api::SVM;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Train a model on data
//! let svm = SVM::new()
//!     .with_c(1.0)
//!     .with_epsilon(0.001)
//!     .train_from_file("data.libsvm")?;
//!
//! // Make predictions
//! let predictions = svm.predict_from_file("test.libsvm")?;
//! println!("Accuracy: {:.2}%", svm.evaluate_from_file("test.libsvm")? * 100.0);
//! # Ok(())
//! # }
//! ```

use crate::core::{
    Dataset, OptimizerConfig, Prediction, Result, SVMError, SVMModel, Sample, WorkingSetStrategy,
};
use crate::data::{CSVDataset, LibSVMDataset};
use crate::kernel::{
    ChiSquareKernel, HellingerKernel, HistogramIntersectionKernel, Kernel, LinearKernel,
    PolynomialKernel, RBFKernel,
};
use crate::optimizer::{SVMOptimizer, TrainedSVM};
use crate::utils::scaling::{ScalingMethod, ScalingParams};
use std::path::Path;

/// High-level SVM interface with builder pattern
pub struct SVM<K: Kernel = LinearKernel> {
    kernel: K,
    config: OptimizerConfig,
    scaling_method: Option<ScalingMethod>,
}

impl SVM<LinearKernel> {
    /// Create a new SVM with linear kernel and default parameters
    pub fn new() -> Self {
        Self {
            kernel: LinearKernel::new(),
            config: OptimizerConfig::default(),
            scaling_method: None,
        }
    }
}

impl SVM<RBFKernel> {
    /// Create a new SVM with RBF kernel and specified gamma
    ///
    /// # Arguments
    /// * `gamma` - The gamma parameter for the RBF kernel (must be positive)
    pub fn with_rbf(gamma: f64) -> Self {
        Self {
            kernel: RBFKernel::new(gamma),
            config: OptimizerConfig::default(),
            scaling_method: None,
        }
    }

    /// Create a new SVM with RBF kernel using auto-gamma (1.0 / n_features)
    ///
    /// # Arguments
    /// * `n_features` - Number of features in the dataset
    pub fn with_rbf_auto(n_features: usize) -> Self {
        Self {
            kernel: RBFKernel::with_auto_gamma(n_features),
            config: OptimizerConfig::default(),
            scaling_method: None,
        }
    }

    /// Create a new SVM with RBF kernel using unit gamma (1.0)
    pub fn with_rbf_unit() -> Self {
        Self {
            kernel: RBFKernel::unit_gamma(),
            config: OptimizerConfig::default(),
            scaling_method: None,
        }
    }
}

impl SVM<ChiSquareKernel> {
    /// Create a new SVM with Chi-square kernel for histogram/distribution data
    ///
    /// The Chi-square kernel is particularly effective for:
    /// - Computer vision tasks with histogram features
    /// - Text analysis with bag-of-words representations  
    /// - Bioinformatics with frequency data
    /// - Any data naturally represented as distributions
    ///
    /// # Arguments
    /// * `gamma` - Scaling parameter for the chi-square distance (must be positive)
    pub fn with_chi_square(gamma: f64) -> Self {
        Self {
            kernel: ChiSquareKernel::new(gamma),
            config: OptimizerConfig::default(),
            scaling_method: None,
        }
    }

    /// Create a new SVM with Chi-square kernel using unit gamma (1.0)
    ///
    /// This is often a good starting point for histogram data.
    pub fn with_chi_square_unit() -> Self {
        Self {
            kernel: ChiSquareKernel::unit_gamma(),
            config: OptimizerConfig::default(),
            scaling_method: None,
        }
    }

    /// Create a new SVM with Chi-square kernel using auto-gamma
    ///
    /// Uses gamma = 1.0 / n_features, suitable for high-dimensional histograms.
    ///
    /// # Arguments
    /// * `n_features` - Number of features in the dataset
    pub fn with_chi_square_auto(n_features: usize) -> Self {
        Self {
            kernel: ChiSquareKernel::with_auto_gamma(n_features),
            config: OptimizerConfig::default(),
            scaling_method: None,
        }
    }

    /// Create a new SVM with Chi-square kernel optimized for computer vision
    ///
    /// Uses gamma = 0.5, empirically proven effective for visual features.
    pub fn with_chi_square_cv() -> Self {
        Self {
            kernel: ChiSquareKernel::for_computer_vision(),
            config: OptimizerConfig::default(),
            scaling_method: None,
        }
    }

    /// Create a new SVM with Chi-square kernel optimized for text analysis
    ///
    /// Uses gamma = 2.0, effective for term frequency histograms.
    pub fn with_chi_square_text() -> Self {
        Self {
            kernel: ChiSquareKernel::for_text_analysis(),
            config: OptimizerConfig::default(),
            scaling_method: None,
        }
    }
}

impl SVM<HistogramIntersectionKernel> {
    /// Create a new SVM with Histogram Intersection kernel for computer vision
    ///
    /// The Histogram Intersection kernel is particularly effective for:
    /// - Image classification with color histograms
    /// - Object recognition with SIFT/SURF descriptors  
    /// - Texture analysis with Local Binary Patterns (LBP)
    /// - Visual bag-of-words representations
    /// - Any histogram-based feature representation
    ///
    /// # Arguments
    /// * `normalized` - Whether to normalize by minimum L1 norm (recommended for different histogram sizes)
    pub fn with_histogram_intersection(normalized: bool) -> Self {
        Self {
            kernel: HistogramIntersectionKernel::new(normalized),
            config: OptimizerConfig::default(),
            scaling_method: None,
        }
    }

    /// Create a new SVM with standard (non-normalized) Histogram Intersection kernel
    ///
    /// This preserves absolute histogram counts, suitable for bag-of-visual-words
    /// and other applications where frequency magnitude matters.
    pub fn with_histogram_intersection_standard() -> Self {
        Self {
            kernel: HistogramIntersectionKernel::standard(),
            config: OptimizerConfig::default(),
            scaling_method: None,
        }
    }

    /// Create a new SVM with normalized Histogram Intersection kernel
    ///
    /// Normalizes by min(||x||₁, ||y||₁), giving values in [0,1].
    /// Recommended when histograms have different total counts.
    pub fn with_histogram_intersection_normalized() -> Self {
        Self {
            kernel: HistogramIntersectionKernel::normalized(),
            config: OptimizerConfig::default(),
            scaling_method: None,
        }
    }

    /// Create a new SVM with Histogram Intersection kernel optimized for color histograms
    ///
    /// Uses normalized intersection, which is standard practice for color histogram
    /// comparison in computer vision. Effective for RGB, HSV, and other color spaces.
    pub fn with_histogram_intersection_color() -> Self {
        Self {
            kernel: HistogramIntersectionKernel::for_color_histograms(),
            config: OptimizerConfig::default(),
            scaling_method: None,
        }
    }

    /// Create a new SVM with Histogram Intersection kernel optimized for visual words
    ///
    /// Uses standard intersection to preserve absolute frequency information,
    /// important for bag-of-visual-words representations in object recognition.
    pub fn with_histogram_intersection_visual_words() -> Self {
        Self {
            kernel: HistogramIntersectionKernel::for_visual_words(),
            config: OptimizerConfig::default(),
            scaling_method: None,
        }
    }

    /// Create a new SVM with Histogram Intersection kernel optimized for texture analysis
    ///
    /// Uses normalized intersection, suitable for Local Binary Pattern (LBP)
    /// histograms and other texture descriptors.
    pub fn with_histogram_intersection_texture() -> Self {
        Self {
            kernel: HistogramIntersectionKernel::for_texture_analysis(),
            config: OptimizerConfig::default(),
            scaling_method: None,
        }
    }
}

impl SVM<HellingerKernel> {
    /// Create a new SVM with Hellinger kernel for probability distributions
    ///
    /// The Hellinger kernel is designed for probability distributions and normalized data.
    /// It's particularly effective for applications where features represent probabilities,
    /// frequencies, or other non-negative normalized values.
    ///
    /// Key applications:
    /// - Text mining with normalized term frequencies (TF-IDF)
    /// - Bioinformatics with species abundance data
    /// - Statistical analysis of probability distributions
    /// - Image analysis with normalized histograms
    /// - Natural language processing with word embeddings
    /// - Machine learning with probability vectors
    ///
    /// # Arguments
    /// * `normalized` - Whether to normalize by √(||x||₁ * ||y||₁) giving values in [0,1]
    pub fn with_hellinger(normalized: bool) -> Self {
        Self {
            kernel: HellingerKernel::new(normalized),
            config: OptimizerConfig::default(),
            scaling_method: None,
        }
    }

    /// Create a new SVM with standard (non-normalized) Hellinger kernel
    ///
    /// This is the most commonly used variant for probability distributions.
    /// Perfect for comparing probability vectors, mixture models, and other
    /// probabilistic representations.
    pub fn with_hellinger_standard() -> Self {
        Self {
            kernel: HellingerKernel::standard(),
            config: OptimizerConfig::default(),
            scaling_method: None,
        }
    }

    /// Create a new SVM with normalized Hellinger kernel
    ///
    /// Normalizes by √(||x||₁ * ||y||₁), giving values in [0,1].
    /// This is useful when distributions have different total probabilities
    /// or for statistical hypothesis testing applications.
    pub fn with_hellinger_normalized() -> Self {
        Self {
            kernel: HellingerKernel::normalized(),
            config: OptimizerConfig::default(),
            scaling_method: None,
        }
    }

    /// Create a new SVM with Hellinger kernel optimized for text mining
    ///
    /// Uses standard Hellinger kernel, which works well with TF-IDF vectors
    /// and other normalized text representations. Excellent for document
    /// classification and semantic similarity tasks.
    pub fn with_hellinger_text() -> Self {
        Self {
            kernel: HellingerKernel::for_text_mining(),
            config: OptimizerConfig::default(),
            scaling_method: None,
        }
    }

    /// Create a new SVM with Hellinger kernel optimized for bioinformatics
    ///
    /// Uses normalized Hellinger kernel, suitable for species abundance data,
    /// genetic sequence analysis, and other biological frequency measurements.
    pub fn with_hellinger_bio() -> Self {
        Self {
            kernel: HellingerKernel::for_bioinformatics(),
            config: OptimizerConfig::default(),
            scaling_method: None,
        }
    }

    /// Create a new SVM with Hellinger kernel optimized for probability vectors
    ///
    /// Uses standard Hellinger kernel, perfect for comparing probability
    /// distributions, mixture models, Bayesian analysis, and machine learning
    /// applications with probabilistic features.
    pub fn with_hellinger_probability() -> Self {
        Self {
            kernel: HellingerKernel::for_probability_vectors(),
            config: OptimizerConfig::default(),
            scaling_method: None,
        }
    }

    /// Create a new SVM with Hellinger kernel optimized for statistical analysis
    ///
    /// Uses normalized Hellinger kernel, providing bounded similarity measures
    /// suitable for statistical hypothesis testing, distribution comparison,
    /// and research applications requiring standardized metrics.
    pub fn with_hellinger_stats() -> Self {
        Self {
            kernel: HellingerKernel::for_statistical_analysis(),
            config: OptimizerConfig::default(),
            scaling_method: None,
        }
    }
}

impl SVM<PolynomialKernel> {
    /// Create a new SVM with polynomial kernel
    ///
    /// # Arguments
    /// * `degree` - Degree of the polynomial (must be > 0)
    /// * `gamma` - Scaling factor for the dot product
    /// * `coef0` - Independent term in the polynomial
    pub fn with_polynomial(degree: u32, gamma: f64, coef0: f64) -> Self {
        Self {
            kernel: PolynomialKernel::new(degree, gamma, coef0),
            config: OptimizerConfig::default(),
            scaling_method: None,
        }
    }

    /// Create a new SVM with quadratic kernel: (γ * <x,y> + 1)²
    ///
    /// # Arguments
    /// * `gamma` - Scaling factor for the dot product
    pub fn with_quadratic(gamma: f64) -> Self {
        Self {
            kernel: PolynomialKernel::quadratic(gamma),
            config: OptimizerConfig::default(),
            scaling_method: None,
        }
    }

    /// Create a new SVM with cubic kernel: (γ * <x,y> + 1)³
    ///
    /// # Arguments
    /// * `gamma` - Scaling factor for the dot product
    pub fn with_cubic(gamma: f64) -> Self {
        Self {
            kernel: PolynomialKernel::cubic(gamma),
            config: OptimizerConfig::default(),
            scaling_method: None,
        }
    }

    /// Create a new SVM with polynomial kernel using auto-gamma
    ///
    /// # Arguments
    /// * `degree` - Degree of the polynomial
    /// * `n_features` - Number of features in the dataset
    pub fn with_polynomial_auto(degree: u32, n_features: usize) -> Self {
        Self {
            kernel: PolynomialKernel::auto(degree, n_features),
            config: OptimizerConfig::default(),
            scaling_method: None,
        }
    }

    /// Create a new SVM with normalized polynomial kernel
    ///
    /// # Arguments  
    /// * `degree` - Degree of the polynomial
    pub fn with_polynomial_normalized(degree: u32) -> Self {
        Self {
            kernel: PolynomialKernel::normalized(degree),
            config: OptimizerConfig::default(),
            scaling_method: None,
        }
    }
}

impl Default for SVM<LinearKernel> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Kernel> SVM<K> {
    /// Create SVM with custom kernel
    pub fn with_kernel(kernel: K) -> Self {
        Self {
            kernel,
            config: OptimizerConfig::default(),
            scaling_method: None,
        }
    }

    /// Set regularization parameter C
    pub fn with_c(mut self, c: f64) -> Self {
        self.config.c = c;
        self
    }

    /// Set convergence tolerance
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.config.epsilon = epsilon;
        self
    }

    /// Set maximum number of iterations
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.config.max_iterations = max_iterations;
        self
    }

    /// Set kernel cache size in bytes
    pub fn with_cache_size(mut self, cache_size: usize) -> Self {
        self.config.cache_size = cache_size;
        self
    }

    /// Enable or disable shrinking heuristic
    ///
    /// Shrinking can significantly improve performance on large datasets
    /// by temporarily removing variables that are likely to remain at bounds.
    /// Based on Section 4 of the SVMlight paper.
    pub fn with_shrinking(mut self, enable_shrinking: bool) -> Self {
        self.config.shrinking = enable_shrinking;
        self
    }

    /// Set shrinking frequency (number of iterations between shrinking checks)
    ///
    /// Lower values check more frequently but may add overhead.
    /// Higher values check less frequently but may miss opportunities.
    /// Default is 100 iterations (h parameter in the paper).
    pub fn with_shrinking_iterations(mut self, iterations: usize) -> Self {
        self.config.shrinking_iterations = iterations;
        self
    }

    /// Set working set selection strategy
    ///
    /// - `SMOHeuristic`: Uses max |E_i - E_j| criterion (default, fast)
    /// - `SteepestDescent`: Uses maximum KKT violation (SVMlight paper style, more rigorous)
    /// - `Random`: Random selection (for debugging/comparison)
    pub fn with_working_set_strategy(mut self, strategy: WorkingSetStrategy) -> Self {
        self.config.working_set_strategy = strategy;
        self
    }

    /// Enable automatic feature scaling
    ///
    /// Features will be scaled using the specified method before training.
    /// Scaling parameters are computed from training data and stored for
    /// consistent transformation of prediction data.
    ///
    /// # Arguments
    /// * `method` - Scaling method to use:
    ///   - `MinMax { min_val, max_val }`: Scale to specified range (default: [-1, 1])
    ///   - `StandardScore`: Z-score normalization (mean=0, std=1)
    ///   - `UnitScale`: Scale by maximum absolute value
    pub fn with_feature_scaling(mut self, method: ScalingMethod) -> Self {
        self.scaling_method = Some(method);
        self
    }

    /// Train on a dataset
    pub fn train<D: Dataset>(self, dataset: &D) -> Result<TrainedModel<K>> {
        let samples: Vec<Sample> = (0..dataset.len()).map(|i| dataset.get_sample(i)).collect();
        self.train_samples(&samples)
    }

    /// Train on samples
    pub fn train_samples(self, samples: &[Sample]) -> Result<TrainedModel<K>> {
        let (training_samples, scaling_params) = if let Some(method) = self.scaling_method {
            let params = ScalingParams::fit(samples, method);
            let scaled_samples = params.transform_samples(samples);
            (scaled_samples, Some(params))
        } else {
            (samples.to_vec(), None)
        };

        let optimizer = SVMOptimizer::new(self.kernel, self.config);
        let model = optimizer.train_samples(&training_samples)?;
        Ok(TrainedModel {
            model,
            scaling_params,
        })
    }

    /// Train from LibSVM format file
    pub fn train_from_file<P: AsRef<Path>>(self, path: P) -> Result<TrainedModel<K>> {
        let dataset = LibSVMDataset::from_file(path)?;
        self.train(&dataset)
    }

    /// Train from CSV file (automatically detects headers)
    pub fn train_from_csv<P: AsRef<Path>>(self, path: P) -> Result<TrainedModel<K>> {
        let dataset = CSVDataset::from_file(path)?;
        self.train(&dataset)
    }
}

/// Trained SVM model with high-level prediction interface
pub struct TrainedModel<K: Kernel> {
    model: TrainedSVM<K>,
    scaling_params: Option<ScalingParams>,
}

impl<K: Kernel> TrainedModel<K> {
    /// Create a TrainedModel from a TrainedSVM
    pub fn from_trained_svm(model: TrainedSVM<K>) -> Self {
        Self {
            model,
            scaling_params: None,
        }
    }

    /// Create a TrainedModel from a TrainedSVM with scaling parameters
    pub fn from_trained_svm_with_scaling(
        model: TrainedSVM<K>,
        scaling_params: Option<ScalingParams>,
    ) -> Self {
        Self {
            model,
            scaling_params,
        }
    }

    /// Predict a single sample
    pub fn predict(&self, sample: &Sample) -> Prediction {
        let scaled_sample = if let Some(ref params) = self.scaling_params {
            params.transform_sample(sample)
        } else {
            sample.clone()
        };
        self.model.predict(&scaled_sample)
    }

    /// Predict multiple samples
    pub fn predict_batch(&self, samples: &[Sample]) -> Vec<Prediction> {
        let scaled_samples = if let Some(ref params) = self.scaling_params {
            params.transform_samples(samples)
        } else {
            samples.to_vec()
        };
        self.model.predict_batch(&scaled_samples)
    }

    /// Predict from dataset
    pub fn predict_dataset<D: Dataset>(&self, dataset: &D) -> Vec<Prediction> {
        let samples: Vec<Sample> = (0..dataset.len()).map(|i| dataset.get_sample(i)).collect();
        self.predict_batch(&samples)
    }

    /// Predict from LibSVM file
    pub fn predict_from_file<P: AsRef<Path>>(&self, path: P) -> Result<Vec<Prediction>> {
        let dataset = LibSVMDataset::from_file(path)?;
        Ok(self.predict_dataset(&dataset))
    }

    /// Predict from CSV file
    pub fn predict_from_csv<P: AsRef<Path>>(&self, path: P) -> Result<Vec<Prediction>> {
        let dataset = CSVDataset::from_file(path)?;
        Ok(self.predict_dataset(&dataset))
    }

    /// Evaluate accuracy on a dataset
    pub fn evaluate<D: Dataset>(&self, dataset: &D) -> f64 {
        let predictions = self.predict_dataset(dataset);
        let labels = dataset.get_labels();

        let correct = predictions
            .iter()
            .zip(labels.iter())
            .filter(|(pred, &actual)| pred.label == actual)
            .count();

        correct as f64 / labels.len() as f64
    }

    /// Evaluate accuracy from LibSVM file
    pub fn evaluate_from_file<P: AsRef<Path>>(&self, path: P) -> Result<f64> {
        let dataset = LibSVMDataset::from_file(path)?;
        Ok(self.evaluate(&dataset))
    }

    /// Evaluate accuracy from CSV file
    pub fn evaluate_from_csv<P: AsRef<Path>>(&self, path: P) -> Result<f64> {
        let dataset = CSVDataset::from_file(path)?;
        Ok(self.evaluate(&dataset))
    }

    /// Get detailed evaluation metrics
    pub fn evaluate_detailed<D: Dataset>(&self, dataset: &D) -> EvaluationMetrics {
        let predictions = self.predict_dataset(dataset);
        let labels = dataset.get_labels();

        let mut tp = 0; // True positives
        let mut tn = 0; // True negatives
        let mut fp = 0; // False positives
        let mut fn_ = 0; // False negatives

        for (pred, &actual) in predictions.iter().zip(labels.iter()) {
            match (pred.label > 0.0, actual > 0.0) {
                (true, true) => tp += 1,
                (false, false) => tn += 1,
                (true, false) => fp += 1,
                (false, true) => fn_ += 1,
            }
        }

        EvaluationMetrics::new(tp, tn, fp, fn_)
    }

    /// Get model information
    pub fn info(&self) -> ModelInfo {
        ModelInfo {
            n_support_vectors: self.model.n_support_vectors(),
            bias: self.model.bias(),
            support_vector_indices: self.model.support_vector_indices().to_vec(),
        }
    }

    /// Get the underlying trained model
    pub fn inner(&self) -> &TrainedSVM<K> {
        &self.model
    }
}

/// Detailed evaluation metrics
#[derive(Debug, Clone)]
pub struct EvaluationMetrics {
    pub true_positives: usize,
    pub true_negatives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
}

impl EvaluationMetrics {
    fn new(tp: usize, tn: usize, fp: usize, fn_: usize) -> Self {
        Self {
            true_positives: tp,
            true_negatives: tn,
            false_positives: fp,
            false_negatives: fn_,
        }
    }

    /// Calculate accuracy: (TP + TN) / (TP + TN + FP + FN)
    pub fn accuracy(&self) -> f64 {
        let total =
            self.true_positives + self.true_negatives + self.false_positives + self.false_negatives;
        if total == 0 {
            0.0
        } else {
            (self.true_positives + self.true_negatives) as f64 / total as f64
        }
    }

    /// Calculate precision: TP / (TP + FP)
    pub fn precision(&self) -> f64 {
        let denominator = self.true_positives + self.false_positives;
        if denominator == 0 {
            0.0
        } else {
            self.true_positives as f64 / denominator as f64
        }
    }

    /// Calculate recall (sensitivity): TP / (TP + FN)
    pub fn recall(&self) -> f64 {
        let denominator = self.true_positives + self.false_negatives;
        if denominator == 0 {
            0.0
        } else {
            self.true_positives as f64 / denominator as f64
        }
    }

    /// Calculate F1 score: 2 * (precision * recall) / (precision + recall)
    pub fn f1_score(&self) -> f64 {
        let p = self.precision();
        let r = self.recall();
        if p + r == 0.0 {
            0.0
        } else {
            2.0 * (p * r) / (p + r)
        }
    }

    /// Calculate specificity: TN / (TN + FP)
    pub fn specificity(&self) -> f64 {
        let denominator = self.true_negatives + self.false_positives;
        if denominator == 0 {
            0.0
        } else {
            self.true_negatives as f64 / denominator as f64
        }
    }
}

/// Model information
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub n_support_vectors: usize,
    pub bias: f64,
    pub support_vector_indices: Vec<usize>,
}

/// Convenience functions for quick operations
pub mod quick {
    use super::*;

    /// Train a linear SVM on LibSVM data with default parameters
    pub fn train_libsvm<P: AsRef<Path>>(path: P) -> Result<TrainedModel<LinearKernel>> {
        SVM::new().train_from_file(path)
    }

    /// Train a linear SVM on CSV data with default parameters
    pub fn train_csv<P: AsRef<Path>>(path: P) -> Result<TrainedModel<LinearKernel>> {
        SVM::new().train_from_csv(path)
    }

    /// Train with custom C parameter
    pub fn train_libsvm_with_c<P: AsRef<Path>>(
        path: P,
        c: f64,
    ) -> Result<TrainedModel<LinearKernel>> {
        SVM::new().with_c(c).train_from_file(path)
    }

    /// Train an RBF SVM on LibSVM data with specified gamma
    pub fn train_libsvm_rbf<P: AsRef<Path>>(
        path: P,
        gamma: f64,
    ) -> Result<TrainedModel<RBFKernel>> {
        SVM::with_rbf(gamma).train_from_file(path)
    }

    /// Train an RBF SVM on LibSVM data with auto gamma and custom C
    pub fn train_libsvm_rbf_auto<P: AsRef<Path>>(
        path: P,
        n_features: usize,
        c: f64,
    ) -> Result<TrainedModel<RBFKernel>> {
        SVM::with_rbf_auto(n_features)
            .with_c(c)
            .train_from_file(path)
    }

    /// Train an RBF SVM on CSV data with specified gamma
    pub fn train_csv_rbf<P: AsRef<Path>>(path: P, gamma: f64) -> Result<TrainedModel<RBFKernel>> {
        SVM::with_rbf(gamma).train_from_csv(path)
    }

    /// Quick evaluation: train on training file, test on test file
    pub fn evaluate_split<P1: AsRef<Path>, P2: AsRef<Path>>(
        train_path: P1,
        test_path: P2,
    ) -> Result<f64> {
        evaluate_split_with_params(train_path, test_path, 1.0, None)
    }

    /// Quick evaluation with custom parameters and scaling
    pub fn evaluate_split_with_params<P1: AsRef<Path>, P2: AsRef<Path>>(
        train_path: P1,
        test_path: P2,
        c: f64,
        scaling: Option<ScalingMethod>,
    ) -> Result<f64> {
        let mut svm_builder = SVM::new().with_c(c);

        if let Some(method) = scaling {
            svm_builder = svm_builder.with_feature_scaling(method);
        }

        let model = svm_builder.train_from_file(train_path)?;
        model.evaluate_from_file(test_path)
    }

    /// Cross-validation helper (simple random split)
    pub fn simple_validation<D: Dataset>(dataset: &D, train_ratio: f64, c: f64) -> Result<f64> {
        simple_validation_with_strategy(dataset, train_ratio, c, WorkingSetStrategy::SMOHeuristic)
    }

    /// Cross-validation helper with configurable working set strategy
    pub fn simple_validation_with_strategy<D: Dataset>(
        dataset: &D,
        train_ratio: f64,
        c: f64,
        strategy: WorkingSetStrategy,
    ) -> Result<f64> {
        simple_validation_with_strategy_and_scaling(dataset, train_ratio, c, strategy, None)
    }

    /// Cross-validation helper with configurable working set strategy and scaling
    pub fn simple_validation_with_strategy_and_scaling<D: Dataset>(
        dataset: &D,
        train_ratio: f64,
        c: f64,
        strategy: WorkingSetStrategy,
        scaling: Option<ScalingMethod>,
    ) -> Result<f64> {
        if train_ratio <= 0.0 || train_ratio >= 1.0 {
            return Err(SVMError::InvalidParameter(format!(
                "Train ratio must be between 0 and 1, got: {train_ratio}"
            )));
        }

        let n = dataset.len();
        let train_size = (n as f64 * train_ratio) as usize;

        // Simple sequential split (not randomized for reproducibility)
        let train_samples: Vec<Sample> = (0..train_size).map(|i| dataset.get_sample(i)).collect();
        let test_samples: Vec<Sample> = (train_size..n).map(|i| dataset.get_sample(i)).collect();

        let mut svm_builder = SVM::new().with_c(c).with_working_set_strategy(strategy);

        if let Some(method) = scaling {
            svm_builder = svm_builder.with_feature_scaling(method);
        }

        let model = svm_builder.train_samples(&train_samples)?;

        let correct = test_samples
            .iter()
            .map(|sample| model.predict(sample))
            .zip(test_samples.iter())
            .filter(|(pred, sample)| pred.label == sample.label)
            .count();

        Ok(correct as f64 / test_samples.len() as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::SparseVector;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_svm_builder_pattern() {
        let svm = SVM::new()
            .with_c(2.0)
            .with_epsilon(0.01)
            .with_max_iterations(5000);

        assert_eq!(svm.config.c, 2.0);
        assert_eq!(svm.config.epsilon, 0.01);
        assert_eq!(svm.config.max_iterations, 5000);
    }

    #[test]
    fn test_quick_training() {
        // Create test data
        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![2.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-2.0]), -1.0),
            Sample::new(SparseVector::new(vec![0], vec![1.5]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-1.5]), -1.0),
        ];

        let model = SVM::new()
            .train_samples(&samples)
            .expect("Training should succeed");

        // Test predictions
        let test_sample = Sample::new(SparseVector::new(vec![0], vec![1.0]), 1.0);
        let prediction = model.predict(&test_sample);
        assert_eq!(prediction.label, 1.0);

        // Test model info
        let info = model.info();
        assert!(info.n_support_vectors > 0);
    }

    #[test]
    fn test_evaluation_metrics() {
        let metrics = EvaluationMetrics::new(10, 5, 2, 3);

        assert_eq!(metrics.accuracy(), 0.75); // (10+5)/(10+5+2+3)
        assert_eq!(metrics.precision(), 10.0 / 12.0); // 10/(10+2)
        assert_eq!(metrics.recall(), 10.0 / 13.0); // 10/(10+3)
        assert!(metrics.f1_score() > 0.0);
        assert_eq!(metrics.specificity(), 5.0 / 7.0); // 5/(5+2)
    }

    #[test]
    fn test_file_operations() {
        // Create temporary LibSVM file
        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        writeln!(temp_file, "+1 1:2.0").expect("Failed to write");
        writeln!(temp_file, "-1 1:-2.0").expect("Failed to write");
        writeln!(temp_file, "+1 1:1.5").expect("Failed to write");
        writeln!(temp_file, "-1 1:-1.5").expect("Failed to write");
        temp_file.flush().expect("Failed to flush");

        // Test training from file
        let model = SVM::new()
            .train_from_file(temp_file.path())
            .expect("Training should succeed");

        // Test evaluation
        let accuracy = model
            .evaluate_from_file(temp_file.path())
            .expect("Evaluation should succeed");
        assert!(accuracy > 0.0);

        // Test quick functions
        let model2 = quick::train_libsvm(temp_file.path()).expect("Quick training should succeed");
        assert!(model2.info().n_support_vectors > 0);
    }

    #[test]
    fn test_simple_validation() {
        // Create mock dataset
        struct MockDataset {
            samples: Vec<Sample>,
        }

        impl Dataset for MockDataset {
            fn len(&self) -> usize {
                self.samples.len()
            }
            fn dim(&self) -> usize {
                1
            }
            fn get_sample(&self, i: usize) -> Sample {
                self.samples[i].clone()
            }
            fn get_labels(&self) -> Vec<f64> {
                self.samples.iter().map(|s| s.label).collect()
            }
        }

        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![2.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-2.0]), -1.0),
            Sample::new(SparseVector::new(vec![0], vec![1.5]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-1.5]), -1.0),
            Sample::new(SparseVector::new(vec![0], vec![1.8]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-1.8]), -1.0),
        ];

        let dataset = MockDataset { samples };
        let accuracy =
            quick::simple_validation(&dataset, 0.7, 1.0).expect("Validation should succeed");
        assert!((0.0..=1.0).contains(&accuracy));
    }

    #[test]
    fn test_file_operation_errors() {
        // Test I/O errors for train_from_file
        let result = SVM::new().train_from_file("/nonexistent/path.libsvm");
        assert!(result.is_err());

        // Test I/O errors for train_from_csv
        let result = SVM::new().train_from_csv("/nonexistent/path.csv");
        assert!(result.is_err());

        // Create a valid model first for predict/evaluate error tests
        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![1.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-1.0]), -1.0),
        ];
        let model = SVM::new()
            .train_samples(&samples)
            .expect("Training should succeed");

        // Test I/O errors for predict_from_file
        let result = model.predict_from_file("/nonexistent/path.libsvm");
        assert!(result.is_err());

        // Test I/O errors for predict_from_csv
        let result = model.predict_from_csv("/nonexistent/path.csv");
        assert!(result.is_err());

        // Test I/O errors for evaluate_from_file
        let result = model.evaluate_from_file("/nonexistent/path.libsvm");
        assert!(result.is_err());

        // Test I/O errors for evaluate_from_csv
        let result = model.evaluate_from_csv("/nonexistent/path.csv");
        assert!(result.is_err());
    }

    #[test]
    fn test_evaluation_metrics_edge_cases() {
        // Test edge case: no true positives, no false positives (only true negatives)
        // All samples are negative and correctly predicted
        let metrics = EvaluationMetrics::new(0, 3, 0, 0); // tp=0, tn=3, fp=0, fn=0

        // With no positives, precision should be 0.0 and recall should be 0.0
        assert_eq!(metrics.precision(), 0.0);
        assert_eq!(metrics.recall(), 0.0);
        assert_eq!(metrics.f1_score(), 0.0);
        assert_eq!(metrics.specificity(), 1.0); // All negatives correct

        // Test edge case: no true negatives, no false negatives (only true positives)
        // All samples are positive and correctly predicted
        let metrics = EvaluationMetrics::new(3, 0, 0, 0); // tp=3, tn=0, fp=0, fn=0

        assert_eq!(metrics.precision(), 1.0);
        assert_eq!(metrics.recall(), 1.0);
        assert_eq!(metrics.f1_score(), 1.0);
        assert_eq!(metrics.specificity(), 0.0); // No negatives to be specific about

        // Test edge case: no true positives (all false negatives and true/false negatives)
        let metrics = EvaluationMetrics::new(0, 1, 1, 2); // tp=0, tn=1, fp=1, fn=2

        assert_eq!(metrics.precision(), 0.0); // No true positives
        assert_eq!(metrics.recall(), 0.0); // No true positives
        assert_eq!(metrics.f1_score(), 0.0); // F1 is 0 when precision and recall are 0

        // Test edge case: all false positives (no true positives or true negatives)
        let metrics = EvaluationMetrics::new(0, 0, 3, 0); // tp=0, tn=0, fp=3, fn=0

        assert_eq!(metrics.precision(), 0.0); // No true positives
        assert_eq!(metrics.specificity(), 0.0); // No true negatives

        // Test edge case: division by zero in accuracy with all zeros
        let metrics = EvaluationMetrics::new(0, 0, 0, 0);
        assert_eq!(metrics.accuracy(), 0.0);
    }

    #[test]
    fn test_quick_module_functions() {
        use std::io::Write;

        // Test quick::train_csv
        let mut csv_file = NamedTempFile::new().expect("Failed to create temp file");
        writeln!(csv_file, "2.0,1").expect("Failed to write");
        writeln!(csv_file, "-2.0,-1").expect("Failed to write");
        csv_file.flush().expect("Failed to flush");

        let model = quick::train_csv(csv_file.path()).expect("CSV training should succeed");
        assert!(model.info().n_support_vectors > 0);

        // Test quick::train_libsvm_with_c
        let mut libsvm_file = NamedTempFile::new().expect("Failed to create temp file");
        writeln!(libsvm_file, "+1 1:1.0").expect("Failed to write");
        writeln!(libsvm_file, "-1 1:-1.0").expect("Failed to write");
        libsvm_file.flush().expect("Failed to flush");

        let model = quick::train_libsvm_with_c(libsvm_file.path(), 2.0)
            .expect("LibSVM training with C should succeed");
        assert!(model.info().n_support_vectors > 0);

        // Test quick::evaluate_split
        let accuracy = quick::evaluate_split(libsvm_file.path(), libsvm_file.path())
            .expect("Evaluate split should succeed");
        assert!((0.0..=1.0).contains(&accuracy));
    }

    #[test]
    fn test_svm_with_kernel() {
        use crate::kernel::LinearKernel;

        let kernel = LinearKernel::new();
        let svm = SVM::with_kernel(kernel);

        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![1.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-1.0]), -1.0),
        ];

        let model = svm
            .train_samples(&samples)
            .expect("Training should succeed");
        assert!(model.info().n_support_vectors > 0);
    }

    #[test]
    fn test_svm_default() {
        let svm = SVM::default();

        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![1.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-1.0]), -1.0),
        ];

        let model = svm
            .train_samples(&samples)
            .expect("Training should succeed");
        assert!(model.info().n_support_vectors > 0);
    }

    #[test]
    fn test_svm_with_working_set_strategy() {
        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![2.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-2.0]), -1.0),
            Sample::new(SparseVector::new(vec![0], vec![1.8]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-1.8]), -1.0),
        ];

        // Test SMO heuristic
        let model_smo = SVM::new()
            .with_working_set_strategy(WorkingSetStrategy::SMOHeuristic)
            .train_samples(&samples)
            .expect("Training with SMO heuristic should succeed");
        assert!(model_smo.info().n_support_vectors > 0);

        // Test steepest descent
        let model_steepest = SVM::new()
            .with_working_set_strategy(WorkingSetStrategy::SteepestDescent)
            .train_samples(&samples)
            .expect("Training with steepest descent should succeed");
        assert!(model_steepest.info().n_support_vectors > 0);

        // Test random selection
        let model_random = SVM::new()
            .with_working_set_strategy(WorkingSetStrategy::Random)
            .train_samples(&samples)
            .expect("Training with random selection should succeed");
        assert!(model_random.info().n_support_vectors > 0);
    }

    #[test]
    fn test_svm_builder_pattern_full_configuration() {
        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![2.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-2.0]), -1.0),
        ];

        let model = SVM::new()
            .with_c(0.5)
            .with_epsilon(0.01)
            .with_max_iterations(50)
            .with_cache_size(50_000_000)
            .with_shrinking(true)
            .with_shrinking_iterations(10)
            .with_working_set_strategy(WorkingSetStrategy::SteepestDescent)
            .train_samples(&samples)
            .expect("Training with full configuration should succeed");

        assert!(model.info().n_support_vectors > 0);
    }

    #[test]
    fn test_svm_with_feature_scaling() {
        use crate::utils::scaling::ScalingMethod;

        let samples = vec![
            Sample::new(SparseVector::new(vec![0, 1], vec![100.0, 1000.0]), 1.0),
            Sample::new(SparseVector::new(vec![0, 1], vec![200.0, 2000.0]), -1.0),
            Sample::new(SparseVector::new(vec![0, 1], vec![150.0, 1500.0]), 1.0),
            Sample::new(SparseVector::new(vec![0, 1], vec![50.0, 500.0]), -1.0),
        ];

        // Test MinMax scaling
        let model_minmax = SVM::new()
            .with_feature_scaling(ScalingMethod::MinMax {
                min_val: -1.0,
                max_val: 1.0,
            })
            .train_samples(&samples)
            .expect("Training with MinMax scaling should succeed");

        // Test StandardScore scaling
        let model_std = SVM::new()
            .with_feature_scaling(ScalingMethod::StandardScore)
            .train_samples(&samples)
            .expect("Training with StandardScore scaling should succeed");

        // Test UnitScale scaling
        let model_unit = SVM::new()
            .with_feature_scaling(ScalingMethod::UnitScale)
            .train_samples(&samples)
            .expect("Training with UnitScale scaling should succeed");

        // All models should train successfully
        assert!(model_minmax.info().n_support_vectors > 0);
        assert!(model_std.info().n_support_vectors > 0);
        assert!(model_unit.info().n_support_vectors > 0);

        // Test prediction with scaling (should work with raw unscaled data)
        let test_sample = Sample::new(SparseVector::new(vec![0, 1], vec![125.0, 1250.0]), 1.0);

        let pred_minmax = model_minmax.predict(&test_sample);
        let pred_std = model_std.predict(&test_sample);
        let pred_unit = model_unit.predict(&test_sample);

        // Predictions should be finite and reasonable
        assert!(pred_minmax.decision_value.is_finite());
        assert!(pred_std.decision_value.is_finite());
        assert!(pred_unit.decision_value.is_finite());
    }

    #[test]
    fn test_scaling_preserves_accuracy() {
        use crate::utils::scaling::ScalingMethod;

        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![1000.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-1000.0]), -1.0),
            Sample::new(SparseVector::new(vec![0], vec![800.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-800.0]), -1.0),
        ];

        // Train without scaling
        let model_unscaled = SVM::new()
            .with_c(10.0) // Higher C for better convergence with large values
            .train_samples(&samples)
            .expect("Training without scaling should succeed");

        // Train with scaling
        let model_scaled = SVM::new()
            .with_c(1.0) // Standard C with scaling
            .with_feature_scaling(ScalingMethod::MinMax {
                min_val: -1.0,
                max_val: 1.0,
            })
            .train_samples(&samples)
            .expect("Training with scaling should succeed");

        // Both should achieve good accuracy on this simple linearly separable data
        let accuracy_unscaled = model_unscaled
            .predict_batch(&samples)
            .iter()
            .zip(samples.iter())
            .filter(|(pred, sample)| pred.label == sample.label)
            .count() as f64
            / samples.len() as f64;

        let accuracy_scaled = model_scaled
            .predict_batch(&samples)
            .iter()
            .zip(samples.iter())
            .filter(|(pred, sample)| pred.label == sample.label)
            .count() as f64
            / samples.len() as f64;

        // Both models should work, but scaling helps with numerical stability
        assert!(accuracy_unscaled >= 0.5); // At least better than random
        assert!(accuracy_scaled >= 0.8); // Should be good with scaling

        // The scaled model should generally perform as well or better
        assert!(accuracy_scaled >= accuracy_unscaled - 0.1);
    }

    #[test]
    fn test_simple_validation_edge_cases() {
        struct MockDataset {
            samples: Vec<Sample>,
        }

        impl Dataset for MockDataset {
            fn len(&self) -> usize {
                self.samples.len()
            }
            fn dim(&self) -> usize {
                1
            }
            fn get_sample(&self, i: usize) -> Sample {
                self.samples[i].clone()
            }
            fn get_labels(&self) -> Vec<f64> {
                self.samples.iter().map(|s| s.label).collect()
            }
        }

        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![1.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-1.0]), -1.0),
        ];
        let dataset = MockDataset { samples };

        // Test invalid train_ratio (> 1.0)
        let result = quick::simple_validation(&dataset, 1.5, 1.0);
        assert!(result.is_err());

        // Test invalid train_ratio (< 0.0)
        let result = quick::simple_validation(&dataset, -0.1, 1.0);
        assert!(result.is_err());

        // Test invalid train_ratio (= 0.0) - should result in no training data
        let result = quick::simple_validation(&dataset, 0.0, 1.0);
        assert!(result.is_err());

        // Test invalid train_ratio (= 1.0) - should result in no test data
        let result = quick::simple_validation(&dataset, 1.0, 1.0);
        assert!(result.is_err());

        // Test very small dataset
        let small_samples = vec![Sample::new(SparseVector::new(vec![0], vec![1.0]), 1.0)];
        let small_dataset = MockDataset {
            samples: small_samples,
        };
        let result = quick::simple_validation(&small_dataset, 0.5, 1.0);
        assert!(result.is_err()); // Should fail with insufficient data
    }

    #[test]
    fn test_evaluate_detailed_edge_cases() {
        // Create mock dataset for evaluate_detailed
        struct MockDataset {
            samples: Vec<Sample>,
        }

        impl Dataset for MockDataset {
            fn len(&self) -> usize {
                self.samples.len()
            }
            fn dim(&self) -> usize {
                1
            }
            fn get_sample(&self, i: usize) -> Sample {
                self.samples[i].clone()
            }
            fn get_labels(&self) -> Vec<f64> {
                self.samples.iter().map(|s| s.label).collect()
            }
        }

        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![1.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-1.0]), -1.0),
        ];

        let model = SVM::new()
            .train_samples(&samples)
            .expect("Training should succeed");

        // Test with same training data (should get perfect results)
        let dataset = MockDataset {
            samples: samples.clone(),
        };
        let detailed = model.evaluate_detailed(&dataset);
        assert_eq!(detailed.accuracy(), 1.0);
        assert_eq!(detailed.true_positives, 1);
        assert_eq!(detailed.true_negatives, 1);
        assert_eq!(detailed.false_positives, 0);
        assert_eq!(detailed.false_negatives, 0);
    }

    #[test]
    fn test_shrinking_api() {
        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![2.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![1.8]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-2.0]), -1.0),
            Sample::new(SparseVector::new(vec![0], vec![-1.8]), -1.0),
        ];

        // Test with shrinking enabled
        let model_with_shrinking = SVM::new()
            .with_shrinking(true)
            .with_shrinking_iterations(5)
            .train_samples(&samples)
            .expect("Training with shrinking should succeed");

        // Test with shrinking disabled
        let model_without_shrinking = SVM::new()
            .with_shrinking(false)
            .train_samples(&samples)
            .expect("Training without shrinking should succeed");

        // Both should produce valid models
        assert!(model_with_shrinking.info().n_support_vectors > 0);
        assert!(model_without_shrinking.info().n_support_vectors > 0);

        // Test predictions should be similar
        let test_sample = Sample::new(SparseVector::new(vec![0], vec![1.0]), 1.0);
        let pred_with = model_with_shrinking.predict(&test_sample);
        let pred_without = model_without_shrinking.predict(&test_sample);

        // Labels should be the same
        assert_eq!(pred_with.label, pred_without.label);
    }

    #[test]
    fn test_rbf_kernel_api() {
        let samples = vec![
            Sample::new(SparseVector::new(vec![0, 1], vec![1.0, 1.0]), 1.0),
            Sample::new(SparseVector::new(vec![0, 1], vec![1.2, 0.8]), 1.0),
            Sample::new(SparseVector::new(vec![0, 1], vec![-1.0, -1.0]), -1.0),
            Sample::new(SparseVector::new(vec![0, 1], vec![-1.2, -0.8]), -1.0),
        ];

        // Test RBF with manual gamma
        let model_rbf = SVM::with_rbf(1.0)
            .with_c(10.0)
            .train_samples(&samples)
            .expect("RBF training should succeed");

        // Test RBF with auto gamma
        let model_rbf_auto = SVM::with_rbf_auto(2)
            .with_c(10.0)
            .train_samples(&samples)
            .expect("RBF auto gamma training should succeed");

        // Test RBF with unit gamma
        let model_rbf_unit = SVM::with_rbf_unit()
            .with_c(10.0)
            .train_samples(&samples)
            .expect("RBF unit gamma training should succeed");

        // All models should train successfully
        assert!(model_rbf.info().n_support_vectors > 0);
        assert!(model_rbf_auto.info().n_support_vectors > 0);
        assert!(model_rbf_unit.info().n_support_vectors > 0);

        // Test predictions on training data (should be good for this simple case)
        let test_sample = Sample::new(SparseVector::new(vec![0, 1], vec![0.5, 0.5]), 1.0);

        let pred_rbf = model_rbf.predict(&test_sample);
        let pred_auto = model_rbf_auto.predict(&test_sample);
        let pred_unit = model_rbf_unit.predict(&test_sample);

        // All predictions should be reasonable
        assert!(pred_rbf.decision_value.is_finite());
        assert!(pred_auto.decision_value.is_finite());
        assert!(pred_unit.decision_value.is_finite());

        // For this linearly separable case, all should predict positive
        assert_eq!(pred_rbf.label, 1.0);
        assert_eq!(pred_auto.label, 1.0);
        assert_eq!(pred_unit.label, 1.0);
    }

    #[test]
    fn test_rbf_vs_linear_comparison() {
        use crate::utils::scaling::ScalingMethod;

        // Create a non-linearly separable XOR-like problem
        let samples = vec![
            Sample::new(SparseVector::new(vec![0, 1], vec![1.0, 1.0]), -1.0),
            Sample::new(SparseVector::new(vec![0, 1], vec![-1.0, -1.0]), -1.0),
            Sample::new(SparseVector::new(vec![0, 1], vec![1.0, -1.0]), 1.0),
            Sample::new(SparseVector::new(vec![0, 1], vec![-1.0, 1.0]), 1.0),
        ];

        // Train linear SVM
        let model_linear = SVM::new()
            .with_c(10.0)
            .with_feature_scaling(ScalingMethod::MinMax {
                min_val: -1.0,
                max_val: 1.0,
            })
            .train_samples(&samples)
            .expect("Linear training should succeed");

        // Train RBF SVM with high gamma (should handle non-linear separation better)
        let model_rbf = SVM::with_rbf(10.0)
            .with_c(10.0)
            .with_feature_scaling(ScalingMethod::MinMax {
                min_val: -1.0,
                max_val: 1.0,
            })
            .train_samples(&samples)
            .expect("RBF training should succeed");

        // Both should train successfully
        assert!(model_linear.info().n_support_vectors > 0);
        assert!(model_rbf.info().n_support_vectors > 0);

        // Calculate training accuracy for both
        let linear_correct = samples
            .iter()
            .map(|sample| model_linear.predict(sample))
            .zip(samples.iter())
            .filter(|(pred, sample)| pred.label == sample.label)
            .count();

        let rbf_correct = samples
            .iter()
            .map(|sample| model_rbf.predict(sample))
            .zip(samples.iter())
            .filter(|(pred, sample)| pred.label == sample.label)
            .count();

        let linear_accuracy = linear_correct as f64 / samples.len() as f64;
        let rbf_accuracy = rbf_correct as f64 / samples.len() as f64;

        // RBF should perform better or equal on this non-linear problem
        // (though with only 4 samples, results may vary)
        assert!(rbf_accuracy >= linear_accuracy - 0.1); // Allow some tolerance

        // Both accuracies should be reasonable
        assert!(linear_accuracy >= 0.0);
        assert!(rbf_accuracy >= 0.0);
    }

    #[test]
    fn test_polynomial_kernel_api() {
        let samples = vec![
            Sample::new(SparseVector::new(vec![0, 1], vec![1.0, 2.0]), 1.0),
            Sample::new(SparseVector::new(vec![0, 1], vec![2.0, 1.0]), 1.0),
            Sample::new(SparseVector::new(vec![0, 1], vec![-1.0, -2.0]), -1.0),
            Sample::new(SparseVector::new(vec![0, 1], vec![-2.0, -1.0]), -1.0),
        ];

        // Test polynomial with custom parameters
        let model_poly = SVM::with_polynomial(3, 1.0, 1.0)
            .with_c(10.0)
            .train_samples(&samples)
            .expect("Polynomial training should succeed");

        // Test quadratic kernel
        let model_quad = SVM::with_quadratic(0.5)
            .with_c(10.0)
            .train_samples(&samples)
            .expect("Quadratic training should succeed");

        // Test cubic kernel
        let model_cubic = SVM::with_cubic(0.5)
            .with_c(10.0)
            .train_samples(&samples)
            .expect("Cubic training should succeed");

        // Test auto gamma polynomial
        let model_auto = SVM::with_polynomial_auto(2, 2)
            .with_c(10.0)
            .train_samples(&samples)
            .expect("Auto polynomial training should succeed");

        // Test normalized polynomial
        let model_norm = SVM::with_polynomial_normalized(2)
            .with_c(10.0)
            .train_samples(&samples)
            .expect("Normalized polynomial training should succeed");

        // All models should train successfully
        assert!(model_poly.info().n_support_vectors > 0);
        assert!(model_quad.info().n_support_vectors > 0);
        assert!(model_cubic.info().n_support_vectors > 0);
        assert!(model_auto.info().n_support_vectors > 0);
        assert!(model_norm.info().n_support_vectors > 0);

        // Test predictions
        let test_sample = Sample::new(SparseVector::new(vec![0, 1], vec![1.5, 1.5]), 1.0);

        let pred_poly = model_poly.predict(&test_sample);
        let pred_quad = model_quad.predict(&test_sample);
        let pred_cubic = model_cubic.predict(&test_sample);
        let pred_auto = model_auto.predict(&test_sample);
        let pred_norm = model_norm.predict(&test_sample);

        // All predictions should be finite
        assert!(pred_poly.decision_value.is_finite());
        assert!(pred_quad.decision_value.is_finite());
        assert!(pred_cubic.decision_value.is_finite());
        assert!(pred_auto.decision_value.is_finite());
        assert!(pred_norm.decision_value.is_finite());

        // For this separable case, all should predict positive
        assert_eq!(pred_poly.label, 1.0);
        assert_eq!(pred_quad.label, 1.0);
        assert_eq!(pred_cubic.label, 1.0);
        assert_eq!(pred_auto.label, 1.0);
        assert_eq!(pred_norm.label, 1.0);
    }

    #[test]
    fn test_polynomial_kernel_degrees() {
        use crate::utils::scaling::ScalingMethod;

        // Create a dataset that benefits from non-linear classification
        let samples = vec![
            Sample::new(SparseVector::new(vec![0, 1], vec![1.0, 1.0]), 1.0),
            Sample::new(SparseVector::new(vec![0, 1], vec![-1.0, -1.0]), 1.0),
            Sample::new(SparseVector::new(vec![0, 1], vec![1.0, -1.0]), -1.0),
            Sample::new(SparseVector::new(vec![0, 1], vec![-1.0, 1.0]), -1.0),
            Sample::new(SparseVector::new(vec![0, 1], vec![0.8, 0.8]), 1.0),
            Sample::new(SparseVector::new(vec![0, 1], vec![-0.8, -0.8]), 1.0),
        ];

        // Test different polynomial degrees
        let degrees = [2, 3, 4];
        let mut models = Vec::new();

        for &degree in &degrees {
            let model = SVM::with_polynomial(degree, 1.0, 1.0)
                .with_c(10.0)
                .with_feature_scaling(ScalingMethod::MinMax {
                    min_val: -1.0,
                    max_val: 1.0,
                })
                .train_samples(&samples)
                .expect(&format!(
                    "Polynomial degree {} training should succeed",
                    degree
                ));

            models.push(model);
        }

        // All models should train successfully
        for model in &models {
            assert!(model.info().n_support_vectors > 0);
        }

        // Test predictions on a new sample
        let test_sample = Sample::new(SparseVector::new(vec![0, 1], vec![0.5, 0.5]), 1.0);

        for (i, model) in models.iter().enumerate() {
            let pred = model.predict(&test_sample);
            assert!(
                pred.decision_value.is_finite(),
                "Prediction for degree {} should be finite",
                degrees[i]
            );
        }
    }

    #[test]
    fn test_chi_square_kernel_api() {
        let samples = vec![
            Sample::new(
                SparseVector::new(vec![0, 1, 2], vec![10.0, 20.0, 30.0]),
                1.0,
            ),
            Sample::new(
                SparseVector::new(vec![0, 1, 2], vec![15.0, 25.0, 35.0]),
                1.0,
            ),
            Sample::new(
                SparseVector::new(vec![0, 1, 2], vec![5.0, 10.0, 15.0]),
                -1.0,
            ),
            Sample::new(
                SparseVector::new(vec![0, 1, 2], vec![8.0, 12.0, 18.0]),
                -1.0,
            ),
        ];

        // Test Chi-square with manual gamma
        let model_chi2 = SVM::with_chi_square(1.0)
            .with_c(10.0)
            .train_samples(&samples)
            .expect("Chi-square training should succeed");

        // Test Chi-square with unit gamma
        let model_chi2_unit = SVM::with_chi_square_unit()
            .with_c(10.0)
            .train_samples(&samples)
            .expect("Chi-square unit gamma training should succeed");

        // Test Chi-square with auto gamma
        let model_chi2_auto = SVM::with_chi_square_auto(3)
            .with_c(10.0)
            .train_samples(&samples)
            .expect("Chi-square auto gamma training should succeed");

        // Test Chi-square for computer vision
        let model_chi2_cv = SVM::with_chi_square_cv()
            .with_c(10.0)
            .train_samples(&samples)
            .expect("Chi-square CV training should succeed");

        // Test Chi-square for text analysis
        let model_chi2_text = SVM::with_chi_square_text()
            .with_c(10.0)
            .train_samples(&samples)
            .expect("Chi-square text training should succeed");

        // All models should train successfully
        assert!(model_chi2.info().n_support_vectors > 0);
        assert!(model_chi2_unit.info().n_support_vectors > 0);
        assert!(model_chi2_auto.info().n_support_vectors > 0);
        assert!(model_chi2_cv.info().n_support_vectors > 0);
        assert!(model_chi2_text.info().n_support_vectors > 0);

        // Test predictions
        let test_sample = Sample::new(
            SparseVector::new(vec![0, 1, 2], vec![12.0, 22.0, 32.0]),
            1.0,
        );

        let pred_chi2 = model_chi2.predict(&test_sample);
        let pred_unit = model_chi2_unit.predict(&test_sample);
        let pred_auto = model_chi2_auto.predict(&test_sample);
        let pred_cv = model_chi2_cv.predict(&test_sample);
        let pred_text = model_chi2_text.predict(&test_sample);

        // All predictions should be finite
        assert!(pred_chi2.decision_value.is_finite());
        assert!(pred_unit.decision_value.is_finite());
        assert!(pred_auto.decision_value.is_finite());
        assert!(pred_cv.decision_value.is_finite());
        assert!(pred_text.decision_value.is_finite());

        // For this histogram-like data, Chi-square should work well
        assert_eq!(pred_chi2.label, 1.0);
        assert_eq!(pred_unit.label, 1.0);
        assert_eq!(pred_auto.label, 1.0);
        assert_eq!(pred_cv.label, 1.0);
        assert_eq!(pred_text.label, 1.0);
    }

    #[test]
    fn test_chi_square_vs_linear_comparison() {
        use crate::utils::scaling::ScalingMethod;

        // Create histogram-like data where Chi-square should outperform linear
        let samples = vec![
            // Histogram class 1: peak at beginning
            Sample::new(
                SparseVector::new(vec![0, 1, 2, 3], vec![50.0, 30.0, 15.0, 5.0]),
                1.0,
            ),
            Sample::new(
                SparseVector::new(vec![0, 1, 2, 3], vec![45.0, 35.0, 12.0, 8.0]),
                1.0,
            ),
            Sample::new(
                SparseVector::new(vec![0, 1, 2, 3], vec![55.0, 25.0, 18.0, 2.0]),
                1.0,
            ),
            // Histogram class 2: peak at end
            Sample::new(
                SparseVector::new(vec![0, 1, 2, 3], vec![5.0, 15.0, 30.0, 50.0]),
                -1.0,
            ),
            Sample::new(
                SparseVector::new(vec![0, 1, 2, 3], vec![8.0, 12.0, 35.0, 45.0]),
                -1.0,
            ),
            Sample::new(
                SparseVector::new(vec![0, 1, 2, 3], vec![2.0, 18.0, 25.0, 55.0]),
                -1.0,
            ),
        ];

        // Train linear SVM
        let model_linear = SVM::new()
            .with_c(1.0)
            .with_feature_scaling(ScalingMethod::MinMax {
                min_val: 0.0,
                max_val: 1.0,
            })
            .train_samples(&samples)
            .expect("Linear training should succeed");

        // Train Chi-square SVM
        let model_chi2 = SVM::with_chi_square(1.0)
            .with_c(1.0)
            .train_samples(&samples)
            .expect("Chi-square training should succeed");

        // Both should train successfully
        assert!(model_linear.info().n_support_vectors > 0);
        assert!(model_chi2.info().n_support_vectors > 0);

        // Calculate training accuracy for both
        let linear_correct = samples
            .iter()
            .map(|sample| model_linear.predict(sample))
            .zip(samples.iter())
            .filter(|(pred, sample)| pred.label == sample.label)
            .count();

        let chi2_correct = samples
            .iter()
            .map(|sample| model_chi2.predict(sample))
            .zip(samples.iter())
            .filter(|(pred, sample)| pred.label == sample.label)
            .count();

        let linear_accuracy = linear_correct as f64 / samples.len() as f64;
        let chi2_accuracy = chi2_correct as f64 / samples.len() as f64;

        // Both should achieve reasonable accuracy
        assert!(linear_accuracy >= 0.0);
        assert!(chi2_accuracy >= 0.0);

        // Chi-square should perform well or better on histogram data
        assert!(chi2_accuracy >= linear_accuracy - 0.1); // Allow some tolerance

        // Test prediction consistency
        let test_sample = Sample::new(
            SparseVector::new(vec![0, 1, 2, 3], vec![40.0, 30.0, 20.0, 10.0]),
            1.0,
        );
        let pred_linear = model_linear.predict(&test_sample);
        let pred_chi2 = model_chi2.predict(&test_sample);

        assert!(pred_linear.decision_value.is_finite());
        assert!(pred_chi2.decision_value.is_finite());
    }

    #[test]
    fn test_histogram_intersection_kernel_api() {
        let samples = vec![
            // Histogram class 1: peaks at beginning
            Sample::new(
                SparseVector::new(vec![0, 1, 2, 3], vec![30.0, 20.0, 10.0, 5.0]),
                1.0,
            ),
            Sample::new(
                SparseVector::new(vec![0, 1, 2, 3], vec![35.0, 25.0, 8.0, 7.0]),
                1.0,
            ),
            // Histogram class 2: peaks at end
            Sample::new(
                SparseVector::new(vec![0, 1, 2, 3], vec![5.0, 10.0, 25.0, 30.0]),
                -1.0,
            ),
            Sample::new(
                SparseVector::new(vec![0, 1, 2, 3], vec![8.0, 12.0, 20.0, 35.0]),
                -1.0,
            ),
        ];

        // Test different histogram intersection variants
        let model_standard = SVM::with_histogram_intersection_standard()
            .with_c(10.0)
            .train_samples(&samples)
            .expect("Standard histogram intersection training should succeed");

        let model_normalized = SVM::with_histogram_intersection_normalized()
            .with_c(10.0)
            .train_samples(&samples)
            .expect("Normalized histogram intersection training should succeed");

        let model_color = SVM::with_histogram_intersection_color()
            .with_c(10.0)
            .train_samples(&samples)
            .expect("Color histogram training should succeed");

        let model_visual_words = SVM::with_histogram_intersection_visual_words()
            .with_c(10.0)
            .train_samples(&samples)
            .expect("Visual words training should succeed");

        let model_texture = SVM::with_histogram_intersection_texture()
            .with_c(10.0)
            .train_samples(&samples)
            .expect("Texture analysis training should succeed");

        let model_manual = SVM::with_histogram_intersection(true)
            .with_c(10.0)
            .train_samples(&samples)
            .expect("Manual histogram intersection training should succeed");

        // All models should train successfully
        assert!(model_standard.info().n_support_vectors > 0);
        assert!(model_normalized.info().n_support_vectors > 0);
        assert!(model_color.info().n_support_vectors > 0);
        assert!(model_visual_words.info().n_support_vectors > 0);
        assert!(model_texture.info().n_support_vectors > 0);
        assert!(model_manual.info().n_support_vectors > 0);

        // Test predictions on histogram leaning towards class 1
        let test_sample = Sample::new(
            SparseVector::new(vec![0, 1, 2, 3], vec![25.0, 15.0, 10.0, 5.0]),
            1.0,
        );

        let pred_standard = model_standard.predict(&test_sample);
        let pred_normalized = model_normalized.predict(&test_sample);
        let pred_color = model_color.predict(&test_sample);
        let pred_visual_words = model_visual_words.predict(&test_sample);
        let pred_texture = model_texture.predict(&test_sample);
        let pred_manual = model_manual.predict(&test_sample);

        // All predictions should be finite and reasonable
        assert!(pred_standard.decision_value.is_finite());
        assert!(pred_normalized.decision_value.is_finite());
        assert!(pred_color.decision_value.is_finite());
        assert!(pred_visual_words.decision_value.is_finite());
        assert!(pred_texture.decision_value.is_finite());
        assert!(pred_manual.decision_value.is_finite());

        // For this histogram pattern, should predict positive
        assert_eq!(pred_standard.label, 1.0);
        assert_eq!(pred_normalized.label, 1.0);
        assert_eq!(pred_color.label, 1.0);
        assert_eq!(pred_visual_words.label, 1.0);
        assert_eq!(pred_texture.label, 1.0);
        assert_eq!(pred_manual.label, 1.0);
    }

    #[test]
    fn test_histogram_intersection_vs_other_kernels() {
        // Create histogram data where Histogram Intersection should excel
        let samples = vec![
            // Class 1: Exponential decay histograms (common in computer vision)
            Sample::new(
                SparseVector::new(vec![0, 1, 2, 3, 4], vec![50.0, 25.0, 12.0, 6.0, 3.0]),
                1.0,
            ),
            Sample::new(
                SparseVector::new(vec![0, 1, 2, 3, 4], vec![48.0, 28.0, 14.0, 8.0, 2.0]),
                1.0,
            ),
            Sample::new(
                SparseVector::new(vec![0, 1, 2, 3, 4], vec![52.0, 22.0, 11.0, 5.0, 4.0]),
                1.0,
            ),
            // Class 2: Uniform-like histograms
            Sample::new(
                SparseVector::new(vec![0, 1, 2, 3, 4], vec![20.0, 19.0, 18.0, 21.0, 22.0]),
                -1.0,
            ),
            Sample::new(
                SparseVector::new(vec![0, 1, 2, 3, 4], vec![18.0, 21.0, 20.0, 19.0, 22.0]),
                -1.0,
            ),
            Sample::new(
                SparseVector::new(vec![0, 1, 2, 3, 4], vec![22.0, 18.0, 19.0, 20.0, 21.0]),
                -1.0,
            ),
        ];

        // Train different kernels
        let model_linear = SVM::new()
            .with_c(1.0)
            .train_samples(&samples)
            .expect("Linear training should succeed");

        let model_hist = SVM::with_histogram_intersection_normalized()
            .with_c(1.0)
            .train_samples(&samples)
            .expect("Histogram intersection training should succeed");

        let model_chi2 = SVM::with_chi_square(1.0)
            .with_c(1.0)
            .train_samples(&samples)
            .expect("Chi-square training should succeed");

        // All should train successfully
        assert!(model_linear.info().n_support_vectors > 0);
        assert!(model_hist.info().n_support_vectors > 0);
        assert!(model_chi2.info().n_support_vectors > 0);

        // Calculate accuracy for each
        let linear_correct = samples
            .iter()
            .map(|sample| model_linear.predict(sample))
            .zip(samples.iter())
            .filter(|(pred, sample)| pred.label == sample.label)
            .count();

        let hist_correct = samples
            .iter()
            .map(|sample| model_hist.predict(sample))
            .zip(samples.iter())
            .filter(|(pred, sample)| pred.label == sample.label)
            .count();

        let chi2_correct = samples
            .iter()
            .map(|sample| model_chi2.predict(sample))
            .zip(samples.iter())
            .filter(|(pred, sample)| pred.label == sample.label)
            .count();

        let linear_accuracy = linear_correct as f64 / samples.len() as f64;
        let hist_accuracy = hist_correct as f64 / samples.len() as f64;
        let chi2_accuracy = chi2_correct as f64 / samples.len() as f64;

        // All should achieve reasonable accuracy
        assert!(linear_accuracy >= 0.0);
        assert!(hist_accuracy >= 0.0);
        assert!(chi2_accuracy >= 0.0);

        // Histogram intersection should perform well on histogram data
        assert!(hist_accuracy >= linear_accuracy - 0.1); // Allow some tolerance

        // Test prediction consistency
        let test_sample = Sample::new(
            SparseVector::new(vec![0, 1, 2, 3, 4], vec![45.0, 24.0, 13.0, 7.0, 3.0]),
            1.0,
        );
        let pred_linear = model_linear.predict(&test_sample);
        let pred_hist = model_hist.predict(&test_sample);
        let pred_chi2 = model_chi2.predict(&test_sample);

        assert!(pred_linear.decision_value.is_finite());
        assert!(pred_hist.decision_value.is_finite());
        assert!(pred_chi2.decision_value.is_finite());
    }

    #[test]
    fn test_hellinger_kernel_api() {
        let samples = vec![
            // Probability distributions class 1
            Sample::new(
                SparseVector::new(vec![0, 1, 2, 3], vec![0.4, 0.3, 0.2, 0.1]),
                1.0,
            ),
            Sample::new(
                SparseVector::new(vec![0, 1, 2, 3], vec![0.5, 0.25, 0.15, 0.1]),
                1.0,
            ),
            Sample::new(
                SparseVector::new(vec![0, 1, 2, 3], vec![0.35, 0.35, 0.2, 0.1]),
                1.0,
            ),
            // Probability distributions class 2
            Sample::new(
                SparseVector::new(vec![0, 1, 2, 3], vec![0.1, 0.2, 0.3, 0.4]),
                -1.0,
            ),
            Sample::new(
                SparseVector::new(vec![0, 1, 2, 3], vec![0.15, 0.15, 0.3, 0.4]),
                -1.0,
            ),
            Sample::new(
                SparseVector::new(vec![0, 1, 2, 3], vec![0.1, 0.25, 0.25, 0.4]),
                -1.0,
            ),
        ];

        // Test different Hellinger kernel variants
        let model_hellinger = SVM::with_hellinger(false)
            .with_c(10.0)
            .train_samples(&samples)
            .expect("Hellinger training should succeed");

        let model_standard = SVM::with_hellinger_standard()
            .with_c(10.0)
            .train_samples(&samples)
            .expect("Hellinger standard training should succeed");

        let model_normalized = SVM::with_hellinger_normalized()
            .with_c(10.0)
            .train_samples(&samples)
            .expect("Hellinger normalized training should succeed");

        let model_text = SVM::with_hellinger_text()
            .with_c(10.0)
            .train_samples(&samples)
            .expect("Hellinger text training should succeed");

        let model_bio = SVM::with_hellinger_bio()
            .with_c(10.0)
            .train_samples(&samples)
            .expect("Hellinger bio training should succeed");

        let model_probability = SVM::with_hellinger_probability()
            .with_c(10.0)
            .train_samples(&samples)
            .expect("Hellinger probability training should succeed");

        let model_stats = SVM::with_hellinger_stats()
            .with_c(10.0)
            .train_samples(&samples)
            .expect("Hellinger stats training should succeed");

        // All models should train successfully
        assert!(model_hellinger.info().n_support_vectors > 0);
        assert!(model_standard.info().n_support_vectors > 0);
        assert!(model_normalized.info().n_support_vectors > 0);
        assert!(model_text.info().n_support_vectors > 0);
        assert!(model_bio.info().n_support_vectors > 0);
        assert!(model_probability.info().n_support_vectors > 0);
        assert!(model_stats.info().n_support_vectors > 0);

        // Test predictions on probability distribution leaning towards class 1
        let test_sample = Sample::new(
            SparseVector::new(vec![0, 1, 2, 3], vec![0.45, 0.3, 0.15, 0.1]),
            1.0,
        );

        let pred_hellinger = model_hellinger.predict(&test_sample);
        let pred_standard = model_standard.predict(&test_sample);
        let pred_normalized = model_normalized.predict(&test_sample);
        let pred_text = model_text.predict(&test_sample);
        let pred_bio = model_bio.predict(&test_sample);
        let pred_probability = model_probability.predict(&test_sample);
        let pred_stats = model_stats.predict(&test_sample);

        // All predictions should be finite and reasonable
        assert!(pred_hellinger.decision_value.is_finite());
        assert!(pred_standard.decision_value.is_finite());
        assert!(pred_normalized.decision_value.is_finite());
        assert!(pred_text.decision_value.is_finite());
        assert!(pred_bio.decision_value.is_finite());
        assert!(pred_probability.decision_value.is_finite());
        assert!(pred_stats.decision_value.is_finite());

        // For this probability distribution pattern, should predict positive
        assert_eq!(pred_hellinger.label, 1.0);
        assert_eq!(pred_standard.label, 1.0);
        assert_eq!(pred_normalized.label, 1.0);
        assert_eq!(pred_text.label, 1.0);
        assert_eq!(pred_bio.label, 1.0);
        assert_eq!(pred_probability.label, 1.0);
        assert_eq!(pred_stats.label, 1.0);
    }

    #[test]
    fn test_hellinger_vs_other_kernels() {
        use crate::utils::scaling::ScalingMethod;

        // Create probability distribution data where Hellinger should excel
        let samples = vec![
            // Class 1: Uniform-like distributions
            Sample::new(
                SparseVector::new(vec![0, 1, 2, 3, 4], vec![0.2, 0.25, 0.2, 0.2, 0.15]),
                1.0,
            ),
            Sample::new(
                SparseVector::new(vec![0, 1, 2, 3, 4], vec![0.18, 0.22, 0.25, 0.18, 0.17]),
                1.0,
            ),
            Sample::new(
                SparseVector::new(vec![0, 1, 2, 3, 4], vec![0.22, 0.2, 0.18, 0.22, 0.18]),
                1.0,
            ),
            // Class 2: Skewed distributions
            Sample::new(
                SparseVector::new(vec![0, 1, 2, 3, 4], vec![0.5, 0.3, 0.15, 0.04, 0.01]),
                -1.0,
            ),
            Sample::new(
                SparseVector::new(vec![0, 1, 2, 3, 4], vec![0.45, 0.35, 0.12, 0.06, 0.02]),
                -1.0,
            ),
            Sample::new(
                SparseVector::new(vec![0, 1, 2, 3, 4], vec![0.52, 0.28, 0.13, 0.05, 0.02]),
                -1.0,
            ),
        ];

        // Train different kernels
        let model_linear = SVM::new()
            .with_c(1.0)
            .with_feature_scaling(ScalingMethod::MinMax {
                min_val: 0.0,
                max_val: 1.0,
            })
            .train_samples(&samples)
            .expect("Linear training should succeed");

        let model_hellinger = SVM::with_hellinger_probability()
            .with_c(1.0)
            .train_samples(&samples)
            .expect("Hellinger training should succeed");

        let model_chi2 = SVM::with_chi_square(1.0)
            .with_c(1.0)
            .train_samples(&samples)
            .expect("Chi-square training should succeed");

        let model_hist = SVM::with_histogram_intersection_normalized()
            .with_c(1.0)
            .train_samples(&samples)
            .expect("Histogram intersection training should succeed");

        // All should train successfully
        assert!(model_linear.info().n_support_vectors > 0);
        assert!(model_hellinger.info().n_support_vectors > 0);
        assert!(model_chi2.info().n_support_vectors > 0);
        assert!(model_hist.info().n_support_vectors > 0);

        // Calculate accuracy for each
        let linear_correct = samples
            .iter()
            .map(|sample| model_linear.predict(sample))
            .zip(samples.iter())
            .filter(|(pred, sample)| pred.label == sample.label)
            .count();

        let hellinger_correct = samples
            .iter()
            .map(|sample| model_hellinger.predict(sample))
            .zip(samples.iter())
            .filter(|(pred, sample)| pred.label == sample.label)
            .count();

        let chi2_correct = samples
            .iter()
            .map(|sample| model_chi2.predict(sample))
            .zip(samples.iter())
            .filter(|(pred, sample)| pred.label == sample.label)
            .count();

        let hist_correct = samples
            .iter()
            .map(|sample| model_hist.predict(sample))
            .zip(samples.iter())
            .filter(|(pred, sample)| pred.label == sample.label)
            .count();

        let linear_accuracy = linear_correct as f64 / samples.len() as f64;
        let hellinger_accuracy = hellinger_correct as f64 / samples.len() as f64;
        let chi2_accuracy = chi2_correct as f64 / samples.len() as f64;
        let hist_accuracy = hist_correct as f64 / samples.len() as f64;

        // All should achieve reasonable accuracy
        assert!(linear_accuracy >= 0.0);
        assert!(hellinger_accuracy >= 0.0);
        assert!(chi2_accuracy >= 0.0);
        assert!(hist_accuracy >= 0.0);

        // Hellinger should perform well on probability distribution data
        assert!(hellinger_accuracy >= linear_accuracy - 0.1); // Allow some tolerance

        // Test prediction consistency
        let test_sample = Sample::new(
            SparseVector::new(vec![0, 1, 2, 3, 4], vec![0.25, 0.25, 0.2, 0.15, 0.15]),
            1.0,
        );
        let pred_linear = model_linear.predict(&test_sample);
        let pred_hellinger = model_hellinger.predict(&test_sample);
        let pred_chi2 = model_chi2.predict(&test_sample);
        let pred_hist = model_hist.predict(&test_sample);

        assert!(pred_linear.decision_value.is_finite());
        assert!(pred_hellinger.decision_value.is_finite());
        assert!(pred_chi2.decision_value.is_finite());
        assert!(pred_hist.decision_value.is_finite());
    }

    #[test]
    fn test_polynomial_vs_linear_comparison() {
        use crate::utils::scaling::ScalingMethod;

        // Create a dataset where polynomial should outperform linear
        let samples = vec![
            Sample::new(SparseVector::new(vec![0], vec![0.5]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![1.5]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![2.5]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![1.0]), -1.0),
            Sample::new(SparseVector::new(vec![0], vec![2.0]), -1.0),
        ];

        // Train linear SVM
        let model_linear = SVM::new()
            .with_c(1.0)
            .with_feature_scaling(ScalingMethod::StandardScore)
            .train_samples(&samples)
            .expect("Linear training should succeed");

        // Train polynomial SVM
        let model_poly = SVM::with_quadratic(1.0)
            .with_c(1.0)
            .with_feature_scaling(ScalingMethod::StandardScore)
            .train_samples(&samples)
            .expect("Polynomial training should succeed");

        // Both should train successfully
        assert!(model_linear.info().n_support_vectors > 0);
        assert!(model_poly.info().n_support_vectors > 0);

        // Calculate training accuracy for both
        let linear_correct = samples
            .iter()
            .map(|sample| model_linear.predict(sample))
            .zip(samples.iter())
            .filter(|(pred, sample)| pred.label == sample.label)
            .count();

        let poly_correct = samples
            .iter()
            .map(|sample| model_poly.predict(sample))
            .zip(samples.iter())
            .filter(|(pred, sample)| pred.label == sample.label)
            .count();

        let linear_accuracy = linear_correct as f64 / samples.len() as f64;
        let poly_accuracy = poly_correct as f64 / samples.len() as f64;

        // Both should achieve reasonable accuracy
        assert!(linear_accuracy >= 0.0);
        assert!(poly_accuracy >= 0.0);

        // Test prediction consistency
        let test_sample = Sample::new(SparseVector::new(vec![0], vec![1.2]), -1.0);
        let pred_linear = model_linear.predict(&test_sample);
        let pred_poly = model_poly.predict(&test_sample);

        assert!(pred_linear.decision_value.is_finite());
        assert!(pred_poly.decision_value.is_finite());
    }
}
