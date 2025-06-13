//! Histogram Intersection Kernel Demonstration
//!
//! This example demonstrates the use of Histogram Intersection kernels for computer vision
//! and histogram-based classification tasks. The Histogram Intersection kernel is particularly
//! effective for applications where features represent frequency counts or histograms.
//!
//! Applications demonstrated:
//! - Color histogram classification (RGB histograms)
//! - Texture analysis (Local Binary Pattern simulation)
//! - Visual bag-of-words (SIFT descriptor simulation)
//! - Object recognition scenarios

use rsvm::api::SVM;
use rsvm::core::{Sample, SparseVector};
use rsvm::kernel::Kernel;
use rsvm::TrainedModel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Histogram Intersection Kernel Demonstration ===");
    println!();

    // Test 1: Color Histogram Classification
    println!("📸 Test 1: Color Histogram Classification (RGB)");
    test_color_histogram_classification()?;
    println!();

    // Test 2: Texture Analysis with LBP-like histograms
    println!("🎯 Test 2: Texture Analysis (Local Binary Pattern simulation)");
    test_texture_analysis()?;
    println!();

    // Test 3: Visual Bag-of-Words
    println!("👁️  Test 3: Visual Bag-of-Words Classification");
    test_visual_bag_of_words()?;
    println!();

    // Test 4: Normalized vs Standard Comparison
    println!("⚖️  Test 4: Normalized vs Standard Intersection");
    test_normalization_comparison()?;
    println!();

    // Test 5: Kernel Comparison on Histogram Data
    println!("🔬 Test 5: Kernel Comparison for Histogram Data");
    test_kernel_comparison()?;

    Ok(())
}

fn test_color_histogram_classification() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Creating RGB color histogram dataset...");

    let mut samples = Vec::new();

    // Class 1: Red-dominant images (fire, sunset, roses)
    // RGB histograms with high red, medium green, low blue
    for i in 0..30 {
        let noise = 0.1 * ((i as f64 / 30.0) - 0.5);

        // Red channel: [0-63, 64-127, 128-191, 192-255] bins
        let red_bins = vec![15.0 + noise, 45.0 + noise, 30.0 + noise, 10.0];
        // Green channel: similar structure but lower values
        let green_bins = vec![8.0, 25.0 + noise, 20.0, 7.0];
        // Blue channel: low values
        let blue_bins = vec![3.0, 8.0, 12.0 + noise, 5.0];

        let mut histogram = Vec::new();
        histogram.extend(red_bins);
        histogram.extend(green_bins);
        histogram.extend(blue_bins);

        let indices: Vec<usize> = (0..histogram.len()).collect();
        samples.push(Sample::new(SparseVector::new(indices, histogram), 1.0));
    }

    // Class 2: Blue-dominant images (sky, water, flowers)
    // RGB histograms with low red, medium green, high blue
    for i in 0..30 {
        let noise = 0.1 * ((i as f64 / 30.0) - 0.5);

        // Red channel: low values
        let red_bins = vec![3.0, 8.0 + noise, 12.0, 5.0];
        // Green channel: medium values
        let green_bins = vec![10.0, 20.0 + noise, 25.0, 15.0];
        // Blue channel: high values
        let blue_bins = vec![12.0, 35.0 + noise, 40.0, 20.0];

        let mut histogram = Vec::new();
        histogram.extend(red_bins);
        histogram.extend(green_bins);
        histogram.extend(blue_bins);

        let indices: Vec<usize> = (0..histogram.len()).collect();
        samples.push(Sample::new(SparseVector::new(indices, histogram), -1.0));
    }

    println!(
        "  Generated {} RGB color histogram samples (12 bins each)",
        samples.len()
    );

    // Train different models
    let model_linear = SVM::new().with_c(1.0).train_samples(&samples)?;

    let model_hist_color = SVM::with_histogram_intersection_color()
        .with_c(1.0)
        .train_samples(&samples)?;

    let model_chi2 = SVM::with_chi_square_cv()
        .with_c(1.0)
        .train_samples(&samples)?;

    // Evaluate models
    let linear_acc = evaluate_model(&model_linear, &samples);
    let hist_acc = evaluate_model(&model_hist_color, &samples);
    let chi2_acc = evaluate_model(&model_chi2, &samples);

    println!("  Results on RGB histogram classification:");
    println!(
        "    Linear kernel:               {:.1}% accuracy ({} SVs)",
        linear_acc * 100.0,
        model_linear.info().n_support_vectors
    );
    println!(
        "    Histogram Intersection:      {:.1}% accuracy ({} SVs)",
        hist_acc * 100.0,
        model_hist_color.info().n_support_vectors
    );
    println!(
        "    Chi-square (CV):             {:.1}% accuracy ({} SVs)",
        chi2_acc * 100.0,
        model_chi2.info().n_support_vectors
    );
    println!(
        "    Hist. Intersection advantage: {:.1} percentage points over linear",
        (hist_acc - linear_acc) * 100.0
    );

    // Test on ambiguous case (purple-ish: medium red, low green, medium blue)
    let purple_histogram = vec![
        8.0, 20.0, 15.0, 7.0, // Red bins
        4.0, 12.0, 8.0, 6.0, // Green bins
        6.0, 25.0, 20.0, 9.0, // Blue bins
    ];
    let indices: Vec<usize> = (0..purple_histogram.len()).collect();
    let purple_sample = Sample::new(SparseVector::new(indices, purple_histogram), 0.0);

    let hist_pred = model_hist_color.predict(&purple_sample);
    println!(
        "  Purple histogram prediction: {} (confidence: {:.3})",
        if hist_pred.label > 0.0 {
            "Red-dominant"
        } else {
            "Blue-dominant"
        },
        hist_pred.decision_value.abs()
    );

    Ok(())
}

fn test_texture_analysis() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Creating Local Binary Pattern-like texture histograms...");

    let mut samples = Vec::new();

    // Class 1: Smooth textures (few transitions, low LBP values dominant)
    for i in 0..25 {
        let base = 0.05 * (i as f64 / 25.0);

        // LBP histogram: 256 bins compressed to 8 bins for demo
        // Smooth textures have high values in low bins (uniform patterns)
        let lbp_histogram = vec![
            40.0 + base, // Uniform patterns
            35.0 + base, // Simple patterns
            15.0,        // Medium complexity
            8.0,         // Higher complexity
            5.0,         // Complex patterns
            3.0,         // Very complex
            2.0,         // Noise-like
            1.0,         // Random noise
        ];

        let indices: Vec<usize> = (0..lbp_histogram.len()).collect();
        samples.push(Sample::new(SparseVector::new(indices, lbp_histogram), 1.0));
    }

    // Class 2: Rough textures (many transitions, higher LBP values)
    for i in 0..25 {
        let base = 0.05 * (i as f64 / 25.0);

        // Rough textures have more distributed values across bins
        let lbp_histogram = vec![
            8.0,         // Few uniform patterns
            12.0 + base, // Some simple patterns
            18.0 + base, // Medium complexity dominant
            22.0,        // Higher complexity
            20.0 + base, // Complex patterns common
            15.0,        // Very complex
            8.0,         // Noise-like patterns
            5.0,         // Random noise
        ];

        let indices: Vec<usize> = (0..lbp_histogram.len()).collect();
        samples.push(Sample::new(SparseVector::new(indices, lbp_histogram), -1.0));
    }

    println!(
        "  Generated {} LBP texture histograms (8 bins each)",
        samples.len()
    );

    // Train models specifically for texture analysis
    let model_hist_texture = SVM::with_histogram_intersection_texture()
        .with_c(1.0)
        .train_samples(&samples)?;

    let model_linear = SVM::new().with_c(1.0).train_samples(&samples)?;

    // Evaluate models
    let hist_acc = evaluate_model(&model_hist_texture, &samples);
    let linear_acc = evaluate_model(&model_linear, &samples);

    println!("  Results on texture classification:");
    println!(
        "    Histogram Intersection (texture): {:.1}% accuracy ({} SVs)",
        hist_acc * 100.0,
        model_hist_texture.info().n_support_vectors
    );
    println!(
        "    Linear kernel:                    {:.1}% accuracy ({} SVs)",
        linear_acc * 100.0,
        model_linear.info().n_support_vectors
    );
    println!(
        "    Improvement:                      {:.1} percentage points",
        (hist_acc - linear_acc) * 100.0
    );

    Ok(())
}

fn test_visual_bag_of_words() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Creating visual bag-of-words histograms (SIFT-like descriptors)...");

    let mut samples = Vec::new();

    // Class 1: Indoor scenes (furniture, rooms) - certain visual words more common
    for i in 0..20 {
        let noise = 0.1 * ((i as f64 / 20.0) - 0.5);

        // Visual word histogram (1000 words compressed to 10 for demo)
        // Indoor scenes: furniture edges, corners, textures
        let visual_words = vec![
            25.0 + noise, // Vertical edges (walls, furniture)
            20.0 + noise, // Horizontal edges (tables, shelves)
            15.0,         // Corner features
            12.0 + noise, // Texture patterns (carpet, wood)
            8.0,          // Smooth regions
            6.0,          // Circular features (lamps, decoration)
            4.0,          // Diagonal patterns
            3.0,          // Complex textures
            2.0,          // Rare features
            1.0,          // Very rare visual words
        ];

        let indices: Vec<usize> = (0..visual_words.len()).collect();
        samples.push(Sample::new(SparseVector::new(indices, visual_words), 1.0));
    }

    // Class 2: Outdoor scenes (nature, streets) - different visual word distribution
    for i in 0..20 {
        let noise = 0.1 * ((i as f64 / 20.0) - 0.5);

        // Outdoor scenes: natural textures, irregular patterns
        let visual_words = vec![
            8.0,          // Fewer vertical edges
            6.0,          // Fewer horizontal edges
            5.0,          // Fewer corner features
            12.0 + noise, // Natural textures (leaves, bark)
            18.0 + noise, // Smooth regions (sky, water)
            15.0 + noise, // Circular/organic features
            10.0,         // Irregular patterns (rocks, clouds)
            8.0 + noise,  // Complex natural textures
            4.0,          // Rare natural features
            2.0,          // Very rare visual words
        ];

        let indices: Vec<usize> = (0..visual_words.len()).collect();
        samples.push(Sample::new(SparseVector::new(indices, visual_words), -1.0));
    }

    println!(
        "  Generated {} visual bag-of-words histograms (10 visual words)",
        samples.len()
    );

    // Train models - visual words typically use standard (non-normalized) intersection
    let model_hist_visual = SVM::with_histogram_intersection_visual_words()
        .with_c(1.0)
        .train_samples(&samples)?;

    let model_hist_normalized = SVM::with_histogram_intersection_normalized()
        .with_c(1.0)
        .train_samples(&samples)?;

    let model_linear = SVM::new().with_c(1.0).train_samples(&samples)?;

    // Evaluate models
    let visual_acc = evaluate_model(&model_hist_visual, &samples);
    let normalized_acc = evaluate_model(&model_hist_normalized, &samples);
    let linear_acc = evaluate_model(&model_linear, &samples);

    println!("  Results on visual bag-of-words classification:");
    println!(
        "    Histogram Intersection (visual):    {:.1}% accuracy ({} SVs)",
        visual_acc * 100.0,
        model_hist_visual.info().n_support_vectors
    );
    println!(
        "    Histogram Intersection (normalized): {:.1}% accuracy ({} SVs)",
        normalized_acc * 100.0,
        model_hist_normalized.info().n_support_vectors
    );
    println!(
        "    Linear kernel:                      {:.1}% accuracy ({} SVs)",
        linear_acc * 100.0,
        model_linear.info().n_support_vectors
    );

    Ok(())
}

fn test_normalization_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Creating histograms with different total counts...");

    let mut samples = Vec::new();

    // Class 1: High-count histograms (bright images, many features)
    for i in 0..15 {
        let scale = 1.0 + 0.1 * (i as f64 / 15.0); // Varying total counts

        let histogram = vec![
            (50.0 * scale) as f64,
            (30.0 * scale) as f64,
            (20.0 * scale) as f64,
            (10.0 * scale) as f64,
            (5.0 * scale) as f64,
        ];

        let indices: Vec<usize> = (0..histogram.len()).collect();
        samples.push(Sample::new(SparseVector::new(indices, histogram), 1.0));
    }

    // Class 2: Low-count histograms (dark images, fewer features)
    for i in 0..15 {
        let scale = 0.3 + 0.05 * (i as f64 / 15.0); // Much lower counts

        let histogram = vec![
            (10.0 * scale) as f64,
            (20.0 * scale) as f64,
            (30.0 * scale) as f64,
            (25.0 * scale) as f64,
            (15.0 * scale) as f64,
        ];

        let indices: Vec<usize> = (0..histogram.len()).collect();
        samples.push(Sample::new(SparseVector::new(indices, histogram), -1.0));
    }

    println!(
        "  Generated {} histograms with varying total counts",
        samples.len()
    );

    // Compare normalized vs standard intersection
    let model_standard = SVM::with_histogram_intersection_standard()
        .with_c(1.0)
        .train_samples(&samples)?;

    let model_normalized = SVM::with_histogram_intersection_normalized()
        .with_c(1.0)
        .train_samples(&samples)?;

    // Evaluate models
    let standard_acc = evaluate_model(&model_standard, &samples);
    let normalized_acc = evaluate_model(&model_normalized, &samples);

    println!("  Results comparing normalization:");
    println!(
        "    Standard intersection:    {:.1}% accuracy ({} SVs)",
        standard_acc * 100.0,
        model_standard.info().n_support_vectors
    );
    println!(
        "    Normalized intersection:  {:.1}% accuracy ({} SVs)",
        normalized_acc * 100.0,
        model_normalized.info().n_support_vectors
    );
    println!(
        "    Normalization advantage:  {:.1} percentage points",
        (normalized_acc - standard_acc) * 100.0
    );

    Ok(())
}

fn test_kernel_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Creating comprehensive histogram dataset for kernel comparison...");

    let mut samples = Vec::new();

    // Create realistic histogram patterns
    for i in 0..30 {
        let t = i as f64 / 30.0;

        // Class 1: Exponential decay histograms (common in computer vision)
        let exp_hist = vec![
            50.0 * (-t).exp(),
            30.0 * (-1.5 * t).exp(),
            20.0 * (-2.0 * t).exp(),
            15.0 * (-2.5 * t).exp(),
            10.0 * (-3.0 * t).exp(),
            5.0 * (-3.5 * t).exp(),
        ];

        // Class 2: Multi-modal histograms (multiple peaks)
        let multimodal_hist = vec![
            20.0 + 10.0 * (t * 8.0).sin().powi(2),
            15.0 + 8.0 * ((t - 0.3) * 8.0).sin().powi(2),
            25.0 + 12.0 * ((t - 0.7) * 8.0).sin().powi(2),
            18.0 + 9.0 * (t * 6.0).cos().powi(2),
            12.0 + 6.0 * ((t - 0.5) * 6.0).cos().powi(2),
            8.0 + 4.0 * (t * 4.0).sin(),
        ];

        let indices: Vec<usize> = (0..6).collect();
        samples.push(Sample::new(
            SparseVector::new(indices.clone(), exp_hist),
            1.0,
        ));
        samples.push(Sample::new(
            SparseVector::new(indices, multimodal_hist),
            -1.0,
        ));
    }

    println!(
        "  Generated {} samples with complex histogram patterns",
        samples.len()
    );

    // Test different kernels
    let kernels = [
        ("Linear", "linear"),
        ("RBF (γ=1.0)", "rbf"),
        ("Chi-square (γ=1.0)", "chi2"),
        ("Histogram Intersection", "hist_std"),
        ("Hist. Intersection (norm)", "hist_norm"),
        ("Polynomial (d=2)", "poly"),
    ];

    println!("  Comprehensive kernel comparison:");
    println!("    Kernel                   | Accuracy | Support Vectors");
    println!("    -------------------------|----------|----------------");

    for (name, kernel_type) in kernels {
        let model: Box<dyn ModelEvaluator> = match kernel_type {
            "linear" => Box::new(SVM::new().with_c(1.0).train_samples(&samples)?),
            "rbf" => Box::new(SVM::with_rbf(1.0).with_c(1.0).train_samples(&samples)?),
            "chi2" => Box::new(
                SVM::with_chi_square(1.0)
                    .with_c(1.0)
                    .train_samples(&samples)?,
            ),
            "hist_std" => Box::new(
                SVM::with_histogram_intersection_standard()
                    .with_c(1.0)
                    .train_samples(&samples)?,
            ),
            "hist_norm" => Box::new(
                SVM::with_histogram_intersection_normalized()
                    .with_c(1.0)
                    .train_samples(&samples)?,
            ),
            "poly" => Box::new(
                SVM::with_quadratic(1.0)
                    .with_c(1.0)
                    .train_samples(&samples)?,
            ),
            _ => unreachable!(),
        };

        let accuracy = model.evaluate(&samples);
        let n_svs = model.n_support_vectors();

        println!("    {:24} | {:7.1}% | {:14}", name, accuracy * 100.0, n_svs);
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
