//! Hellinger Kernel Demonstration
//!
//! This example demonstrates the use of Hellinger kernels for probability distribution
//! and normalized data classification tasks. The Hellinger kernel is particularly
//! effective for applications where features represent probabilities, frequencies,
//! or other non-negative normalized values.
//!
//! Applications demonstrated:
//! - Text mining with TF-IDF vectors
//! - Bioinformatics with species abundance data
//! - Statistical analysis of probability distributions
//! - Document classification and semantic similarity
//! - Normalized feature comparison

use rsvm::api::SVM;
use rsvm::core::{Sample, SparseVector};
use rsvm::kernel::Kernel;
use rsvm::TrainedModel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Hellinger Kernel Demonstration ===");
    println!();

    // Test 1: Text Mining with TF-IDF-like Vectors
    println!("📄 Test 1: Text Mining with TF-IDF Vectors");
    test_text_mining_classification()?;
    println!();

    // Test 2: Bioinformatics with Species Abundance
    println!("🧬 Test 2: Bioinformatics Species Abundance Analysis");
    test_species_abundance_classification()?;
    println!();

    // Test 3: Probability Distribution Comparison
    println!("📊 Test 3: Probability Distribution Classification");
    test_probability_distribution_classification()?;
    println!();

    // Test 4: Normalized vs Standard Hellinger
    println!("⚖️  Test 4: Normalized vs Standard Hellinger Comparison");
    test_normalization_effects()?;
    println!();

    // Test 5: Kernel Comparison on Distribution Data
    println!("🔬 Test 5: Kernel Comparison for Distribution Data");
    test_kernel_comparison()?;

    Ok(())
}

fn test_text_mining_classification() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Creating TF-IDF document vectors for topic classification...");

    let mut samples = Vec::new();

    // Class 1: Technology documents (normalized TF-IDF vectors)
    // Terms: [machine, learning, algorithm, computer, science, data, neural, network, programming, software]
    for i in 0..25 {
        let noise = 0.02 * ((i as f64 / 25.0) - 0.5);

        // Tech documents emphasize ML/CS terms
        let tfidf_vector = vec![
            0.25 + noise, // machine
            0.22 + noise, // learning
            0.18 + noise, // algorithm
            0.15,         // computer
            0.12,         // science
            0.08,         // data
            0.0,          // neural (some docs don't have this)
            0.0,          // network
            0.0,          // programming
            0.0,          // software
        ];

        let indices: Vec<usize> = (0..tfidf_vector.len()).collect();
        samples.push(Sample::new(SparseVector::new(indices, tfidf_vector), 1.0));
    }

    // Class 2: Biology documents (normalized TF-IDF vectors)
    // Terms: [cell, protein, gene, evolution, species, organism, molecular, biology, dna, genetic]
    for i in 0..25 {
        let noise = 0.02 * ((i as f64 / 25.0) - 0.5);

        // Biology documents use different vocabulary
        let tfidf_vector = vec![
            0.0,          // machine (not used)
            0.0,          // learning (not used)
            0.0,          // algorithm (not used)
            0.0,          // computer (not used)
            0.08,         // science (shared term)
            0.12 + noise, // data (shared term, lower weight)
            0.18 + noise, // cell (bio term)
            0.22 + noise, // protein (bio term)
            0.25 + noise, // gene (bio term)
            0.15,         // evolution (bio term)
        ];

        let indices: Vec<usize> = (0..tfidf_vector.len()).collect();
        samples.push(Sample::new(SparseVector::new(indices, tfidf_vector), -1.0));
    }

    println!(
        "  Generated {} TF-IDF document vectors (10 terms each)",
        samples.len()
    );

    // Train different models for text mining
    let model_hellinger_text = SVM::with_hellinger_text()
        .with_c(1.0)
        .train_samples(&samples)?;

    let model_linear = SVM::new().with_c(1.0).train_samples(&samples)?;

    let model_chi2 = SVM::with_chi_square_text()
        .with_c(1.0)
        .train_samples(&samples)?;

    // Evaluate models
    let hellinger_acc = evaluate_model(&model_hellinger_text, &samples);
    let linear_acc = evaluate_model(&model_linear, &samples);
    let chi2_acc = evaluate_model(&model_chi2, &samples);

    println!("  Results on text classification:");
    println!(
        "    Hellinger (text):    {:.1}% accuracy ({} SVs)",
        hellinger_acc * 100.0,
        model_hellinger_text.info().n_support_vectors
    );
    println!(
        "    Linear kernel:       {:.1}% accuracy ({} SVs)",
        linear_acc * 100.0,
        model_linear.info().n_support_vectors
    );
    println!(
        "    Chi-square (text):   {:.1}% accuracy ({} SVs)",
        chi2_acc * 100.0,
        model_chi2.info().n_support_vectors
    );
    println!(
        "    Hellinger advantage: {:.1} percentage points over linear",
        (hellinger_acc - linear_acc) * 100.0
    );

    // Test on ambiguous document (interdisciplinary: bioinformatics)
    let bioinformatics_doc = vec![
        0.15, // machine (some ML in bioinformatics)
        0.12, // learning (some ML terms)
        0.18, // algorithm (bioinformatics algorithms)
        0.08, // computer (computational biology)
        0.10, // science (shared)
        0.20, // data (lots of biological data)
        0.10, // cell (biological context)
        0.07, // protein (biological context)
        0.0,  // gene
        0.0,  // evolution
    ];
    let indices: Vec<usize> = (0..bioinformatics_doc.len()).collect();
    let bio_sample = Sample::new(SparseVector::new(indices, bioinformatics_doc), 0.0);

    let hellinger_pred = model_hellinger_text.predict(&bio_sample);
    println!(
        "  Bioinformatics doc prediction: {} (confidence: {:.3})",
        if hellinger_pred.label > 0.0 {
            "Technology"
        } else {
            "Biology"
        },
        hellinger_pred.decision_value.abs()
    );

    Ok(())
}

fn test_species_abundance_classification() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Creating species abundance vectors for ecosystem classification...");

    let mut samples = Vec::new();

    // Class 1: Forest ecosystems (normalized species abundance)
    // Species: [oak, pine, maple, birch, deer, squirrel, owl, robin, fern, moss]
    for i in 0..20 {
        let variation = 0.05 * ((i as f64 / 20.0) - 0.5);

        // Forest ecosystems: trees dominate, forest animals present
        let abundance = vec![
            0.28 + variation, // oak (dominant)
            0.25 + variation, // pine (dominant)
            0.15,             // maple
            0.12,             // birch
            0.08,             // deer
            0.06,             // squirrel
            0.03,             // owl
            0.02,             // robin
            0.01,             // fern
            0.0,              // moss (minimal)
        ];

        let indices: Vec<usize> = (0..abundance.len()).collect();
        samples.push(Sample::new(SparseVector::new(indices, abundance), 1.0));
    }

    // Class 2: Grassland ecosystems (different species distribution)
    // Same species list but different abundance patterns
    for i in 0..20 {
        let variation = 0.05 * ((i as f64 / 20.0) - 0.5);

        // Grasslands: fewer trees, more grass-adapted species
        let abundance = vec![
            0.02,             // oak (rare)
            0.01,             // pine (rare)
            0.03,             // maple (rare)
            0.02,             // birch (rare)
            0.25 + variation, // deer (adapted to grassland)
            0.08,             // squirrel (fewer without trees)
            0.05,             // owl
            0.20 + variation, // robin (thrives in grassland)
            0.18 + variation, // fern (ground cover)
            0.16 + variation, // moss (ground cover)
        ];

        let indices: Vec<usize> = (0..abundance.len()).collect();
        samples.push(Sample::new(SparseVector::new(indices, abundance), -1.0));
    }

    println!(
        "  Generated {} species abundance vectors (10 species)",
        samples.len()
    );

    // Train models specifically for bioinformatics
    let model_hellinger_bio = SVM::with_hellinger_bio()
        .with_c(1.0)
        .train_samples(&samples)?;

    let model_hellinger_std = SVM::with_hellinger_standard()
        .with_c(1.0)
        .train_samples(&samples)?;

    let model_linear = SVM::new().with_c(1.0).train_samples(&samples)?;

    // Evaluate models
    let bio_acc = evaluate_model(&model_hellinger_bio, &samples);
    let std_acc = evaluate_model(&model_hellinger_std, &samples);
    let linear_acc = evaluate_model(&model_linear, &samples);

    println!("  Results on ecosystem classification:");
    println!(
        "    Hellinger (bio):        {:.1}% accuracy ({} SVs)",
        bio_acc * 100.0,
        model_hellinger_bio.info().n_support_vectors
    );
    println!(
        "    Hellinger (standard):   {:.1}% accuracy ({} SVs)",
        std_acc * 100.0,
        model_hellinger_std.info().n_support_vectors
    );
    println!(
        "    Linear kernel:          {:.1}% accuracy ({} SVs)",
        linear_acc * 100.0,
        model_linear.info().n_support_vectors
    );
    println!(
        "    Bio optimization gain:  {:.1} percentage points",
        (bio_acc - std_acc) * 100.0
    );

    Ok(())
}

fn test_probability_distribution_classification() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Creating probability distributions for mixture model classification...");

    let mut samples = Vec::new();

    // Class 1: Gaussian-like distributions (bell curve approximation)
    for i in 0..15 {
        let t = i as f64 / 15.0;

        // Discrete approximation of Gaussian: more probability in center
        let gaussian_approx = vec![
            0.05 + 0.02 * t, // tail
            0.15 + 0.05 * t, // approaching center
            0.30 + 0.08 * t, // center-left
            0.25 + 0.06 * t, // center-right
            0.15 + 0.04 * t, // approaching tail
            0.10 + 0.02 * t, // tail
        ];

        // Normalize to ensure sum = 1.0
        let sum: f64 = gaussian_approx.iter().sum();
        let normalized: Vec<f64> = gaussian_approx.iter().map(|x| x / sum).collect();

        let indices: Vec<usize> = (0..normalized.len()).collect();
        samples.push(Sample::new(SparseVector::new(indices, normalized), 1.0));
    }

    // Class 2: Exponential-like distributions (decay pattern)
    for i in 0..15 {
        let t = i as f64 / 15.0;

        // Exponential decay pattern: high probability at start, decreasing
        let exponential_approx = vec![
            0.50 + 0.10 * t, // high start
            0.25 + 0.05 * t, // decay
            0.15 + 0.03 * t, // further decay
            0.06 + 0.02 * t, // low
            0.03 + 0.01 * t, // very low
            0.01,            // minimal tail
        ];

        // Normalize to ensure sum = 1.0
        let sum: f64 = exponential_approx.iter().sum();
        let normalized: Vec<f64> = exponential_approx.iter().map(|x| x / sum).collect();

        let indices: Vec<usize> = (0..normalized.len()).collect();
        samples.push(Sample::new(SparseVector::new(indices, normalized), -1.0));
    }

    println!(
        "  Generated {} probability distributions (6 bins each)",
        samples.len()
    );

    // Train models for probability distribution analysis
    let model_hellinger_prob = SVM::with_hellinger_probability()
        .with_c(1.0)
        .train_samples(&samples)?;

    let model_hellinger_stats = SVM::with_hellinger_stats()
        .with_c(1.0)
        .train_samples(&samples)?;

    let model_hist_norm = SVM::with_histogram_intersection_normalized()
        .with_c(1.0)
        .train_samples(&samples)?;

    let model_linear = SVM::new().with_c(1.0).train_samples(&samples)?;

    // Evaluate models
    let prob_acc = evaluate_model(&model_hellinger_prob, &samples);
    let stats_acc = evaluate_model(&model_hellinger_stats, &samples);
    let hist_acc = evaluate_model(&model_hist_norm, &samples);
    let linear_acc = evaluate_model(&model_linear, &samples);

    println!("  Results on probability distribution classification:");
    println!(
        "    Hellinger (probability): {:.1}% accuracy ({} SVs)",
        prob_acc * 100.0,
        model_hellinger_prob.info().n_support_vectors
    );
    println!(
        "    Hellinger (stats):       {:.1}% accuracy ({} SVs)",
        stats_acc * 100.0,
        model_hellinger_stats.info().n_support_vectors
    );
    println!(
        "    Histogram Intersection:  {:.1}% accuracy ({} SVs)",
        hist_acc * 100.0,
        model_hist_norm.info().n_support_vectors
    );
    println!(
        "    Linear kernel:           {:.1}% accuracy ({} SVs)",
        linear_acc * 100.0,
        model_linear.info().n_support_vectors
    );

    // Test on mixed distribution (bimodal: between Gaussian and exponential)
    let bimodal_dist = vec![0.25, 0.15, 0.10, 0.15, 0.20, 0.15];
    let indices: Vec<usize> = (0..bimodal_dist.len()).collect();
    let mixed_sample = Sample::new(SparseVector::new(indices, bimodal_dist), 0.0);

    let prob_pred = model_hellinger_prob.predict(&mixed_sample);
    println!(
        "  Bimodal distribution prediction: {} (confidence: {:.3})",
        if prob_pred.label > 0.0 {
            "Gaussian-like"
        } else {
            "Exponential-like"
        },
        prob_pred.decision_value.abs()
    );

    Ok(())
}

fn test_normalization_effects() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Creating distributions with varying total masses...");

    let mut samples = Vec::new();

    // Class 1: High-magnitude distributions (bright/intense features)
    for i in 0..12 {
        let scale = 2.0 + 0.5 * (i as f64 / 12.0); // Varying magnitudes

        let high_intensity = vec![
            (0.3 * scale) as f64,
            (0.25 * scale) as f64,
            (0.2 * scale) as f64,
            (0.15 * scale) as f64,
            (0.1 * scale) as f64,
        ];

        let indices: Vec<usize> = (0..high_intensity.len()).collect();
        samples.push(Sample::new(SparseVector::new(indices, high_intensity), 1.0));
    }

    // Class 2: Low-magnitude distributions (dim/weak features)
    for i in 0..12 {
        let scale = 0.4 + 0.1 * (i as f64 / 12.0); // Much lower magnitudes

        let low_intensity = vec![
            (0.15 * scale) as f64,
            (0.2 * scale) as f64,
            (0.25 * scale) as f64,
            (0.25 * scale) as f64,
            (0.15 * scale) as f64,
        ];

        let indices: Vec<usize> = (0..low_intensity.len()).collect();
        samples.push(Sample::new(SparseVector::new(indices, low_intensity), -1.0));
    }

    println!(
        "  Generated {} distributions with varying magnitudes",
        samples.len()
    );

    // Compare standard vs normalized Hellinger
    let model_standard = SVM::with_hellinger_standard()
        .with_c(1.0)
        .train_samples(&samples)?;

    let model_normalized = SVM::with_hellinger_normalized()
        .with_c(1.0)
        .train_samples(&samples)?;

    // Evaluate models
    let standard_acc = evaluate_model(&model_standard, &samples);
    let normalized_acc = evaluate_model(&model_normalized, &samples);

    println!("  Results comparing normalization effects:");
    println!(
        "    Standard Hellinger:   {:.1}% accuracy ({} SVs)",
        standard_acc * 100.0,
        model_standard.info().n_support_vectors
    );
    println!(
        "    Normalized Hellinger: {:.1}% accuracy ({} SVs)",
        normalized_acc * 100.0,
        model_normalized.info().n_support_vectors
    );
    println!(
        "    Normalization gain:   {:.1} percentage points",
        (normalized_acc - standard_acc) * 100.0
    );

    // Demonstrate effect on prediction values
    let test_high = vec![0.6, 0.5, 0.4, 0.3, 0.2];
    let test_low = vec![0.12, 0.10, 0.08, 0.06, 0.04];
    let indices: Vec<usize> = (0..5).collect();

    let high_sample = Sample::new(SparseVector::new(indices.clone(), test_high), 0.0);
    let low_sample = Sample::new(SparseVector::new(indices, test_low), 0.0);

    let std_high = model_standard.predict(&high_sample);
    let std_low = model_standard.predict(&low_sample);
    let norm_high = model_normalized.predict(&high_sample);
    let norm_low = model_normalized.predict(&low_sample);

    println!("  Prediction values for similar patterns, different scales:");
    println!(
        "    Standard kernel:  High={:.3}, Low={:.3} (ratio: {:.2})",
        std_high.decision_value,
        std_low.decision_value,
        std_high.decision_value / std_low.decision_value.max(0.001)
    );
    println!(
        "    Normalized kernel: High={:.3}, Low={:.3} (ratio: {:.2})",
        norm_high.decision_value,
        norm_low.decision_value,
        norm_high.decision_value / norm_low.decision_value.max(0.001)
    );

    Ok(())
}

fn test_kernel_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Creating comprehensive dataset for kernel comparison...");

    let mut samples = Vec::new();

    // Create diverse probability distributions and normalized feature patterns
    for i in 0..20 {
        let t = i as f64 / 20.0;

        // Class 1: Multi-modal distributions (complex patterns)
        let multimodal = vec![
            0.20 + 0.10 * (t * 6.0).sin().powi(2),
            0.15 + 0.08 * ((t - 0.3) * 6.0).sin().powi(2),
            0.25 + 0.12 * ((t - 0.7) * 6.0).sin().powi(2),
            0.18 + 0.09 * (t * 4.0).cos().powi(2),
            0.12 + 0.06 * ((t - 0.5) * 4.0).cos().powi(2),
            0.10 + 0.05 * (t * 8.0).sin(),
        ];

        // Normalize to valid probability distribution
        let sum: f64 = multimodal.iter().sum();
        let normalized: Vec<f64> = multimodal.iter().map(|x| x / sum).collect();

        // Class 2: Power-law distributions (heavy-tailed)
        let power_law = vec![
            0.40 * (1.0 + t).powf(-1.5),
            0.30 * (2.0 + t).powf(-1.5),
            0.20 * (3.0 + t).powf(-1.5),
            0.15 * (4.0 + t).powf(-1.5),
            0.10 * (5.0 + t).powf(-1.5),
            0.05 * (6.0 + t).powf(-1.5),
        ];

        // Normalize power-law distribution
        let sum2: f64 = power_law.iter().sum();
        let normalized2: Vec<f64> = power_law.iter().map(|x| x / sum2).collect();

        let indices: Vec<usize> = (0..6).collect();
        samples.push(Sample::new(
            SparseVector::new(indices.clone(), normalized),
            1.0,
        ));
        samples.push(Sample::new(SparseVector::new(indices, normalized2), -1.0));
    }

    println!(
        "  Generated {} samples with complex distribution patterns",
        samples.len()
    );

    // Test comprehensive kernel comparison
    let kernels = [
        ("Linear", "linear"),
        ("RBF (γ=1.0)", "rbf"),
        ("Chi-square (γ=1.0)", "chi2"),
        ("Histogram Intersection", "hist"),
        ("Hellinger (standard)", "hellinger_std"),
        ("Hellinger (normalized)", "hellinger_norm"),
        ("Hellinger (probability)", "hellinger_prob"),
        ("Polynomial (d=2)", "poly"),
    ];

    println!("  Comprehensive kernel comparison on probability distributions:");
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
            "hist" => Box::new(
                SVM::with_histogram_intersection_normalized()
                    .with_c(1.0)
                    .train_samples(&samples)?,
            ),
            "hellinger_std" => Box::new(
                SVM::with_hellinger_standard()
                    .with_c(1.0)
                    .train_samples(&samples)?,
            ),
            "hellinger_norm" => Box::new(
                SVM::with_hellinger_normalized()
                    .with_c(1.0)
                    .train_samples(&samples)?,
            ),
            "hellinger_prob" => Box::new(
                SVM::with_hellinger_probability()
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
