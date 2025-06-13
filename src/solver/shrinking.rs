//! Shrinking heuristic implementation
//!
//! Implements the shrinking strategy from Section 4 of the SVMlight paper
//! "Making Large-Scale SVM Learning Practical" by Thorsten Joachims.
//!
//! The shrinking heuristic identifies variables that are likely to remain
//! at their bounds (0 or C) and temporarily removes them from optimization,
//! reducing the size of the problem and improving performance.

use crate::core::Sample;
use std::collections::VecDeque;

/// Shrinking strategy for SVM optimization
///
/// Tracks Lagrange multiplier estimates over h iterations to identify
/// variables that can be safely shrunk to their bounds.
#[derive(Debug)]
pub struct ShrinkingStrategy {
    /// History of lower bound indicators for each variable
    lower_bound_history: Vec<VecDeque<bool>>,
    /// History of upper bound indicators for each variable  
    upper_bound_history: Vec<VecDeque<bool>>,
    /// Maximum history size (h in the paper)
    history_size: usize,
    /// Current iteration count
    current_iteration: usize,
}

impl ShrinkingStrategy {
    /// Create a new shrinking strategy
    ///
    /// # Arguments
    /// * `n_samples` - Number of training samples
    /// * `history_size` - Number of iterations to track history (h in paper)
    pub fn new(n_samples: usize, history_size: usize) -> Self {
        Self {
            lower_bound_history: vec![VecDeque::with_capacity(history_size); n_samples],
            upper_bound_history: vec![VecDeque::with_capacity(history_size); n_samples],
            history_size,
            current_iteration: 0,
        }
    }

    /// Update multiplier estimates and track history
    ///
    /// Based on equations 29-31 from the SVMlight paper:
    /// - λ^eq = (1/|A|) Σ[yi - Σ αj*yj*K(xi,xj)] for i in A
    /// - λi^lo = yi*[Σ αj*yj*K(xi,xj) + λ^eq] - 1  
    /// - λi^up = -yi*[Σ αj*yj*K(xi,xj) + λ^eq] + 1
    pub fn update(&mut self, alpha: &[f64], error_cache: &[f64], samples: &[Sample], c: f64) {
        let n = samples.len();

        // Calculate λ^eq (equation 29 from paper) - currently unused but kept for future use
        let _lambda_eq = self.estimate_lambda_eq(alpha, error_cache, samples, c);

        // Update history for each variable
        for i in 0..n {
            let yi = samples[i].label;
            let ei = error_cache[i];

            // From the paper equations 30-31:
            // λi^lo = yi*[Σ αj*yj*K(xi,xj) + λ^eq] - 1
            // λi^up = -yi*[Σ αj*yj*K(xi,xj) + λ^eq] + 1
            // Since error_i = output_i - yi = [Σ αj*yj*K(xi,xj) + b] - yi
            // We have: Σ αj*yj*K(xi,xj) + b = ei + yi
            // So: Σ αj*yj*K(xi,xj) + λ^eq = ei + yi (assuming b ≈ λ^eq)
            let prediction = ei + yi; // This is Σ αj*yj*K(xi,xj) + b
            let lambda_lower = yi * prediction - 1.0;
            let lambda_upper = -yi * prediction + 1.0;

            // Variable should be shrunk if:
            // - At lower bound (alpha ≈ 0) and lower multiplier > 0 (wants to stay at 0)
            // - At upper bound (alpha ≈ C) and upper multiplier < 0 (wants to stay at C)
            let at_lower_bound = alpha[i] <= 1e-8 && lambda_lower > 1e-6;
            let at_upper_bound = alpha[i] >= c - 1e-8 && lambda_upper < -1e-6;

            // Update history queues
            Self::update_history(
                &mut self.lower_bound_history[i],
                at_lower_bound,
                self.history_size,
            );
            Self::update_history(
                &mut self.upper_bound_history[i],
                at_upper_bound,
                self.history_size,
            );
        }

        self.current_iteration += 1;
    }

    /// Estimate λ^eq using free support vectors (equation 29)
    fn estimate_lambda_eq(
        &self,
        alpha: &[f64],
        error_cache: &[f64],
        samples: &[Sample],
        c: f64,
    ) -> f64 {
        let mut sum = 0.0;
        let mut count = 0;

        // Use free support vectors (0 < alpha < C) for estimation
        for i in 0..alpha.len() {
            if alpha[i] > 1e-8 && alpha[i] < c - 1e-8 {
                // For free SVs: yi * (output + b) = 1, so b = (yi - output) / yi = yi - yi*output
                // Since error_i = output_i - yi, we have output_i = error_i + yi
                // Therefore: b = yi - yi*(error_i + yi) = yi - yi*error_i - yi^2 = -yi*error_i (since yi^2 = 1)
                sum += -samples[i].label * error_cache[i];
                count += 1;
            }
        }

        if count > 0 {
            sum / count as f64
        } else {
            0.0
        }
    }

    /// Update history queue for a variable
    fn update_history(history: &mut VecDeque<bool>, value: bool, history_size: usize) {
        if history.len() >= history_size {
            history.pop_front();
        }
        history.push_back(value);
    }

    /// Get variables that can be shrunk
    ///
    /// Returns (lower_bound_indices, upper_bound_indices) where:
    /// - lower_bound_indices: variables to shrink to alpha = 0
    /// - upper_bound_indices: variables to shrink to alpha = C
    pub fn get_shrinkable_variables(&self) -> (Vec<usize>, Vec<usize>) {
        let mut shrink_to_lower = Vec::new();
        let mut shrink_to_upper = Vec::new();

        for i in 0..self.lower_bound_history.len() {
            // Check if consistently at lower bound
            if self.lower_bound_history[i].len() == self.history_size
                && self.lower_bound_history[i].iter().all(|&x| x)
            {
                shrink_to_lower.push(i);
            }

            // Check if consistently at upper bound
            if self.upper_bound_history[i].len() == self.history_size
                && self.upper_bound_history[i].iter().all(|&x| x)
            {
                shrink_to_upper.push(i);
            }
        }

        (shrink_to_lower, shrink_to_upper)
    }

    /// Check if enough history has been accumulated
    pub fn has_sufficient_history(&self) -> bool {
        self.current_iteration >= self.history_size
    }

    /// Reset history (used when unshrinking variables)
    pub fn reset_history(&mut self) {
        for history in &mut self.lower_bound_history {
            history.clear();
        }
        for history in &mut self.upper_bound_history {
            history.clear();
        }
        self.current_iteration = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::SparseVector;

    fn create_test_samples() -> Vec<Sample> {
        vec![
            Sample::new(SparseVector::new(vec![0], vec![1.0]), 1.0),
            Sample::new(SparseVector::new(vec![0], vec![-1.0]), -1.0),
            Sample::new(SparseVector::new(vec![0], vec![2.0]), 1.0),
        ]
    }

    #[test]
    fn test_shrinking_strategy_creation() {
        let strategy = ShrinkingStrategy::new(3, 5);
        assert_eq!(strategy.lower_bound_history.len(), 3);
        assert_eq!(strategy.upper_bound_history.len(), 3);
        assert_eq!(strategy.history_size, 5);
        assert_eq!(strategy.current_iteration, 0);
        assert!(!strategy.has_sufficient_history());
    }

    #[test]
    fn test_lambda_eq_estimation() {
        let strategy = ShrinkingStrategy::new(3, 5);
        let samples = create_test_samples();
        let alpha = vec![0.5, 0.5, 0.0]; // Two free SVs, one at bound
        let error_cache = vec![0.1, -0.1, -1.0]; // Errors for each sample
        let c = 1.0;

        let lambda_eq = strategy.estimate_lambda_eq(&alpha, &error_cache, &samples, c);

        // For free SVs (indices 0,1): lambda_eq should be average of -yi*ei
        // -y0*e0 = -1.0*0.1 = -0.1
        // -y1*e1 = -(-1.0)*(-0.1) = -0.1
        // Average: (-0.1 + -0.1) / 2 = -0.1
        assert!((lambda_eq - (-0.1)).abs() < 1e-6);
    }

    #[test]
    fn test_update_history() {
        let mut strategy = ShrinkingStrategy::new(2, 3);
        let samples = create_test_samples()[..2].to_vec();

        // For alpha=0 to be optimal, we need yi*(output + b) >= 1
        // This means output >= yi - b
        // Since error = output - yi, we get output = error + yi
        // So we need error + yi >= yi - b, which gives error >= -b
        // For b=0, we need error >= 0
        // But if we want to shrink, we need the multiplier estimate to be positive
        // Let's create a scenario where variable 0 should be shrunk to lower bound

        for _ in 0..3 {
            let alpha = vec![0.0, 0.5]; // First at bound, second free
                                        // For shrinking to lower bound: need lambda_lower > 0
                                        // lambda_lower = yi * prediction - 1 = yi * (ei + yi) - 1
                                        // For yi=1: lambda_lower = 1 * (ei + 1) - 1 = ei
                                        // So we need ei > 0 for lower bound shrinking
            let error_cache = vec![0.5, 0.0]; // Positive error for first variable
            strategy.update(&alpha, &error_cache, &samples, 1.0);
        }

        assert!(strategy.has_sufficient_history());
        let (lower, upper) = strategy.get_shrinkable_variables();

        // First variable should be identified for shrinking to lower bound
        assert!(lower.contains(&0));
        assert!(upper.is_empty());
    }

    #[test]
    fn test_shrinking_to_upper_bound() {
        let mut strategy = ShrinkingStrategy::new(2, 3);
        let samples = create_test_samples()[..2].to_vec();
        let c = 1.0;

        // For shrinking to upper bound: need lambda_upper < 0
        // lambda_upper = -yi * prediction + 1 = -yi * (ei + yi) + 1
        // For yi=1: lambda_upper = -(ei + 1) + 1 = -ei
        // So we need ei > 0 for upper bound shrinking (lambda_upper < 0)

        for _ in 0..3 {
            let alpha = vec![c, 0.5]; // First at upper bound, second free
            let error_cache = vec![0.5, 0.0]; // Positive error for upper bound shrinking
            strategy.update(&alpha, &error_cache, &samples, c);
        }

        let (lower, upper) = strategy.get_shrinkable_variables();

        // First variable should be identified for shrinking to upper bound
        assert!(upper.contains(&0));
        assert!(lower.is_empty());
    }

    #[test]
    fn test_insufficient_history() {
        let mut strategy = ShrinkingStrategy::new(2, 5);
        let samples = create_test_samples()[..2].to_vec();

        // Only update twice (less than history_size)
        for _ in 0..2 {
            let alpha = vec![0.0, 0.5];
            let error_cache = vec![-1.0, 0.0];
            strategy.update(&alpha, &error_cache, &samples, 1.0);
        }

        let (lower, upper) = strategy.get_shrinkable_variables();

        // Should not shrink with insufficient history
        assert!(lower.is_empty());
        assert!(upper.is_empty());
    }

    #[test]
    fn test_reset_history() {
        let mut strategy = ShrinkingStrategy::new(2, 3);
        let samples = create_test_samples()[..2].to_vec();

        // Build up some history
        for _ in 0..3 {
            let alpha = vec![0.0, 0.5];
            let error_cache = vec![-1.0, 0.0];
            strategy.update(&alpha, &error_cache, &samples, 1.0);
        }

        assert!(strategy.has_sufficient_history());

        strategy.reset_history();

        assert!(!strategy.has_sufficient_history());
        assert_eq!(strategy.current_iteration, 0);

        let (lower, upper) = strategy.get_shrinkable_variables();
        assert!(lower.is_empty());
        assert!(upper.is_empty());
    }

    #[test]
    fn test_mixed_shrinking_behavior() {
        let mut strategy = ShrinkingStrategy::new(3, 4);
        let samples = create_test_samples();
        let c = 1.0;

        // Simulate different behaviors for each variable
        for iteration in 0..4 {
            let alpha = if iteration < 2 {
                vec![0.0, c, 0.3] // var0->lower, var1->upper, var2->free
            } else {
                vec![0.0, c, 0.8] // var0->lower, var1->upper, var2->free
            };
            // Use appropriate error values for shrinking conditions
            // var0 (yi=1): need ei > 0 for lower bound shrinking (lambda_lower = ei > 0)
            // var1 (yi=-1): need ei such that lambda_upper < 0
            //   lambda_upper = -(-1) * (ei + (-1)) + 1 = ei - 1 + 1 = ei
            //   So need ei < 0 for upper bound shrinking
            let error_cache = vec![0.5, -0.5, 0.0]; // Correct conditions for shrinking
            strategy.update(&alpha, &error_cache, &samples, c);
        }

        let (lower, upper) = strategy.get_shrinkable_variables();

        // Both first variables should be shrinkable
        assert!(lower.contains(&0));
        assert!(upper.contains(&1));
        // Third variable should not be shrinkable (free)
        assert!(!lower.contains(&2));
        assert!(!upper.contains(&2));
    }
}
