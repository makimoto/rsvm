//! Rust implementation of Support Vector Machine (SVM)
//!
//! Based on "Making Large-Scale SVM Learning Practical" by Thorsten Joachims

pub mod api;
pub mod cache;
pub mod core;
pub mod data;
pub mod kernel;
pub mod optimizer;
pub mod solver;
pub mod utils;

// Re-export main types for convenience
pub use crate::api::{EvaluationMetrics, ModelInfo, TrainedModel, SVM};
pub use crate::cache::{CacheStats, KernelCache};
pub use crate::core::traits::*;
pub use crate::core::types::*;
pub use crate::data::{CSVDataset, LibSVMDataset};
pub use crate::kernel::{Kernel, LinearKernel};
pub use crate::optimizer::{SVMOptimizer, TrainedSVM};
pub use crate::utils::SparseVectorStats;

// Version info
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
