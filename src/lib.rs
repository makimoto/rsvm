//! Rust implementation of Support Vector Machine (SVM)
//!
//! Based on "Making Large-Scale SVM Learning Practical" by Thorsten Joachims

pub mod cache;
pub mod core;
pub mod data;
pub mod kernel;
pub mod optimizer;
pub mod solver;
pub mod utils;

// Re-export main types
pub use crate::core::traits::*;
pub use crate::core::types::*;
pub use crate::data::{LibSVMDataset};
pub use crate::kernel::{Kernel, LinearKernel};

// Version info
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
