//! SVM solver implementations
//!
//! This module implements the Sequential Minimal Optimization (SMO) algorithm
//! as described in "Making Large-Scale SVM Learning Practical" by Thorsten Joachims.

pub mod smo;

pub use self::smo::*;
