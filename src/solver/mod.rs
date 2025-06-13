//! SVM solver implementations
//!
//! This module implements the Sequential Minimal Optimization (SMO) algorithm
//! as described in "Making Large-Scale SVM Learning Practical" by Thorsten Joachims.

pub mod shrinking;
pub mod smo;

pub use self::shrinking::*;
pub use self::smo::*;
