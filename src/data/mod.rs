//! Data loading and dataset implementations
//!
//! This module provides implementations of the Dataset trait for various
//! data formats commonly used in machine learning.

pub mod libsvm;

pub use self::libsvm::*;
