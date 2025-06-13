//! Kernel functions for SVM

pub mod linear;
pub mod polynomial;
pub mod rbf;
pub mod traits;

pub use self::linear::*;
pub use self::polynomial::*;
pub use self::rbf::*;
pub use self::traits::*;
