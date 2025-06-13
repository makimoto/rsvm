//! Kernel functions for SVM

pub mod chi_square;
pub mod hellinger;
pub mod histogram_intersection;
pub mod linear;
pub mod polynomial;
pub mod rbf;
pub mod sigmoid;
pub mod traits;

pub use self::chi_square::*;
pub use self::hellinger::*;
pub use self::histogram_intersection::*;
pub use self::linear::*;
pub use self::polynomial::*;
pub use self::rbf::*;
pub use self::sigmoid::*;
pub use self::traits::*;
