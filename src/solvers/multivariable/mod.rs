mod gaussnewton;
mod gaussnewton_fd;
mod levenberg_marquardt;
mod levenberg_marquardt_fd;
mod multinewton;
mod multinewton_fd;

pub use gaussnewton::GaussNewton;
pub use gaussnewton_fd::GaussNewtonFD;
pub use levenberg_marquardt::LevenbergMarquardt;
pub use levenberg_marquardt_fd::LevenbergMarquardtFD;
pub use multinewton::MultiVarNewton;
pub use multinewton_fd::MultiVarNewtonFD;

use super::{MatrixType, VectorType};
