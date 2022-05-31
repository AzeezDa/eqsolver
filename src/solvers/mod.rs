mod secant;
pub use secant::*;
mod newton;
pub use newton::*;
mod fdnewton;
pub use fdnewton::*;

const DEFAULT_TOL: f64 = 1e-9;
const DEFAULT_ITERMAX: usize = 50;

pub enum SolverError {
    MaxIterReached,
    IncorrectInput
}