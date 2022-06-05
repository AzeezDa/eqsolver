mod secant;
pub use secant::*;
mod newton;
pub use newton::*;
mod fdnewton;
pub use fdnewton::*;
mod multinewton;
pub use multinewton::*;
mod multinewton_fd;
pub use multinewton_fd::*;
mod gaussnewton;
pub use gaussnewton::*;
mod gaussnewton_fd;
pub use gaussnewton_fd::*;

pub const DEFAULT_TOL: f64 = 1e-9;
pub const DEFAULT_ITERMAX: usize = 50;

#[derive(Debug)]
pub enum SolverError {
    MaxIterReached,
    IncorrectInput,
    BadJacobian
}