mod secant;
mod newton;
mod fdnewton;
mod multinewton;
mod multinewton_fd;
mod gaussnewton;
mod gaussnewton_fd;

pub use self::{
    secant::SecantSolver,
    newton::NewtonSolver,
    fdnewton::FDNewton,
};

pub const DEFAULT_TOL: f64 = 1e-9;
pub const DEFAULT_ITERMAX: usize = 50;

#[derive(Debug)]
pub enum SolverError {
    MaxIterReached,
    IncorrectInput,
    BadJacobian
}