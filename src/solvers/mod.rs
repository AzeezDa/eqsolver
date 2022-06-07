mod secant;
mod newton;
mod fdnewton;
mod multinewton;
mod multinewton_fd;
mod gaussnewton;
mod gaussnewton_fd;

pub mod single_variable {
    pub use super::{
        secant::Secant,
        newton::Newton,
        fdnewton::FDNewton,
    };
}

pub mod multivariable {
    pub use super::{
        multinewton::MultiVarNewton,
        multinewton_fd::MultiVarNewtonFD,
        gaussnewton::GaussNewton,
        gaussnewton_fd::GaussNewtonFD
    };
}


pub const DEFAULT_TOL: f64 = 1e-6;
pub const DEFAULT_ITERMAX: usize = 50;

#[derive(Debug, PartialEq)]
pub enum SolverError {
    MaxIterReached,
    NotANumber,
    IncorrectInput,
    BadJacobian
}

pub type SolverResult<T> = Result<T, SolverError>;