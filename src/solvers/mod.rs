use nalgebra::{allocator::Allocator, DefaultAllocator, Matrix, Vector};

mod fdnewton;
mod newton;
mod secant;

mod gaussnewton;
mod gaussnewton_fd;
mod levenberg_marquardt;
mod levenberg_marquardt_fd;
mod multinewton;
mod multinewton_fd;
mod ode_solver;
mod particle_swarm;

/// Methods of derivative approximation
pub mod finite_differences;

/// Root-finders for equations of a single variable
pub mod single_variable {
    pub use super::{fdnewton::FDNewton, newton::Newton, secant::Secant};
}

/// Root-finders for equations of multiple variables
pub mod multivariable {
    pub use super::{
        gaussnewton::GaussNewton, gaussnewton_fd::GaussNewtonFD,
        levenberg_marquardt::LevenbergMarquardt, levenberg_marquardt_fd::LevenbergMarquardtFD,
        multinewton::MultiVarNewton, multinewton_fd::MultiVarNewtonFD,
    };
}

/// Finds global optimums of objective functions
pub mod global_optimisers {
    pub use super::particle_swarm::ParticleSwarm;
}

/// Ordinary Differential Equation solvers
pub use ode_solver::*;

/// Default tolerance (error of magnitude)
pub const DEFAULT_TOL: f64 = 1e-6;

/// Default amount of max iterations for the iterative methods
pub const DEFAULT_ITERMAX: usize = 50;

/// Types of errors encountered by the solvers
#[derive(Debug, PartialEq)]
pub enum SolverError {
    /// The amount of iterations reached the limit
    MaxIterReached,

    /// The value evalutated is a `NaN`
    NotANumber,

    /// The given input is not correct
    IncorrectInput,

    /// A Jacobian Matrix in the iteration was singular or ill-defined
    BadJacobian,
}

/// Alias for `Result<T, SolverError>`
pub type SolverResult<T> = Result<T, SolverError>;

type VectorType<T, D> = Vector<T, D, <DefaultAllocator as Allocator<D>>::Buffer<T>>;
type MatrixType<T, R, C> = Matrix<T, R, C, <DefaultAllocator as Allocator<R, C>>::Buffer<T>>;
