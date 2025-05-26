use nalgebra::{allocator::Allocator, DefaultAllocator, Matrix, Vector};

/// Methods of derivative approximation
pub mod finite_differences;

/// Root-finders for equations of a single variable
pub mod single_variable;

/// Root-finders for equations of multiple variables
pub mod multivariable;

/// Finds global optimums of objective functions
pub mod global_optimisers;

mod ode_solver;
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

    /// An error from an external crate or library
    ExternalError,
}

/// Alias for `Result<T, SolverError>`
pub type SolverResult<T> = Result<T, SolverError>;

type VectorType<T, D> = Vector<T, D, <DefaultAllocator as Allocator<D>>::Buffer<T>>;
type MatrixType<T, R, C> = Matrix<T, R, C, <DefaultAllocator as Allocator<R, C>>::Buffer<T>>;
