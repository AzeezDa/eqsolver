mod secant;
mod newton;
mod fdnewton;
mod multinewton;
mod multinewton_fd;
mod gaussnewton;
mod gaussnewton_fd;
mod ode_solver;

/// Methods of derivative approximation
pub mod finite_differences;

/// Root-finders for equations of a single variable
pub mod single_variable {
    pub use super::{
        secant::Secant,
        newton::Newton,
        fdnewton::FDNewton,
    };
}

/// Root-finders for equations of multiple variables
pub mod multivariable {
    pub use super::{
        multinewton::MultiVarNewton,
        multinewton_fd::MultiVarNewtonFD,
        gaussnewton::GaussNewton,
        gaussnewton_fd::GaussNewtonFD
    };
}

pub use ode_solver::ODESolver;

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

/// Types of methods for ODE solving
pub enum ODESolverMethod {
    /// The Explicit Euler method, ([Wikipedia](https://en.wikipedia.org/wiki/Euler_method))
    /// 
    /// Order of accuracy: 1
    EulerForward,

    /// Heun's Method (aka Runge-Kutta 2), ([Wikipedia](https://en.wikipedia.org/wiki/Heun%27s_method))
    /// 
    /// Order of accuracy: 2
    Heun,

    /// Runge-Kutta 4, ([Wikipedia](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods))
    /// 
    /// Order of accuracy: 4
    RungeKutta4
}

type SolverResult<T> = Result<T, SolverError>;