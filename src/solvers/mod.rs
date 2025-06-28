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
