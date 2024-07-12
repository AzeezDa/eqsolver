use nalgebra::{vector, Vector1};

use crate::{
    multivariable::{GaussNewton, GaussNewtonFD, MultiVarNewton, MultiVarNewtonFD},
    single_variable::{FDNewton, Newton, Secant},
    SolverError,
};

#[test]
fn mutable_state_function() {
    use std::cell::RefCell;
    let trace = RefCell::new(vec![]);
    let f = |x: f64| {
        trace.borrow_mut().push(x);
        x * x - 2.
    };

    let solution = Secant::new(f).with_tol(1e-3).solve(0., 2.).unwrap();
    assert!((solution - 2_f64.sqrt()).abs() <= 1e-3);
    assert!(trace.borrow().len() > 0);
}

#[test]
fn max_iter_reached_detection() {
    // "f(x)=x^2 + 1" has no roots, so solver can't ever converge
    let f = |v: Vector1<f64>| Vector1::new(v[0].powi(2) + 1.0);
    let j = |v: Vector1<f64>| Vector1::new(2. * v[0]);

    let solution = MultiVarNewton::new(f, j).with_tol(1e-3).solve(vector![3.0]);
    assert_eq!(solution, Err(SolverError::MaxIterReached));

    let solution = MultiVarNewtonFD::new(f).with_tol(1e-3).solve(vector![3.0]);
    assert_eq!(solution, Err(SolverError::MaxIterReached));

    let solution = GaussNewton::new(f, j).with_tol(1e-3).solve(vector![3.0]);
    assert_eq!(solution, Err(SolverError::MaxIterReached));

    let solution = GaussNewtonFD::new(f).with_tol(1e-3).solve(vector![3.0]);
    assert_eq!(solution, Err(SolverError::MaxIterReached));

    // Single-variable solvers
    let f = |v: f64| v.powi(2) + 1.;
    let j = |v: f64| 2. * v;

    let solution = Newton::new(f, j).with_tol(1e-3).solve(3.);
    assert_eq!(solution, Err(SolverError::MaxIterReached));

    let solution = FDNewton::new(f).with_tol(1e-3).solve(3.);
    assert_eq!(solution, Err(SolverError::MaxIterReached));

    let solution = Secant::new(f).with_tol(1e-3).solve(3., 4.);
    assert_eq!(solution, Err(SolverError::MaxIterReached));
}
