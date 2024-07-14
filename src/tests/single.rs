use crate::{finite_differences::{backward, central, forward}, single_variable::{FDNewton, Newton, Secant}};

#[test]
fn solve_secant() {
    let f = |x: f64| x * x - 2.;

    let solution = Secant::new(f).with_tol(1e-3).solve(0., 2.).unwrap();

    assert!((solution - 2_f64.sqrt()).abs() <= 1e-3);
    assert!((solution - 2_f64.sqrt()).abs() > 1e-12);
}

#[test]
fn solve_newton() {
    let f = |x: f64| x.sin() * 2. - x;
    let df = |x: f64| x.cos() * 2. - 1.;
    const SOLUTION: f64 = 1.8954942670339809471; // From Wolfram Alpha

    let solution = Newton::new(f, df).with_tol(1e-3).solve(2.).unwrap();

    assert!((solution - SOLUTION).abs() <= 1e-3);
    assert!((solution - SOLUTION).abs() > 1e-12);
}

#[test]
fn finite_differences() {
    let f = |x: f64| x.powi(3);
    let x = -1.;

    const SOLUTION: f64 = 3.;
    let dx_c = central(f, x, f64::EPSILON.sqrt());
    let dx_f = forward(f, x, f64::EPSILON);
    let dx_b = backward(f, x, f64::EPSILON);

    assert!((dx_c - SOLUTION).abs() <= f64::EPSILON);
    assert!((dx_f - SOLUTION).abs() <= f64::EPSILON);
    assert!((dx_b - SOLUTION).abs() <= f64::EPSILON);
}

#[test]
fn newton_with_finite_differences() {
    let f = |x: f64| x.exp().sin() / (1. + x * x) - (-x).exp();

    const SOLUTION: f64 = 0.1168941457861853920; // From Wolfram Alpha

    let solution = FDNewton::new(f).with_tol(1e-3).solve(0.).unwrap();

    assert!((solution - SOLUTION).abs() <= 1e-3);
    assert!((solution - SOLUTION).abs() > 1e-12);
}