use eqsolver::{ODESolver, SolverError};
use nalgebra::{vector, Vector2};

#[test]
fn first_order_ode_solver() {
    let f = |t: f64, y: f64| t * y; // y' = f(t, y) = ty
    let (x0, y0) = (0., 0.2);
    let x_end = 2.;
    let step_size = 1e-3;
    const SOLUTION: f64 = 1.4778112197861300;

    let solver = ODESolver::new(f, x0, y0, step_size);
    let solution = solver.solve(x_end).unwrap();

    assert!((solution - SOLUTION).abs() < 1e-2);
    assert_eq!(solver.solve(-1.).unwrap_err(), SolverError::IncorrectInput); // Solvers only go forward
}

#[test]
fn ode_system_solver() {
    let f = |t: f64, y: Vector2<f64>| Vector2::new(y[1], t - y[0]); // y'' = f(t, y) = ty
    let (x0, y0) = (0., vector![1., 1.]);
    let x_end = 2.;
    let step_size = 1e-3;
    const SOLUTION: f64 = 1.5838531634528576;

    let solver = ODESolver::new(f, x0, y0, step_size);
    let solution = solver.solve(x_end).unwrap();

    assert!((solution[0] - SOLUTION).abs() < 1e-2);
}
