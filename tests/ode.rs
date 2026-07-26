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

    assert!((solution - SOLUTION).abs() < 1e-6);
    assert!(match solver.solve(-1.).unwrap_err() {
        SolverError::IncorrectInput { details: _ } => true,
        _ => false,
    }); // Solvers only go forward
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

    assert!((solution[0] - SOLUTION).abs() < 1e-6);
}

#[test]
fn rk4_order_of_accuracy() {
    // RK4 is 4th order: halving the step size must cut the error by roughly 2^4 = 16.
    // The off-by-one in solve dropped the final step, degrading this to 1st order (#15).
    let f = |_t: f64, y: f64| -y;
    let (x0, y0, x_end) = (0., 1., 1.);
    let exact = (-1.0_f64).exp();

    let err = |steps| {
        (ODESolver::new(f, x0, y0, 1.)
            .with_steps(x_end, steps)
            .solve(x_end)
            .unwrap()
            - exact)
            .abs()
    };
    let ratio = err(50) / err(100);

    assert!(ratio > 8., "expected ~16 for 4th-order RK4, got {ratio}");
}

#[test]
fn solve_returns_value_at_x_end() {
    // solve must return y(x_end); the off-by-one returned y(x_end - h).
    let f = |_t: f64, y: f64| -y;
    let solution = ODESolver::new(f, 0., 1., 1e-3).solve(1.).unwrap();
    assert!((solution - (-1.0_f64).exp()).abs() < 1e-8);
}

#[test]
fn solve_reaches_x_end_for_indivisible_step() {
    // A step size that does not divide the interval must still land on x_end,
    // not stop a whole step short.
    let f = |_t: f64, y: f64| -y;
    let solution = ODESolver::new(f, 0., 1., 0.3).solve(1.).unwrap();
    assert!((solution - (-1.0_f64).exp()).abs() < 1e-3);
}
