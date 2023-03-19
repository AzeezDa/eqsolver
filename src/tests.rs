use crate::finite_differences::{backward, central, forward};
use crate::{multivariable::*, single_variable::*, ODESolver, SolverError};
use nalgebra::{vector, Matrix2, Matrix3x2, Vector2, Vector3, DVector, DMatrix};

#[test]
fn solve_secant() {
    let f = |x: f64| x * x - 2.;

    let solution = Secant::new(f).with_tol(1e-3).solve(0., 2.).ok().unwrap();

    assert!((solution - 2_f64.sqrt()).abs() <= 1e-3);
    assert!((solution - 2_f64.sqrt()).abs() > 1e-12);
}

#[test]
fn solve_newton() {
    let f = |x: f64| x.sin() * 2. - x;
    let df = |x: f64| x.cos() * 2. - 1.;
    const SOLUTION: f64 = 1.8954942670339809471; // From Wolfram Alpha

    let solution = Newton::new(f, df).with_tol(1e-3).solve(2.).ok().unwrap();

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

#[test]
fn multi_var_newton() {
    // Vectorial Function (x, y) ↦ (x^2-y-1, xy - 2)
    let f = |v: Vector2<f64>| Vector2::new(v[0].powi(2) - v[1] - 1., v[0] * v[1] - 2.);

    // Jacobian of F
    let j = |v: Vector2<f64>| Matrix2::new(2. * v[0], -1., v[1], v[0]);

    // Solved analytically but form was ugly so here is approximation
    const SOLUTION: Vector2<f64> =
        Vector2::new(1.521379706804567569604081, 1.314596212276751981650111);

    // Use struct generated above to solve the problem defined in f
    let solution = MultiVarNewton::new(f, j)
        .with_tol(1e-3)
        .with_itermax(50)
        .solve(Vector2::repeat(1.))
        .unwrap();

    // Use struct generated above to solve the problem defined in f
    let solution_fd = MultiVarNewtonFD::new(f)
        .with_tol(1e-3)
        .with_itermax(50)
        .solve(Vector2::repeat(1.))
        .unwrap();

    assert!((SOLUTION - solution).norm() <= 1e-3);
    assert!((SOLUTION - solution).norm() > 1e-12);
    assert!((SOLUTION - solution_fd).norm() <= 1e-3);
    assert!((SOLUTION - solution_fd).norm() > 1e-12);
}

#[test]
fn gauss_newton() {
    // Test is about finding point closest to three circles in R2

    // [Center_x, Center_y, Radius]
    let c0 = [3., 5., 3.];
    let c1 = [1., 0., 4.];
    let c2 = [6., 2., 2.];

    // Want to (x, y) such that F(x, y) = (x - X)^2 + (y - Y) - R^2 is minimized in a Least Square sense for data in c0, c1, c2
    let f = |v: Vector2<f64>| {
        Vector3::new(
            (v[0] - c0[0]).powi(2) + (v[1] - c0[1]).powi(2) - c0[2] * c0[2],
            (v[0] - c1[0]).powi(2) + (v[1] - c1[1]).powi(2) - c1[2] * c1[2],
            (v[0] - c2[0]).powi(2) + (v[1] - c2[1]).powi(2) - c2[2] * c2[2],
        )
    };

    let j = |v: Vector2<f64>| {
        Matrix3x2::new(
            2. * (v[0] - c0[0]),
            2. * (v[1] - c0[1]),
            2. * (v[0] - c1[0]),
            2. * (v[1] - c1[1]),
            2. * (v[0] - c2[0]),
            2. * (v[1] - c2[1]),
        )
    };

    // Solved using Octave (Can also be checked visually in Desmos or similar)
    const SOLUTION: Vector2<f64> = vector![4.217265312839526, 2.317879970005811];

    let solution_gn = GaussNewton::new(f, j)
        .with_tol(1e-3)
        .solve(vector![4.5, 2.5])
        .unwrap();

    let solution_gn_fd = GaussNewtonFD::new(f)
        .with_tol(1e-3)
        .solve(vector![4.5, 2.5])
        .unwrap();

    assert!((SOLUTION - solution_gn).norm() <= 1e-3);
    assert!((SOLUTION - solution_gn).norm() > 1e-12);
    assert!((SOLUTION - solution_gn_fd).norm() <= 1e-3);
    assert!((SOLUTION - solution_gn_fd).norm() > 1e-12);
}


#[test]
fn dyn_multi_var_newton() {
    // Vectorial Function (x, y) ↦ (x^2-y-1, xy - 2)
    let f = |v: DVector<f64>| DVector::from_vec(vec![v[0].powi(2) - v[1] - 1., v[0] * v[1] - 2.]);

    // Jacobian of F
    let j = |v: DVector<f64>| DMatrix::from_vec(2, 2, vec![2. * v[0], v[1], -1., v[0]]);

    // Solved analytically but form was ugly so here is approximation
    const SOLUTION: Vector2<f64> =
        Vector2::new(1.521379706804567569604081, 1.314596212276751981650111);

    // Use struct generated above to solve the problem defined in f
    let solution = MultiVarNewton::new(f, j)
        .with_tol(1e-3)
        .with_itermax(50)
        .solve(DVector::repeat(2, 1.))
        .unwrap();

    // Use struct generated above to solve the problem defined in f
    let solution_fd = MultiVarNewtonFD::new(f)
        .with_tol(1e-3)
        .with_itermax(50)
        .solve(DVector::repeat(2, 1.))
        .unwrap();

    assert!((SOLUTION - &solution).norm() <= 1e-3);
    assert!((SOLUTION - &solution).norm() > 1e-12);
    assert!((SOLUTION - &solution_fd).norm() <= 1e-3);
    assert!((SOLUTION - &solution_fd).norm() > 1e-12);
}

#[test]
fn dyn_gauss_newton() {
    // Test is about finding point closest to three circles in R2

    // [Center_x, Center_y, Radius]
    let c0 = [3., 5., 3.];
    let c1 = [1., 0., 4.];
    let c2 = [6., 2., 2.];

    // Want to (x, y) such that F(x, y) = (x - X)^2 + (y - Y) - R^2 is minimized in a Least Square sense for data in c0, c1, c2
    let f = |v: DVector<f64>| {
        DVector::from_vec(vec![
            (v[0] - c0[0]).powi(2) + (v[1] - c0[1]).powi(2) - c0[2] * c0[2],
            (v[0] - c1[0]).powi(2) + (v[1] - c1[1]).powi(2) - c1[2] * c1[2],
            (v[0] - c2[0]).powi(2) + (v[1] - c2[1]).powi(2) - c2[2] * c2[2],
        ])
    };

    let j = |v: DVector<f64>| {
        DMatrix::from_vec(3, 2, vec![
            2. * (v[0] - c0[0]), 2. * (v[0] - c1[0]), 2. * (v[0] - c2[0]),
            2. * (v[1] - c0[1]), 2. * (v[1] - c1[1]), 2. * (v[1] - c2[1]),
        ])
    };

    // Solved using Octave (Can also be checked visually in Desmos or similar)
    const SOLUTION: Vector2<f64> = vector![4.217265312839526, 2.317879970005811];

    let solution_gn = GaussNewton::new(f, j)
        .with_tol(1e-3)
        .solve(DVector::from_vec(vec![4.5, 2.5]))
        .unwrap();

    let solution_gn_fd = GaussNewtonFD::new(f)
        .with_tol(1e-3)
        .solve(DVector::from_vec(vec![4.5, 2.5]))
        .unwrap();

    assert!((SOLUTION - &solution_gn).norm() <= 1e-3);
    assert!((SOLUTION - &solution_gn).norm() > 1e-12);
    assert!((SOLUTION - &solution_gn_fd).norm() <= 1e-3);
    assert!((SOLUTION - &solution_gn_fd).norm() > 1e-12);
}

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
