use crate::{
    finite_differences::{backward, central, forward},
    global_optimisers::*,
    multivariable::*,
    single_variable::*,
    ODESolver, SolverError,
};
use nalgebra::{vector, DMatrix, DVector, Matrix2, Matrix3x2, Vector1, Vector2, Vector3};
use rand::thread_rng;
use rand_distr::Distribution;

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
fn levenberg_marquardt() {
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

    let solution_lm = LevenbergMarquardt::new(f, j)
        .with_tol(1e-3)
        .solve(vector![4.5, 2.5])
        .unwrap();

    let solution_lm_fd = LevenbergMarquardtFD::new(f)
        .with_tol(1e-3)
        .solve(vector![4.5, 2.5])
        .unwrap();

    assert!((SOLUTION - solution_lm).norm() <= 1e-3);
    assert!((SOLUTION - solution_lm).norm() > 1e-12);
    assert!((SOLUTION - solution_lm_fd).norm() <= 1e-3);
    assert!((SOLUTION - solution_lm_fd).norm() > 1e-12);
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
        DMatrix::from_vec(
            3,
            2,
            vec![
                2. * (v[0] - c0[0]),
                2. * (v[0] - c1[0]),
                2. * (v[0] - c2[0]),
                2. * (v[1] - c0[1]),
                2. * (v[1] - c1[1]),
                2. * (v[1] - c2[1]),
            ],
        )
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

/// Not really a test since Particle Swarm Optimisation is random. This test, therefore,
/// acts solely as documentation. The only thing assured is that the output's cost is not
/// more than the initial cost.
#[test]
fn particle_swarm() {
    const SIZE: usize = 16;
    let rastrigin = |v: nalgebra::SVector<f64, SIZE>| {
        let mut total = 10. * SIZE as f64;

        for &w in v.iter() {
            total += w * w - 10. * (2. * std::f64::consts::PI * w).cos();
        }

        total
    };

    let mut guess = nalgebra::SVector::zeros();
    for i in 0..SIZE {
        guess[i] = rand::distributions::Uniform::new(-50., 50.).sample(&mut thread_rng());
    }
    let cost = rastrigin(guess);
    let bounds = nalgebra::SVector::repeat(100.);

    let optimised_position = ParticleSwarm::new(rastrigin, -bounds, bounds)
        .with_tolerance(1e-3)
        .with_particle_count(1500)
        .solve(guess)
        .unwrap();
    let optimised_cost = rastrigin(optimised_position);

    // Run `cargo test particle_swarm -- --nocapture` to see this output and the improvement
    println!("{cost} -> {optimised_cost}\n{optimised_position}");

    assert!(optimised_cost <= cost)
}

/// Not really a test since Cross Entropy uses randomness. This test, therefore,
/// acts solely as documentation.
#[test]
fn cross_entropy() {
    const SIZE: usize = 16;
    let rastrigin = |v: nalgebra::SVector<f64, SIZE>| {
        let mut total = 10. * SIZE as f64;

        for &w in v.iter() {
            total += w * w - 10. * (2. * std::f64::consts::PI * w).cos();
        }

        total
    };

    let mut guess = nalgebra::SVector::zeros();
    let variance = nalgebra::SVector::repeat(100.);
    for i in 0..SIZE {
        guess[i] = rand::distributions::Uniform::new(-50., 50.).sample(&mut thread_rng());
    }
    let cost = rastrigin(guess);

    let optimised_position = CrossEntropy::new(rastrigin)
        .with_iter_max(120)
        .with_sample_size(200)
        .with_importance_selection_size(5)
        .with_tolerance(1e-12)
        .with_std_dev(variance)
        .solve(guess)
        .unwrap();
    let optimised_cost = rastrigin(optimised_position);

    // Run `cargo test particle_swarm -- --nocapture` to see this output and the improvement
    println!("{cost} -> {optimised_cost}\n{optimised_position}");

    assert!(optimised_cost <= cost)
}
