use nalgebra::{vector, DMatrix, DVector, Matrix2, Matrix3x2, Vector2, Vector3};

use crate::multivariable::{
    GaussNewton, GaussNewtonFD, LevenbergMarquardt, LevenbergMarquardtFD, MultiVarNewton,
    MultiVarNewtonFD,
};

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
