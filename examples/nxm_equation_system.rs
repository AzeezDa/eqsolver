use eqsolver::multivariable::{
    GaussNewton, GaussNewtonFD, LevenbergMarquardt, LevenbergMarquardtFD,
};
use nalgebra::{vector, Matrix3x2, Vector2, Vector3};

fn main() {
    // This example is about finding the point in R2 closest to three circles (in R2)

    // [Center_x, Center_y, Radius] of the circles
    let c0 = [3., 5., 3.];
    let c1 = [1., 0., 4.];
    let c2 = [6., 2., 2.];

    // We want (x, y) such that F(x, y) = (x - X)^2 + (y - Y) - R^2 is minimized in a
    // least-square sense for data (X, Y, R) in c0, c1, c2
    let f = |v: Vector2<f64>| {
        Vector3::new(
            (v[0] - c0[0]).powi(2) + (v[1] - c0[1]).powi(2) - c0[2] * c0[2],
            (v[0] - c1[0]).powi(2) + (v[1] - c1[1]).powi(2) - c1[2] * c1[2],
            (v[0] - c2[0]).powi(2) + (v[1] - c2[1]).powi(2) - c2[2] * c2[2],
        )
    };

    // Jacobian matrix of f
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

    let guess = vector![4.5, 2.5];

    let solution_gn = GaussNewton::new(f, j).solve(guess).unwrap();
    let solution_gn_fd = GaussNewtonFD::new(f).solve(guess).unwrap();
    let solution_lm = LevenbergMarquardt::new(f, j).solve(guess).unwrap();
    let solution_lm_fd = LevenbergMarquardtFD::new(f).solve(guess).unwrap();

    // The solutions match up to at least 6 decimals since the default tolerance is 1e-6
    println!("Gauss-Newton:          {solution_gn:?}");
    println!("Gauss-NewtonFD:        {solution_gn_fd:?}");
    println!("Levenberg-Marquardt:   {solution_lm:?}");
    println!("Levenberg-MarquardtFD: {solution_lm_fd:?}");
}
