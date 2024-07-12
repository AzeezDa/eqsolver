use eqsolver::multivariable::{MultiVarNewton, MultiVarNewtonFD};
use nalgebra::{Matrix2, Vector2};

fn main() {
    // Want to solve x^2 - y = 1 and xy = 2. We do this by solving:
    // { x^2 - y - 1 = 0
    // { xy - 2      = 0
    let f = |v: Vector2<f64>| Vector2::new(v[0].powi(2) - v[1] - 1., v[0] * v[1] - 2.);

    // Jacobian of F
    let j = |v: Vector2<f64>| Matrix2::new(2. * v[0], -1., v[1], v[0]);

    let guess = Vector2::new(1., 1.); // Starting guess is (1, 1)

    // Newton's method requires the Jacobian matrix as input
    let solution_newton = MultiVarNewton::new(f, j).solve(guess).unwrap();

    // The finite difference method approximates the Jacobian matrix and thus is not required to be inputted
    let solution_newtonfd = MultiVarNewtonFD::new(f).solve(guess).unwrap();

    println!("Newton:   {solution_newton:?}");
    println!("NewtonFD: {solution_newtonfd:?}");
}
