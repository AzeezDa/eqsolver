use eqsolver::ODESolver;
use nalgebra::{vector, Vector2};

fn main() {
    // We want to solve y'' = t-y'. This can be written as the system:
    // {v1 = y'  = y[1]
    // {v2 = t-y = t-y[0]

    let f = |t: f64, y: Vector2<f64>| Vector2::new(y[1], t - y[0]);
    let (x0, y0) = (0., vector![1., 1.]);
    let x_end = 2.;
    let step_size = 1e-3;

    let solver = ODESolver::new(f, x0, y0, step_size);
    let solution = solver.solve(x_end).unwrap();

    println!("Solution: {solution}");
}
