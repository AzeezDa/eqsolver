use eqsolver::ODESolver;

fn main() {
    // Want to solve: y' = f(t, y) = t*y, starting at (0, 0.2) and ending at t = 2
    let f = |t: f64, y: f64| t * y;
    let (x0, y0) = (0., 0.2);
    let x_end = 2.;
    let step_size = 1e-3;

    let solver = ODESolver::new(f, x0, y0, step_size);
    let solution = solver.solve(x_end).unwrap();

    println!("Solution: {solution}");
}
