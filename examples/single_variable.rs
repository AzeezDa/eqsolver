use eqsolver::single_variable::{FDNewton, Newton, Secant};

fn main() {
    let f = |x: f64| x.cos() - x.sin();
    let df = |x: f64| -x.sin() - x.cos(); // Derivative of f

    let solution_newton = Newton::new(f, df).solve(0.8).unwrap(); // Starting guess is 0.8

    // Finite difference Newton requires no derivative to be inputted (it approximates it)
    let solution_fdnewton = FDNewton::new(f).solve(0.8).unwrap(); // Starting guess is 0.8

    // Secant method requires no derivative too but needs two starting points
    let solution_secant = Secant::new(f).solve(0.5, 1.).unwrap();

    println!("Newton:   {solution_newton}");
    println!("FDNewton: {solution_fdnewton}");
    println!("Secant:   {solution_secant}");
}
