use eqsolver::integrators::{AdaptiveNewtonCotes, NewtonCotes};

fn main() {
    // We will integrate the function e^(x^2) from 0 to 1
    let f = |x: f64| (x * x).exp();

    let newton_cotes_result = NewtonCotes::new(f).integrate(0., 1.).unwrap();

    // The adaptive Newton-Cotes allows for setting the tolerance
    let adaptive_newton_cotes_result = AdaptiveNewtonCotes::new(f)
        .with_tolerance(0.001)
        .integrate(0., 1.)
        .unwrap();

    println!("Newton-Cotes:          {}", newton_cotes_result);
    println!("Adaptive Newton-Cotes: {}", adaptive_newton_cotes_result);
}
