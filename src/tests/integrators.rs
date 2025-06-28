use crate::{
    integrators::{AdaptiveNewtonCotes, NewtonCotes, Formula},
    DEFAULT_TOL,
};

fn gaussian(x: f64) -> f64 {
    (-x * x).exp()
}
const APPROXIMATE_INTEGRAL_GAUSSIAN_0_TO_1: f64 = 0.746824132812427025;

#[test]
fn trapezium() {
    let result = NewtonCotes::new(gaussian)
        .with_formula(Formula::Trapezium)
        .with_subdivisions(1000)
        .integrate(0., 1.)
        .unwrap();

    assert!((result - APPROXIMATE_INTEGRAL_GAUSSIAN_0_TO_1).abs() <= 1e-3)
}

#[test]
fn simpsons_one_third() {
    let result = NewtonCotes::new(gaussian)
        .with_formula(Formula::SimpsonsOneThird)
        .with_subdivisions(1000)
        .integrate(0., 1.)
        .unwrap();

    assert!((result - APPROXIMATE_INTEGRAL_GAUSSIAN_0_TO_1).abs() <= 1e-3)
}

#[test]
fn simpsons_three_eighths() {
    let result = NewtonCotes::new(gaussian)
        .with_formula(Formula::SimpsonsThreeEighths)
        .with_subdivisions(1000)
        .integrate(0., 1.)
        .unwrap();

    assert!((result - APPROXIMATE_INTEGRAL_GAUSSIAN_0_TO_1).abs() <= 1e-3)
}

#[test]
fn adaptive_trapezium() {
    let result = AdaptiveNewtonCotes::new(gaussian)
        .with_formula(Formula::Trapezium)
        .with_tolerance(DEFAULT_TOL)
        .integrate(0., 1.);

    assert!((result.unwrap() - APPROXIMATE_INTEGRAL_GAUSSIAN_0_TO_1).abs() <= DEFAULT_TOL)
}

#[test]
fn adaptive_simpsons_one_third() {
    let result = AdaptiveNewtonCotes::new(gaussian)
        .with_formula(Formula::SimpsonsOneThird)
        .with_tolerance(DEFAULT_TOL)
        .integrate(0., 1.);

    assert!((result.unwrap() - APPROXIMATE_INTEGRAL_GAUSSIAN_0_TO_1).abs() <= DEFAULT_TOL)
}

#[test]
fn adaptive_simpsons_three_eighths() {
    let result = AdaptiveNewtonCotes::new(gaussian)
        .with_formula(Formula::SimpsonsThreeEighths)
        .with_tolerance(DEFAULT_TOL)
        .integrate(0., 1.);

    assert!((result.unwrap() - APPROXIMATE_INTEGRAL_GAUSSIAN_0_TO_1).abs() <= DEFAULT_TOL)
}
