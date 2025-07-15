use eqsolver::{
    integrators::{
        AdaptiveNewtonCotes, Formula, MonteCarlo, NewtonCotes, OrthotopeRandomIntegrator, MISER,
    },
    DEFAULT_TOL,
};
use nalgebra::{vector, Vector2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn gaussian(x: f64) -> f64 {
    (-x * x).exp()
}

fn sincos(v: Vector2<f64>) -> f64 {
    (v[1].cos() + v[0]).sin()
}

const APPROXIMATE_INTEGRAL_GAUSSIAN_0_TO_1: f64 = 0.746824132812427025;
const APPROXIMATE_INTEGRAL_SINX_ADD_COSY_0_TO_1: f64 = 0.9248464799519;
const CHACHA8_SEED: u64 = 1729;

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

#[test]
fn monte_carlo_1d() {
    let mut rng = ChaCha8Rng::seed_from_u64(CHACHA8_SEED);
    let result = MonteCarlo::new(gaussian)
        .integrate_with_rng(0., 1., &mut rng)
        .unwrap()
        .mean;

    assert!((result - APPROXIMATE_INTEGRAL_GAUSSIAN_0_TO_1).abs() <= 0.1)
}

#[test]
fn monte_carlo_2d() {
    let mut rng = ChaCha8Rng::seed_from_u64(CHACHA8_SEED);
    let result = MonteCarlo::new(sincos)
        .integrate_with_rng(vector![0., 0.], vector![1., 1.], &mut rng)
        .unwrap()
        .mean;

    assert!((result - APPROXIMATE_INTEGRAL_SINX_ADD_COSY_0_TO_1).abs() <= 0.1)
}

#[test]
fn miser_2d() {
    let mut rng = ChaCha8Rng::seed_from_u64(CHACHA8_SEED);
    let result = MISER::new(sincos)
        .integrate_with_rng(vector![0., 0.], vector![1., 1.], &mut rng)
        .unwrap();

    assert!((result.mean - APPROXIMATE_INTEGRAL_SINX_ADD_COSY_0_TO_1).abs() <= 0.001)
}

#[test]
fn general_monte_carlo() {
    use eqsolver::integrators::MonteCarlo;
    use nalgebra::{vector, Vector2};
    use rand_chacha::ChaCha8Rng; // or any rng you like
    use rand_distr::{Distribution, Uniform};
    use std::f64::consts::{FRAC_PI_2, PI};
    let mut rng = ChaCha8Rng::seed_from_u64(1729);
    let f = |v: Vector2<f64>| v.norm_squared();
    let unit_uniform = Uniform::new_inclusive(0., 1.).unwrap();
    let sampler = || {
        let radius = f64::sqrt(unit_uniform.sample(&mut rng));
        let angle = 2. * PI * unit_uniform.sample(&mut rng);
        radius * (vector![angle.cos(), angle.sin()])
    };
    let result = MonteCarlo::new(f)
        .integrate_with_sampler(sampler, PI)
        .unwrap();
    assert!((result.mean - FRAC_PI_2).abs() <= 0.1)
}
