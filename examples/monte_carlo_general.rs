use eqsolver::integrators::MonteCarlo;
use nalgebra::{vector, Vector2};
use rand::rng;
use rand_distr::{Distribution, Uniform};
use std::f64::consts::PI;

fn main() {
    // We integrate f(x, y) = x^2 + y^2 over the unit circle.
    let f = |v: Vector2<f64>| v.norm_squared();

    // We sample a random point in the unit circle: random angle in [0, 2π], and random radius in
    // [0, 1].
    let unit_uniform = Uniform::new_inclusive(0., 1.).unwrap();
    let sampler = || {
        let radius = f64::sqrt(unit_uniform.sample(&mut rng())); // sqrt for uniformity
        let angle = 2. * PI * unit_uniform.sample(&mut rng());
        radius * (vector![angle.cos(), angle.sin()])
    };

    let result = MonteCarlo::new(f)
        .integrate_with_sampler(sampler, PI) // π is the volume (area) of the unit circle
        .unwrap();

    println!("{result:?}");
}
