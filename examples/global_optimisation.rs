use eqsolver::global_optimisers::{CrossEntropy, ParticleSwarm};
use nalgebra::{vector, SVector};
use std::f64::consts::PI;

fn main() {
    // This is the Rastrigin function, an objective function with many local
    // minima and a global minima at the zero vector
    const SIZE: usize = 10;
    let rastrigin = |v: SVector<f64, SIZE>| {
        let mut total = 10. * SIZE as f64;

        for &w in v.iter() {
            total += w * w - 10. * (2. * PI * w).cos();
        }

        total
    };

    let guess = vector![80., -81., 82., -83., 84., -85., 86., -87., 88., -89.]; // Guess way off the global minimum!

    // Particle Swarm Optimisation (pso) requires bounds
    let lower_bounds = SVector::repeat(-100.);
    let upper_bounds = SVector::repeat(100.);
    let solution_pso = ParticleSwarm::new(rastrigin, lower_bounds, upper_bounds)
        .unwrap()
        .solve(guess)
        .unwrap();

    // Cross-Entropy (ce) method uses a vector of standard deviations (how
    // uncertain the guess is). The default is a vector of 1s
    let standard_deviations = SVector::repeat(100.);
    let solution_ce = CrossEntropy::new(rastrigin)
        .with_std_dev(standard_deviations)
        .solve(guess)
        .unwrap();

    println!("Guess:    {guess:?}");
    println!("f(Guess): {}\n", rastrigin(guess));

    println!("PSO:      {solution_pso:?}");
    println!("f(PSO):   {}\n", rastrigin(solution_pso));

    println!("CE:       {solution_ce:?}");
    println!("f(CE):    {}\n", rastrigin(solution_ce));
}
