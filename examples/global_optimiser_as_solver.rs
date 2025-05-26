use eqsolver::global_optimisers::{CrossEntropy, ParticleSwarm};
use nalgebra::{vector, Vector2, Vector3};

fn main() {
    /*
    This example is about finding the point in R2 closest to three circles (in
    R2), which is the similar to the example in examples/nxm_equation_system.rs

    Here, it is however done using the global optimisers by converting the
    equation system problem into a global optimisation problem. More
    generally, say we want to solve F(x) = b, for x in Rm and b in Rn.  We
    can use the Levenberg-Marquardt or Gauss-Newton methods or we can use
    global optimisers.  For the latter we, create a new function G(x) = ||F(x)
    - b||, i.e. the norm of F(x) - b. The range of G is one dimensional and
    its domain is in Rn; it this thus an objective function whose global
    minimum X gives zero objective function value if it is a solution to the
    equation F(x) = b.  If there are no solutions, the closest x that
    minimises the norm will be calculated.
    */

    // [Center_x, Center_y, Radius] of the circles
    let c0 = [3., 5., 3.];
    let c1 = [1., 0., 4.];
    let c2 = [6., 2., 2.];

    // F(x, y) = (x - Xcenter)^2 + (y - Ycenter)^2
    let f = |v: Vector2<f64>| {
        Vector3::new(
            (v[0] - c0[0]).powi(2) + (v[1] - c0[1]).powi(2),
            (v[0] - c1[0]).powi(2) + (v[1] - c1[1]).powi(2),
            (v[0] - c2[0]).powi(2) + (v[1] - c2[1]).powi(2),
        )
    };

    let b = vector![c0[2].powi(2), c1[2].powi(2), c2[2].powi(2)];

    // G(x, y) = ||F(x, y) - b||, where b is the vector of radii squared.
    let f_normed = |v: Vector2<f64>| (f(v) - b).norm();

    let guess = vector![4.5, 2.5];

    let lower_bounds = vector![1., 1.];
    let upper_bounds = vector![7., 7.];
    let solution_pso = ParticleSwarm::new(f_normed, lower_bounds, upper_bounds)
        .unwrap()
        .solve(guess)
        .unwrap();

    let solution_ce = CrossEntropy::new(f_normed).solve(guess).unwrap();

    println!("Particle Swarm:        {solution_pso:?}");
    println!("Cross-Entropy:         {solution_ce:?}");
}
