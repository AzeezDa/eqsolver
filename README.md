# `eqsolver` - An Equation Solver and Optimisation Library for Rust

This Rust library is aimed at numerically solving equations and optimising objective functions.

The library is **passively-maintained**, meaning no other features will be added. However, issues on the GitHub will be answered and solved.

Contributions and feedback to this library are more than welcome! 

## Supported Methods
The following methods are available to use in the library. Their descriptions use the largest possible domain and codomain for the functions, which is Rn. However, any (well-behaved) subset of Rn also works. Additionally, the methods that use multivariate input or output heavily utilises the linear algebra library for Rust [nalgebra](https://nalgebra.org/).

### Single Variable
<details>
<summary>Newton-Raphson's Method</summary>
Finds a root of a univariate function f(x) given its derivative Df(x) and an initial guess. This method has a quadratic rate of convergence.
</details>

<details>
<summary>Newton-Raphson's Method with Finite Differences</summary>
Finds a root of a univariate function f(x) by approximating its derivative Df(x) using finite differences, given an initial guess of the root. This method has a quadratic rate of convergence but requires a little more computation than the non-finite-difference version, making the wall time slightly longer.
</details>

<details>
<summary>Secant Method</summary>
Finds a root of a univariate function f(x) given two unique starting values. This method has a slightly lower rate of convergence (equal to the golden ratio) but only does one function call per iteration making its wall time sometimes lower than the Newton-Raphson methods.
</details>

### Multivariate

<details>
<summary>Newton-Raphson's Method (with and without Finite Differences)</summary>
For a function F: Rn → Rn, this method finds x such that F(x) is the zero vector, which is equivalent to solving a system of n equations with n unknowns.

There are two versions of this method, one requires the Jacobian matrix to be given and the other approximates it using finite differences. The latter version has, therefore, slightly longer wall time. Both methods require an initial guess.

For certain ill-posed problems, this method will fail. For a slower but more robust method, see the Levenberg-Marquardt method below.
</details>

<details>
<summary>Gauss-Newton's Method (with and without Finite Differences)</summary>
For a function F: Rm → Rn, this method finds x such that F(x) is the zero vector, which is equivalent to solving a system of n equations with m unknowns. This is done by solving a least-square problem in each iteration which makes this method's wall time slightly longer than Newton-Raphson's method.

There are two versions of this method, one requires the Jacobian matrix to be given and the other approximates it using finite differences. The latter version has, therefore, slightly longer wall time. Both methods require an initial guess.

For certain ill-posed problems, this method will fail. For a slower but more robust method, see the Levenberg-Marquardt method below.
</details>

<details>
<summary>Levenberg-Marquardt's Method (with and without Finite Differences)</summary>
For a function F: Rm → Rn, this method finds x such that F(x) is the zero vector, which is equivalent to solving a system of n equations with m unknowns. This is done by solving a dampened least-square problem (more computation than the usual least-square problem) in each iteration which makes this method's wall time slightly longer than Gauss-Newton's method.

There are two versions of this method, one requires the Jacobian matrix to be given and the other approximates it using finite differences. The latter version has, therefore, slightly longer wall time. Both methods require an initial guess.
</details>

### Global Optimisers of Objective Functions
<details>
<summary>Particle Swarm Optimisation</summary>
For a function F: Rn → R, this method finds x such that F(x) <= F(y) for all y, i.e. the global minimum. This method requires an initial guess and bounds for which the global minimum exists.

Use this method if you know the bounds of the parameters.
</details>

<details>
<summary>Cross-Entropy Method</summary>
For a function F: Rn → R, this method finds x such that F(x) <= F(y) for all y, i.e. the global minimum. This method requires an initial guess and a Rn vector of standard deviations (uncertainty of each parameter).

Use this method if you DON'T KNOW the bounds of the parameters but KNOW how uncertain each parameter is.
</details>

### Ordinary Differential Equations (or systems of them)
There is a single `struct` for ordinary differential equations (ODE) which can be modified (using the builder pattern) to use one of the following step methods:
<details>
<summary>Euler Forward</summary>
This method requires one call to the function corresponding to the equation and is thus fast. It has, however, an order of accuracy of 1 and is unstable for certain functions.
</details>

<details>
<summary>Heun's Method (Runge-Kutta 2)</summary>
This method requires two calls to the function corresponding to the equation and is thus slower than Euler Forward. This method has an order of accuracy of 2.
</details>

<details>
<summary>Runge-Kutta 4 (Default)</summary>
This method requires four calls to the function corresponding to the equation and is thus slower than Heun's Method. This method has an order of accuracy of 4. The ODE solver uses this method as the default.
</details>

## Examples
### Example of Newton-Raphson's method with finite differences.
```rust
use eqsolver::single_variable::FDNewton;

let f = |x: f64| x.exp() - 1./x; // e^x = 1/x
let solution = FDNewton::new(f).solve(0.5); // Starting guess is 0.5
```

### Example of Newton-Raphson's method with finite differences for system of equations
```rust
use eqsolver::multivariable::MultiVarNewtonFD;
use nalgebra::{vector, Vector2};

// Want to solve x^2 - y = 1 and xy = 2
let f = |v: Vector2<f64>| vector![v[0].powi(2) - v[1] - 1., v[0] * v[1] - 2.];

let solution = MultiVarNewtonFD::new(f).solve(vector![1., 1.]); // Starting guess is (1, 1)
```

### Example of solution for a single first order ODEs
```rust
use eqsolver::ODESolver;

let f = |t: f64, y: f64| t * y; // y' = f(t, y) = ty
let (x0, y0) = (0., 0.2);
let x_end = 2.;
let step_size = 1e-3;

let solution = ODESolver::new(f, x0, y0, step_size).solve(x_end);
```

### Example of solving a non-linear least square problem with the Levenberg-Marquardt method
```rust
use eqsolver::multivariable::LevenbergMarquardtFD;
use nalgebra::{vector, Vector2};

let c0 = [3., 5., 3.];
let c1 = [1., 0., 4.];
let c2 = [6., 2., 2.];

// Function from R2 to R3
let f = |v: Vector2<f64>| {
    vector!(
        (v[0] - c0[0]).powi(2) + (v[1] - c0[1]).powi(2) - c0[2] * c0[2],
        (v[0] - c1[0]).powi(2) + (v[1] - c1[1]).powi(2) - c1[2] * c1[2],
        (v[0] - c2[0]).powi(2) + (v[1] - c2[1]).powi(2) - c2[2] * c2[2],
    )
};

let solution_lm = LevenbergMarquardtFD::new(f)
    .solve(vector![4.5, 2.5]) // Guess
    .unwrap();
```

### Example of using global optimisers on the Rastrigin function
```rust
use eqsolver::global_optimisers::{CrossEntropy, ParticleSwarm};
use nalgebra::SVector;
use std::f64::consts::PI;

const SIZE: usize = 10;
let rastrigin = |v: SVector<f64, SIZE>| {
    v.fold(10. * SIZE as f64, |acc, x| {
        acc + x * x - 10. * f64::cos(2. * PI * x)
    })
};

let bounds = SVector::repeat(10.);
let standard_deviations = SVector::repeat(10.);
let guess = SVector::repeat(5.);

let opt_pso = ParticleSwarm::new(rastrigin, -bounds, bounds).solve(guess);
let opt_ce = CrossEntropy::new(rastrigin)
    .with_std_dev(standard_deviations)
    .solve(guess);
```
For more examples, please see the [examples](examples) directory.