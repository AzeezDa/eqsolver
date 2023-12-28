# `eqsolver` - An Equation Solver library for Rust

This is a Rust Library aimed at applying Numerical Methods to solve equations of many sorts.

Currently the library provides methods for solving single and multivariate functions as well as Ordinary Differential Equations (ODE) and systems of ODEs.

The library is **passively-maintained**, which means there will be no other features added. However, issues on the GitHub will be answered and solved.
Contributions and feedback to this library are more than welcome! 

## Single Variable

There are 2 types of root-finders for single variable functions, the Secant Method and Newton-Raphson's method, in which the latter can either be given the derivative of the function if available or the function can be approximated using finite differences.

### Example of Newton-Raphson's method.
```rust
use eqsolver::single_variable::Newton;
let f = |x: f64| x.cos() - x.sin();
let df = |x: f64| -x.sin() - x.cos(); // Derivative of f

let solution = Newton::new(f, df).solve(0.8); // Starting guess is 0.8
```

### Example of Newton-Raphson's method with finite differences.
```rust
use eqsolver::single_variable::FDNewton;

let f = |x: f64| x.exp() - 1./x; // e^x = 1/x
let solution = FDNewton::new(f).solve(0.5); // Starting guess is 0.5
```

## Multivariate Functions

There is one root-finder for multivariate functions and that is Newton-Raphson's method for systems. The library also provides a way solve non-linear square problems using the Gauss-Newton algorithm.

This part is heavily built using The Linear Algebra Library for Rust [nalgebra](https://nalgebra.org/).

### Example of Newton-Raphson's method for system of equations
```rust 
use eqsolver::multivariable::MultiVarNewton;
use nalgebra::{Vector2, Matrix2};
// Want to solve x^2 - y = 1 and xy = 2
let F = |v: Vector2<f64>| Vector2::new(v[0].powi(2) - v[1] - 1., v[0] * v[1] - 2.);
 
// Jacobian of F
let J = |v: Vector2<f64>| Matrix2::new(2. * v[0], -1., 
                                            v[1], v[0]);

let solution = MultiVarNewton::new(F, J).solve(Vector2::new(1., 1.)); // Starting guess is (1, 1)
```


### Example of Newton-Raphson's method with finite differences for system of equations
```rust
use eqsolver::multivariable::MultiVarNewtonFD;
use nalgebra::{Vector2, Matrix2};
// Want to solve x^2 - y = 1 and xy = 2
let F = |v: Vector2<f64>| Vector2::new(v[0].powi(2) - v[1] - 1., v[0] * v[1] - 2.);

let solution = MultiVarNewtonFD::new(F).solve(Vector2::new(1., 1.)); // Starting guess is (1, 1)
```

## Ordinary Differential Equations (or systems of them)
The library provides a single `struct` for solving ODEs which can be modified to to use Euler Forward, Heun's Method or Runge-Kutta 4.

Example of solution for a single first order ODEs
```rust
let f = |t: f64, y: f64| t * y; // y' = f(t, y) = ty
let (x0, y0) = (0., 0.2);
let x_end = 2.;
let step_size = 1e-3;

let solver = ODESolver::new(f, x0, y0, step_size);
let solution = solver.solve(x_end).unwrap();
```

Example of system of first order ODEs
```rust
let f = |t: f64, y: Vector2<f64>| Vector2::new(y[1], t - y[0]); // {v1 = y'  = y[1]
                                                                // {v2 = t-y = t-y[0]
let (x0, y0) = (0., vector![1., 1.]);
let x_end = 2.;
let step_size = 1e-3;

let solver = ODESolver::new(f, x0, y0, step_size);
let solution = solver.solve(x_end).unwrap();
```

Note there is no explicit solver for higher order ODEs since higher order ODEs can be rewritten as a system of first order ODEs.
The example above is equivalent to solving `y'' = t-y`.

## Mutable Functions
Sometimes the function to be solved requires mutation of its environment, for example saving the values that function was called with, or for memoization. This mutation behavior can be done using a `std::cell::RefCell` as in this example:

```rust
use std::cell::RefCell;
let trace = RefCell::new(vec![]);
let f = |x: f64| {
    trace.borrow_mut().push(x);
    x * x - 2.
};

let solution = Secant::new(f).with_tol(1e-3).solve(0., 2.).unwrap();

// At this point the trace vector will be: 
// [0.0, 2.0, 1.0, 1.3333333333333333, 1.4285714285714286, 1.4137931034482758, 1.41421143847487]
```

See [this issue](https://github.com/AzeezDa/eqsolver/issues/5) for the reasoning behind this technicality.