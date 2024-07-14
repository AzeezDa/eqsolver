use crate::{SolverError, SolverResult, DEFAULT_ITERMAX, DEFAULT_TOL};
use num_traits::Float;
use std::ops::Fn;

/// # Newton-Raphson
///
/// Newton solves an equation `f(x) = 0` given the function `f` and its derivative `df` as closures that takes a `Float` and outputs a `Float`.
/// This function uses the Newton-Raphson's method ([Wikipedia](https://en.wikipedia.org/wiki/Newton%27s_method)).
///
/// **Default Tolerance:** `1e-6`
///
/// **Default Max Iterations:** `50`
///
/// ## Examples
///
/// ### A solution exists
///
/// ```
/// // Want to solve x in cos(x) = sin(x). This is equivalent to solving x in cos(x) - sin(x) = 0.
/// use eqsolver::single_variable::Newton;
/// let f = |x: f64| x.cos() - x.sin();
/// let df = |x: f64| -x.sin() - x.cos(); // Derivative of f
///
/// // Solve with Newton's Method. Error is less than 1E-6. Starting guess is around 0.8.
/// let solution = Newton::new(f, df)
///     .with_tol(1e-6)
///     .solve(0.8)
///     .unwrap();
/// assert!((solution - std::f64::consts::FRAC_PI_4).abs() <= 1e-6); // Exactly x = pi/4
/// ```
///
/// ### A solution does not exist
///
/// ```
/// use eqsolver::{single_variable::Newton, SolverError};
/// let f = |x: f64| x*x + 1.;
/// let df = |x: f64| 2.*x;
///
/// // Solve with Newton's Method. Error is less than 1E-6. Starting guess is around 1.
/// let solution = Newton::new(f, df).solve(1.);
/// assert_eq!(solution.err().unwrap(), SolverError::NotANumber); // No solution, will diverge
/// ```
pub struct Newton<T, F, D> {
    f: F,
    df: D,
    tolerance: T,
    iter_max: usize,
}

impl<T, F, D> Newton<T, F, D>
where
    T: Float,
    F: Fn(T) -> T,
    D: Fn(T) -> T,
{
    /// Set up the solver
    ///
    /// Instantiates the solver using the given closure representing the function `f` to find roots for. This function also takes `f`'s derivative `df`
    pub fn new(f: F, df: D) -> Self {
        Self {
            f,
            df,
            tolerance: T::from(DEFAULT_TOL).unwrap(),
            iter_max: DEFAULT_ITERMAX,
        }
    }

    /// Updates the solver's tolerance (Magnitude of Error).
    ///
    /// **Default Tolerance:** `1e-6`
    ///
    /// ## Examples
    /// ```
    /// use eqsolver::single_variable::Newton;
    /// let f = |x: f64| x*x - 2.; // Solve x^2 = 2
    /// let df = |x: f64| 2.*x; // Derivative of f
    /// let solution = Newton::new(f, df)
    ///     .with_tol(1e-12)
    ///     .solve(1.4)
    ///     .unwrap();
    /// assert!((solution - 2_f64.sqrt()).abs() <= 1e-12);
    /// ```
    pub fn with_tol(&mut self, tol: T) -> &mut Self {
        self.tolerance = tol;
        self
    }

    /// Updates the solver's amount of iterations done before terminating the iteration
    ///
    /// **Default Max Iterations:** `50`
    ///
    /// ## Examples
    /// ```
    /// use eqsolver::{single_variable::Newton, SolverError};
    ///
    /// let f = |x: f64| x.powf(-x); // Solve x^-x = 0
    /// let df = |x: f64| -x.powf(-x) * (1. + x.ln()); // Derivative of f
    /// let solution = Newton::new(f, df)
    ///     .with_itermax(20)
    ///     .solve(1.); // Solver will terminate after 20 iterations
    /// assert_eq!(solution.err().unwrap(), SolverError::MaxIterReached);
    /// ```
    pub fn with_itermax(&mut self, max: usize) -> &mut Self {
        self.iter_max = max;
        self
    }

    /// Solves for `x` in `f(x) = 0` where `f` is the stored function.
    ///
    /// ## Examples
    /// ```
    /// use eqsolver::{DEFAULT_TOL, single_variable::Newton};
    /// let f = |x: f64| x*x - 2.; // Solve x^2 = 2
    /// let df = |x: f64| 2.*x; // Derivative of f
    /// let solution = Newton::new(f, df)
    ///     .solve(1.4)
    ///     .unwrap();
    /// assert!((solution - 2_f64.sqrt()).abs() <= DEFAULT_TOL); // Default Tolerance = 1e-6
    /// ```
    pub fn solve(&self, mut x0: T) -> SolverResult<T> {
        let mut dx = T::max_value(); // We assume error is infinite at the start
        let mut iter = 1;

        // Newton-Raphson's Iteration
        while dx.abs() > self.tolerance && iter <= self.iter_max {
            dx = (self.f)(x0) / (self.df)(x0);
            x0 = x0 - dx;
            iter += 1;
        }

        if iter >= self.iter_max {
            return Err(SolverError::MaxIterReached);
        }

        if x0.is_nan() {
            return Err(SolverError::NotANumber);
        }

        Ok(x0)
    }
}
