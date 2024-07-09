use crate::{
    finite_differences::{backward, central, forward, FiniteDifferenceType},
    {SolverError, SolverResult, DEFAULT_ITERMAX, DEFAULT_TOL},
};
use num_traits::Float;
use std::ops::Fn;

/// # Newton-Raphson with Finite Differences
///
/// FDNewton solves an equation `f(x) = 0` given the function `f` as a closure that takes a `Float` and outputs a `Float`.
/// This function uses the Newton-Raphson's method ([Wikipedia](https://en.wikipedia.org/wiki/Newton%27s_method)) but approximates the derivative in the iteration using finite differences.
///
/// **Default Tolerance:** `1e-6`
///
/// **Default Max Iterations:** `50`
///
/// **Default Finite Difference:** `Central`
///
/// **Default Step length for Finite Difference:** `√(Machine Epsilon)`
///
/// ## Examples
///
/// ### A solution exists
///
/// ```
/// // Want to solve x in cos(x) = sin(x). This is equivalent to solving x in cos(x) - sin(x) = 0.
/// use eqsolver::single_variable::FDNewton;
/// let f = |x: f64| x.cos() - x.sin();
///
/// // Solve with Newton's Method with finite differences. Error is less than 1E-6. Starting guess is around 0.8.
/// let solution = FDNewton::new(f).with_tol(1e-6).solve(0.8).unwrap();
/// assert!((solution - std::f64::consts::FRAC_PI_4).abs() <= 1e-6); // Exactly x = pi/4
/// ```
///
/// ### A solution does not exist
///
/// ```
/// use eqsolver::{single_variable::FDNewton, SolverError};
/// let f = |x: f64| x*x + 1.;
///
/// // Solve with Newton's Method with finite differences. Error is less than 1E-6. Starting guess is around 0.8.
/// let solution = FDNewton::new(f).solve(1.);
/// assert_eq!(solution.err().unwrap(), SolverError::NotANumber); // No solutions, will diverge!
/// ```
pub struct FDNewton<T, F> {
    f: F,
    finite_diff: fn(F, T, T) -> T,
    fd_step_length: T,
    tolerance: T,
    iter_max: usize,
}

impl<T, F> FDNewton<T, F>
where
    T: Float,
    F: Fn(T) -> T + Copy,
{
    /// Set up the solver
    ///
    /// Instantiates the solver using the given closure representing the function to find roots for.
    pub fn new(f: F) -> Self {
        Self {
            f,
            finite_diff: central,
            fd_step_length: T::epsilon().sqrt(),
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
    /// use eqsolver::single_variable::FDNewton;
    /// let f = |x: f64| x*x - 2.; // Solve x^2 = 2
    /// let solution = FDNewton::new(f)
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
    /// use eqsolver::{single_variable::FDNewton, SolverError};
    ///
    /// let f = |x: f64| x.powf(-x); // Solve x^-x = 0
    /// let solution = FDNewton::new(f)
    ///     .with_itermax(20)
    ///     .solve(1.); // Solver will terminate after 20 iterations
    /// assert_eq!(solution.err().unwrap(), SolverError::MaxIterReached);
    /// ```
    pub fn with_itermax(&mut self, max: usize) -> &mut Self {
        self.iter_max = max;
        self
    }

    /// Updates the step length used in the finite difference
    ///
    /// **Default Step length for Finite Difference:** `√(Machine Epsilon)`
    ///
    /// ## Examples
    /// ```
    /// # use eqsolver::single_variable::FDNewton;
    /// # let f = |x: f64| x.exp() - 2.; // Solve e^x = 2
    /// // -- snip --
    /// let solution = FDNewton::new(f)
    ///     .with_fd_step_length(0.1)
    ///     .solve(0.7);
    /// # assert!((solution.unwrap() - 2_f64.ln()).abs() <= 1e-6);
    /// ```
    pub fn with_fd_step_length(&mut self, h: T) -> &mut Self {
        self.fd_step_length = h;
        self
    }

    /// Updates the type of finite difference used in the solver for derivative approximation.
    ///
    /// There are 3 types available: `Forward`, `Backward` and `Central`.
    ///
    /// **Default Finite Difference:** `Central`
    ///
    /// ## Examples
    /// ```
    /// # use eqsolver::{DEFAULT_TOL, single_variable::FDNewton};
    /// use eqsolver::finite_differences::FiniteDifferenceType;
    /// // -- snip --
    /// # let f = |x: f64| x.exp() - 2.; // Solve e^x = 2
    /// let solution = FDNewton::new(f)
    ///     .with_finite_difference(FiniteDifferenceType::Forward)
    ///     .solve(0.7);
    /// # assert!((solution.unwrap() - 2_f64.ln()).abs() <= DEFAULT_TOL);
    /// ```
    pub fn with_finite_difference(&mut self, fd_type: FiniteDifferenceType) -> &mut Self {
        match fd_type {
            FiniteDifferenceType::Central => self.finite_diff = central,
            FiniteDifferenceType::Forward => self.finite_diff = forward,
            FiniteDifferenceType::Backward => self.finite_diff = backward,
        }
        self
    }

    /// Solves `x` in `f(x) = 0` where `f` is the stored function. The given parameter `x0` is the starting guess.
    ///
    /// ## Examples
    /// ```
    /// use eqsolver::{DEFAULT_TOL, single_variable::FDNewton};
    /// let f = |x: f64| x*x - 2.; // Solve x^2 = 2
    /// let solution = FDNewton::new(f)
    ///     .solve(1.4)
    ///     .unwrap();
    /// assert!((solution - 2_f64.sqrt()).abs() <= DEFAULT_TOL); // Default Tolerance = 1e-6
    /// ```
    pub fn solve(&self, mut x0: T) -> SolverResult<T> {
        let mut dx = T::max_value(); // We assume error is infinite at the start
        let mut iter = 1;

        // Newton-Raphson Iteration
        while dx.abs() > self.tolerance && iter <= self.iter_max {
            dx = (self.f)(x0) / (self.finite_diff)(self.f, x0, self.fd_step_length);
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
