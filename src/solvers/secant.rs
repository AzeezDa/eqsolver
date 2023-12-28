use num_traits::Float;
use super::{SolverError, DEFAULT_ITERMAX, DEFAULT_TOL};
use std::ops::Fn;


/// # Secant Method
/// 
/// Secant solves an equation `f(x) = 0` given the function `f` as a closure that takes a `Float` and outputs a `Float`.
/// This function uses the Secant method ([Wikipedia](https://en.wikipedia.org/wiki/Secant_method)).
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
/// use eqsolver::single_variable::Secant;
/// let f = |x: f64| x.cos() - x.sin();
/// 
/// // Solve with Secant Method. Error is less than 1E-6. Starting guesses is 0.7 and 0.8.
/// let solution = Secant::new(f)
///     .with_tol(1e-6)
///     .solve(0.7, 0.8)
///     .unwrap();
/// assert!((solution - std::f64::consts::FRAC_PI_4).abs() <= 1e-6); // Exactly x = pi/4
/// ```
/// 
/// ### A solution does not exist
/// 
/// ```
/// use eqsolver::{single_variable::Secant, SolverError};
/// let f = |x: f64| x*x + 1.;
/// 
/// // Solve with Secant Method. Error is less than 1E-6. Starting guesses is 0 and 1
/// let solution = Secant::new(f).solve(0., 1.);
/// assert_eq!(solution.err().unwrap(), SolverError::NotANumber); // No solution, will diverge
/// ```
pub struct Secant<T, F>
where
    T: Float,
    F: Fn(T) -> T,
{
    f: F,
    tolerance: T,
    iter_max: usize,
}

impl<T, F> Secant<T, F>
where
    T: Float,
    F: Fn(T) -> T,
{
    /// Set up the solver
    /// 
    /// Instantiates the solver using the given closure representing the function `f` to find the roots for.
    pub fn new(f: F) -> Self {
        Self {
            f,
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
    /// use eqsolver::single_variable::Secant;
    /// let f = |x: f64| x*x - 2.; // Solve x^2 = 2
    /// let solution = Secant::new(f)
    ///     .with_tol(1e-12)
    ///     .solve(1.4, 1.5)
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
    /// use eqsolver::{single_variable::Secant, SolverError};
    /// 
    /// let f = |x: f64| x.powf(-x); // Solve x^-x = 0
    /// let solution = Secant::new(f)
    ///     .with_itermax(20)
    ///     .solve(0.5, 1.); // Solver will terminate after 20 iterations
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
    /// use eqsolver::{DEFAULT_TOL, single_variable::Secant};
    /// let f = |x: f64| x*x - 2.; // Solve x^2 = 2
    /// let solution = Secant::new(f)
    ///     .solve(1.4, 1.5)
    ///     .unwrap();
    /// assert!((solution - 2_f64.sqrt()).abs() <= DEFAULT_TOL); // Default Tolerance = 1e-6
    /// ```
    /// 
    /// Giving the same point as starting guesses
    /// ```
    /// use eqsolver::{DEFAULT_TOL, single_variable::Secant, SolverError};
    /// let f = |x: f64| x*x - 2.; // Solve x^2 = 2
    /// let solution = Secant::new(f)
    ///     .solve(1.4, 1.4);
    /// assert_eq!(solution.err().unwrap(), SolverError::IncorrectInput);
    /// ```
    pub fn solve(&self, mut x0: T, mut x1: T) -> Result<T, SolverError> {
        if x0 == x1 { // If the same point is given as starting guesses, return error
            return Err(SolverError::IncorrectInput);
        };

        let mut dx = T::max_value(); // We assume error is infinite at the start
        let mut iter = 1;

        // Secant Method
        let mut f0 = (self.f)(x0);
        let mut f1 = (self.f)(x1);
        while dx.abs() > self.tolerance && iter <= self.iter_max {
            dx = f1 * (x1 - x0) / (f1 - f0); // Approximation of tangent line is secant
            x0 = x1;
            x1 = x1 - dx;
            f0 = f1;
            f1 = (self.f)(x1);
            iter += 1;
        }

        if iter >= self.iter_max {
            return Err(SolverError::MaxIterReached);
        }

        if x1.is_nan() {
            return Err(SolverError::NotANumber);
        }

        Ok(x1)
    }
}
