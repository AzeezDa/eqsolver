use num_traits::Float;

use super::newton_cotes::Formula;
use crate::integrators::DEFAULT_MAXIMUM_CUT_COUNT;
use crate::DEFAULT_TOL;
use crate::{SolverError, SolverResult};


/// # Adaptive Newton-Cotes
///
/// A numerical integrator of functions `f: R -> R` based on the adaptive Newton-Cotes
/// method. In summary, it first computes the integral on the given interval [a, b] using
/// the given Newton-Cotes formula. Afterwards, it computes the integral on the intervals
/// `[a, m]` and `[m, b]` where `m = (a+b)/2`, the midpoint and adds the result. Thereafter,
/// it compares the integral on `[a, b]` with that of `[a, m]` plus `[m, b]`, and if the
/// absolute difference is under the given intervals it stops. Otherwise, it recursively
/// continues the procedure on `[a, m]` and `[m, b]`, scaling the tolerance down by a factor
/// of 2.
///
/// Note that `AdaptiveNewtonCotes` may be faster than `NewtonCotes` because of its adaptive
/// nature. For example, the former will stop when the tolerance is achieved, whereas the
/// accuracy of `NewtonCotes` is static and based on the given number of subdivisions.
/// For more details, run the benchmarks.
///
/// **Default Formula:** Simpson's 1/3
///
/// **Default Maximum Cut Amount**: 1000
///
/// **Default Tolerance**: 1e-6
///
/// ## Examples
/// Given `f(x) = e^(sin(x))`, we want to integrate from 0 to 1.
/// ```
/// use eqsolver::integrators::AdaptiveNewtonCotes;
/// let f = |x: f64| x.sin().exp();
/// let integral = AdaptiveNewtonCotes::new(f).with_tolerance(1e-6)
///                                           .integrate(0., 1.)
///                                           .unwrap();
/// const RESULT: f64 = 1.63186960841805134;
/// assert!((integral - RESULT).abs() <= 1e-6);
/// ```
pub struct AdaptiveNewtonCotes<T, F> {
    f: F,
    formula: Formula,
    maximum_cut_count: usize,
    tolerance: T,
}

impl<T, F> AdaptiveNewtonCotes<T, F>
where
    T: Float,
    F: Fn(T) -> T,
{
    /// Create a new instance of the algorithm
    ///
    /// Instantiates the Adaptive Newton-Cotes integrator using a function `f: R -> R`
    pub fn new(f: F) -> Self {
        Self {
            f,
            formula: Formula::SimpsonsOneThird,
            maximum_cut_count: DEFAULT_MAXIMUM_CUT_COUNT,
            tolerance: T::from(DEFAULT_TOL).unwrap(),
        }
    }

    /// Specify the Newton-Cotes formula to use in the integrators
    ///
    /// There are 3 formulas available:
    /// [The Trapezium Rule](https://en.wikipedia.org/wiki/Trapezoidal_rule), [Simpson's 1/3 Rule](https://en.wikipedia.org/wiki/Simpson%27s_rule), and [Simpson's 3/8 Rule](https://en.wikipedia.org/wiki/Simpson%27s_rule#Simpson's_3/8_rule).
    /// The default is Simpson's 1/3 Rule.
    ///
    /// ## Examples
    ///
    /// ```
    /// use eqsolver::integrators::{AdaptiveNewtonCotes, Formula};
    /// let f = |x: f64| x.sin().exp();
    /// let integral = AdaptiveNewtonCotes::new(f).with_formula(Formula::Trapezium)
    ///                                           .integrate(0., 1.)
    ///                                           .unwrap();
    /// const RESULT: f64 = 1.63186960841805134;
    /// assert!((integral - RESULT).abs() <= 1e-6);
    /// ```
    pub fn with_formula(&mut self, formula: Formula) -> &mut Self {
        self.formula = formula;
        self
    }

    /// Specify the maximum number of times there will be an interval splitting.
    /// For example `[0, 1]` being cut into `[0, 0.25]`, `[0.25, 0.5]`, and `[0.5, 1]` counts
    /// as 2 cuts: (1) `[0, 1]` into `[0, 0.5]` and `[0.5, 1]`, and (2) `[0, 0.5]` into
    /// `[0, 0.25]` and `[0.25, 0.5]`.
    ///
    /// The default is 1000.
    ///
    /// ## Examples
    /// ```
    /// use eqsolver::integrators::{AdaptiveNewtonCotes, Formula};
    /// let f = |x: f64| x.sin().exp();
    /// let integral = AdaptiveNewtonCotes::new(f).with_maximum_cut_count(200)
    ///                                           .integrate(0., 1.)
    ///                                           .unwrap();
    /// const RESULT: f64 = 1.63186960841805134;
    /// assert!((integral - RESULT).abs() <= 1e-6);
    /// ```
    pub fn with_maximum_cut_count(&mut self, maximum_cut_count: usize) -> &mut Self {
        self.maximum_cut_count = maximum_cut_count;
        self
    }

    /// Updates the solver's tolerance (Magnitude of Error).
    ///
    /// **Default Tolerance:** `1e-6`
    ///
    /// ## Examples
    /// ```
    /// use eqsolver::integrators::{AdaptiveNewtonCotes, Formula};
    /// let f = |x: f64| x.sin().exp();
    /// let integral = AdaptiveNewtonCotes::new(f).with_tolerance(1e-9)
    ///                                           .integrate(0., 1.)
    ///                                           .unwrap();
    /// const RESULT: f64 = 1.63186960841805134;
    /// assert!((integral - RESULT).abs() <= 1e-9);
    /// ```
    pub fn with_tolerance(&mut self, tolerance: T) -> &mut Self {
        self.tolerance = tolerance;
        self
    }

    /// Run the integrator given the integration interval
    ///
    /// Numerically calculate the integral of `f` from `from` to `to`.
    ///
    /// ## Examples
    /// Given `f(x) = e^(-x^2)`, we want to integrate from 0 to 1.
    /// ```
    /// use eqsolver::integrators::AdaptiveNewtonCotes;
    /// let f = |x: f64| (-x*x).exp();
    /// let integral = AdaptiveNewtonCotes::new(f).integrate(0., 1.).unwrap();
    ///
    /// const RESULT: f64 = 0.74682413281242702;
    /// assert!((integral - RESULT).abs() <= 1e-6);
    /// ```
    pub fn integrate(&self, from_0: T, to_0: T) -> SolverResult<T> {
        let mut intervals = Vec::with_capacity(self.maximum_cut_count);
        intervals.push((from_0, to_0));
        let mut result = T::zero();
        let mut number_of_cuts = 0;
        let delta_0 = to_0 - from_0;

        type FormulaF<T, F> = fn(&AdaptiveNewtonCotes<T, F>, T, T) -> T;
        let (formula, error_scaling): (FormulaF<T, F>, T) = match self.formula {
            Formula::Trapezium => (Self::trapezium, T::from(3.).unwrap()),
            Formula::SimpsonsOneThird => (Self::simpsons_one_third, T::from(15.).unwrap()),
            Formula::SimpsonsThreeEighths => (Self::simpsons_three_eighths, T::from(15.).unwrap()),
        };
        let scaled_tolerance = error_scaling * self.tolerance;
        let half = T::from(0.5).unwrap();

        while let Some((from_i, to_i)) = intervals.pop() {
            let delta_i = (to_i - from_i) * half;
            let mid_i = from_i + delta_i;

            let integral_full = formula(&self, from_i, to_i);
            let integral_split = formula(&self, from_i, mid_i) + formula(&self, mid_i, to_i);

            let error = (integral_full - integral_split).abs();

            if error < scaled_tolerance * delta_i / delta_0 {
                result = result + integral_split
            } else {
                intervals.push((from_i, mid_i));
                intervals.push((mid_i, to_i));
                number_of_cuts += 1;
            }
            if number_of_cuts > self.maximum_cut_count {
                return Err(SolverError::MaxIterReached);
            }
        }

        Ok(result)
    }

    // === PRIVATE FUNCTIONS: The Newton-Cotes formulas ===

    fn trapezium(&self, x0: T, x1: T) -> T {
        let half = T::from(0.5).unwrap();
        let h = x1 - x0;
        let f = &self.f;
        half * h * (f(x0) + f(x1))
    }

    fn simpsons_one_third(&self, x0: T, x2: T) -> T {
        let third = T::from(1. / 3.).unwrap();
        let four = T::from(4.).unwrap();
        let half = T::from(0.5).unwrap();
        let h = (x2 - x0) * half;
        let x1 = x0 + h;
        let f = &self.f;
        third * h * (f(x0) + four * f(x1) + f(x2))
    }

    fn simpsons_three_eighths(&self, x0: T, x3: T) -> T {
        let three_eighths = T::from(3. / 8.).unwrap();
        let three = T::from(3.0).unwrap();
        let third = three.recip();
        let h = (x3 - x0) * third;
        let x1 = x0 + h;
        let x2 = x1 + h;
        let f = &self.f;
        three_eighths * h * (f(x0) + three * (f(x1) + f(x2)) + f(x3))
    }
}
