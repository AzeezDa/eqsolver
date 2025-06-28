use crate::{SolverError, SolverResult};
use num_traits::Float;

/// Default number of subdivisions the interval is cut in when using a composite Newton-Cotes
pub const DEFAULT_SUBDIVISIONS: usize = 1000;

/// # `Formula`
/// The Newton-Cotes formula used in the integrator.
/// Only includes Trapezium, Simpson's 1/3, and Simpson's 3/8
pub enum Formula {
    /// [The Trapezium Rule](https://en.wikipedia.org/wiki/Trapezoidal_rule)
    ///
    /// Uses linear interpolation
    Trapezium,

    /// [Simpson's 1/3 Rule](https://en.wikipedia.org/wiki/Simpson%27s_rule)
    ///
    /// Uses quadratic interpolation
    SimpsonsOneThird,

    /// [Simpson's 3/8 Rule](https://en.wikipedia.org/wiki/Simpson%27s_rule#Simpson's_3/8_rule)
    ///
    /// Uses cubic interpolation
    SimpsonsThreeEighths,
}

/// # Newton-Cotes
///
/// A numerical integrator of functions `f: R -> R` based on the composite
/// [Newton-Cotes](https://en.wikipedia.org/wiki/Newton%E2%80%93Cotes_formulas).
///
///
/// **Default Formula:** Simpson's 1/3
///
/// **Default Subdivisions**: 1000
///
/// **NOTE!** For an integrator with tolerance use `AdaptiveNewtonCotes`.
///
/// ## Examples
/// Given `f(x) = e^(sin(x))`, we want to integrate from 0 to 1.
/// ```
/// use eqsolver::integrators::NewtonCotes;
/// let f = |x: f64| x.sin().exp();
/// let integral = NewtonCotes::new(f).integrate(0., 1.).unwrap();
/// assert!((integral - 1.631).abs() <= 1e-3);
/// ```
pub struct NewtonCotes<T, F> {
    f: F,
    formula: fn(&Self, &[T], T) -> T,
    subdivisions: usize,
}

impl<T, F> NewtonCotes<T, F>
where
    F: Fn(T) -> T,
    T: Float,
{
    /// Create a new instance of the algorithm
    ///
    /// Instantiates the Newton-Cotes integrator using a function `f: R -> R`
    pub fn new(f: F) -> Self {
        Self {
            f,
            subdivisions: DEFAULT_SUBDIVISIONS,
            formula: Self::simpsons_one_third,
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
    /// use eqsolver::integrators::{NewtonCotes, Formula};
    /// let f = |x: f64| x.sin().exp();
    /// let integral = NewtonCotes::new(f).with_formula(Formula::Trapezium)
    ///                                   .integrate(0., 1.)
    ///                                   .unwrap();
    /// assert!((integral - 1.631).abs() <= 1e-3);
    /// ```
    pub fn with_formula(&mut self, formula: Formula) -> &mut Self {
        self.formula = match formula {
            Formula::Trapezium => Self::trapezium,
            Formula::SimpsonsOneThird => Self::simpsons_one_third,
            Formula::SimpsonsThreeEighths => Self::simpsons_three_eighths,
        };
        self
    }

    /// Specify the number of subintervals the integration interval is split in.
    ///
    /// The default is 1000.
    ///
    /// ## Examples
    /// ```
    /// use eqsolver::integrators::{NewtonCotes, Formula};
    /// let f = |x: f64| x.sin().exp();
    /// let integral = NewtonCotes::new(f).with_subdivisions(200)
    ///                                   .integrate(0., 1.)
    ///                                   .unwrap();
    /// assert!((integral - 1.631).abs() <= 1e-3);
    /// ```
    pub fn with_subdivisions(&mut self, subdivisions: usize) -> &mut Self {
        self.subdivisions = subdivisions;
        self
    }

    /// Run the integrator given the integration interval
    ///
    /// Numerically calculate the integral of `f` from `from` to `to`.
    ///
    /// ## Examples
    /// Given `f(x) = e^(-x^2)`, we want to integrate from 0 to 1.
    /// ```
    /// use eqsolver::integrators::NewtonCotes;
    /// let f = |x: f64| (-x*x).exp();
    /// let integral = NewtonCotes::new(f).integrate(0., 1.).unwrap();
    /// assert!((integral - 0.7468).abs() <= 1e-3);
    /// ```
    pub fn integrate(&self, from: T, to: T) -> SolverResult<T> {
        if from > to {
            return Err(SolverError::IncorrectInput);
        }

        let mut subdivision_values = vec![T::zero(); self.subdivisions + 1];
        let Some(subdivisions_as_t) = T::from(self.subdivisions) else {
            return Err(SolverError::IncorrectInput);
        };

        let delta = (to - from) / subdivisions_as_t;

        for i in 0..=self.subdivisions {
            subdivision_values[i] = (self.f)(from + T::from(i).unwrap() * delta);
        }

        Ok((self.formula)(&self, &subdivision_values, delta))
    }

    // === PRIVATE FUNCTIONS: The composite Newton-Cotes formulas ===

    fn trapezium(&self, subdivision_values: &[T], delta: T) -> T {
        let half = T::from(0.5).unwrap();
        half * delta
            * subdivision_values
                .windows(2)
                .map(|f| f[0] + f[1])
                .fold(T::zero(), T::add)
    }

    fn simpsons_one_third(&self, subdivision_values: &[T], delta: T) -> T {
        let one_third = T::from(1. / 3.).unwrap();
        let four = T::from(4.0).unwrap();
        one_third
            * delta
            * subdivision_values
                .windows(3)
                .step_by(2)
                .map(|f| f[0] + four * f[1] + f[2])
                .fold(T::zero(), T::add)
    }

    fn simpsons_three_eighths(&self, subdivision_values: &[T], delta: T) -> T {
        let one_third = T::from(3. / 8.).unwrap();
        let three = T::from(3.).unwrap();
        one_third
            * delta
            * subdivision_values
                .windows(4)
                .step_by(3)
                .map(|f| f[0] + three * (f[1] + f[2]) + f[3])
                .fold(T::zero(), T::add)
    }
}
