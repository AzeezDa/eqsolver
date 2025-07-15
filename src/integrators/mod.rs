mod adaptive_newton_cotes;
mod miser;
mod monte_carlo;
mod newton_cotes;

pub use adaptive_newton_cotes::AdaptiveNewtonCotes;
pub use miser::MISER;
pub use monte_carlo::MonteCarlo;
pub use newton_cotes::{Formula, NewtonCotes, DEFAULT_SUBDIVISIONS};

use crate::{MeanVariance, SolverResult};
use rand::{rng, Rng};

/// Default number of cuts the interval is split in during adaptive integration methods
pub const DEFAULT_MAXIMUM_CUT_COUNT: usize = 1000;

/// Default number of samples drawn during Monte Carlo integration algorithms
pub const DEFAULT_SAMPLE_COUNT: usize = 1000;

/// Orthotope Random Integrator
///
/// Provides an interface for Monte Carlo-like integrators that use randomness to evaluate
/// integrals over `[a1, b1] x [a2, b2] x ... x [an, bn]` for `a1, ..., an, b1, ..., bn ∈ Rn`.
/// The randomness can be controlled using by providing a `rand::Rng`, otherwise, the default is
/// `rand::rng()` (previously known as `rand::thread_rng()`.
pub trait OrthotopeRandomIntegrator<V, T> {
    /// Run the integrator over an interval or a Cartesian product of intervals. Represented as
    /// either a `Float` or a `nalgebra::Vector`. Since the integrator is random, a random number
    /// generator is also an input to this function of type `rng::Rng`.
    ///
    /// ## Examples
    ///
    /// ### 1D Monte Carlo
    /// ```
    /// use eqsolver::integrators::{OrthotopeRandomIntegrator, MonteCarlo};
    /// use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng}; // or any rng you like
    /// const SOLUTION: f64 = 0.746824132812427025;
    /// let mut rng = ChaCha8Rng::seed_from_u64(1729);
    /// let f = |x: f64| (-x * x).exp();
    /// let result = MonteCarlo::new(f).integrate_with_rng(0., 1., &mut rng);
    /// assert!((result.unwrap().mean - SOLUTION).abs() <= 0.1);
    /// ```
    ///
    /// ### 2D Monte Carlo
    /// ```
    /// use eqsolver::integrators::{OrthotopeRandomIntegrator, MonteCarlo};
    /// use nalgebra::{Vector2, vector};
    /// use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng}; // or any rng you like
    /// const SOLUTION: f64 = 0.9248464799519;
    /// let mut rng = ChaCha8Rng::seed_from_u64(1729);
    /// let f = |v: Vector2<f64>| (v[1].cos() + v[0]).sin();
    /// let result = MonteCarlo::new(f)
    ///     .integrate_with_rng(vector![0., 0.], vector![1., 1.], &mut rng);
    /// assert!((result.unwrap().mean - SOLUTION).abs() <= 0.1);
    /// ```
    fn integrate_with_rng(
        &self,
        from: V,
        to: V,
        rng: &mut impl Rng,
    ) -> SolverResult<MeanVariance<T>>;

    /// Run the integrator over an interval or a Cartesian product of intervals using the
    /// `rand::rng()` as the underlying random number generator. The output of this function is the
    /// mean and variance of the output of the randomised integration algorithm represented as a
    /// [`MeanVariance`].
    ///
    /// This function is equivalent to
    /// ```ignore
    /// OrthotopeRandomIntegrator::integrate_with_rng(from, to, &mut rng())
    /// ```
    /// i.e. the RNG used is
    /// `rand::rng()` (formerly known as `thread_rng()`). Therefore, for examples, please read the
    /// docstring of [`OrthotopeRandomIntegrator::integrate_with_rng`].
    fn integrate_to_mean_variance(&self, from: V, to: V) -> SolverResult<MeanVariance<T>> {
        self.integrate_with_rng(from, to, &mut rng())
    }

    /// Run the integrator over an interval or a Cartesian product of intervals using `rand::rng()`
    /// as the underlying random number generator and return only the mean.
    ///
    /// This function is equivalent to
    /// ```ignore
    /// OrthotopeRandomIntegrator::integrate_to_mean_variance(from, to)
    ///                           .map(|result| result.mean)
    /// ```
    /// which in turn is equivalent to
    /// ```ignore
    /// OrthotopeRandomIntegrator::integrate_with_rng(from, to, &mut rng())
    ///                           .map(|result| result.mean)
    /// ```
    /// Therefore, for details and examples, please read the docstring of
    /// [`OrthotopeRandomIntegrator::integrate_with_rng`] and
    /// [`OrthotopeRandomIntegrator::integrate_to_mean_variance`]
    fn integrate(&self, from: V, to: V) -> SolverResult<T> {
        self.integrate_to_mean_variance(from, to)
            .map(|result| result.mean)
    }
}
