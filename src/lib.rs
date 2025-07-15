pub extern crate nalgebra;
use nalgebra::{allocator::Allocator, DefaultAllocator, Matrix, Vector};
use num_traits::Float;

mod solvers;
pub use solvers::*;

/// Numerical integrators
pub mod integrators;

/// Default tolerance (error of magnitude)
pub const DEFAULT_TOL: f64 = 1e-6;

/// Default amount of max iterations for the iterative methods
pub const DEFAULT_ITERMAX: usize = 50;

/// Types of errors encountered by the solvers
#[derive(Debug, PartialEq)]
pub enum SolverError {
    /// The amount of iterations (or similar limits) reached the limit
    MaxIterReached,

    /// The value evalutated is a `NaN`
    NotANumber,

    /// The given input is not correct
    IncorrectInput,

    /// A Jacobian Matrix in the iteration was singular or ill-defined
    BadJacobian,

    /// A conversion between types, typically between `usize` and `T: Float` has failed.
    TypeConversionError,

    /// An error from an external crate or library
    ExternalError,
}

/// Alias for `Result<T, SolverError>`
pub type SolverResult<T> = Result<T, SolverError>;

type VectorType<T, D> = Vector<T, D, <DefaultAllocator as Allocator<D>>::Buffer<T>>;
type MatrixType<T, R, C> = Matrix<T, R, C, <DefaultAllocator as Allocator<R, C>>::Buffer<T>>;

/// A struct encapsulating mean and variance of a measured quantity
///
/// NOTE! A precondition of constructing and modifying this struct is that
/// the `variance` field is always non-negative.
#[derive(Debug, Copy, Clone)]
pub struct MeanVariance<T> {
    pub mean: T,
    pub variance: T,
}

impl<T: Float> MeanVariance<T> {
    /// Creates a `MeanVariance` given the values for the mean and variance.
    ///
    /// This will fail if the variance is negative.
    ///
    /// ## Examples
    /// ```
    /// use eqsolver::MeanVariance;
    ///
    /// let mean_variance = MeanVariance::new(1., 2.).unwrap();
    /// assert_eq!(mean_variance.mean, 1.);
    /// assert_eq!(mean_variance.variance, 2.);
    /// ```
    pub fn new(mean: T, variance: T) -> SolverResult<Self> {
        if variance < T::zero() {
            return Err(SolverError::IncorrectInput);
        }
        Ok(Self { mean, variance })
    }

    /// Returns a `MeanVariance` with the mean and variance both equal to zero.
    ///
    /// ## Examples
    /// ```
    /// use eqsolver::MeanVariance;
    ///
    /// let mean_variance: MeanVariance<f64> = MeanVariance::zero();
    /// assert_eq!(mean_variance.mean, 0.);
    /// assert_eq!(mean_variance.variance, 0.);
    /// ```
    pub fn zero() -> Self {
        Self {
            mean: T::zero(),
            variance: T::zero(),
        }
    }

    /// Returns the standard deviation (square root of the variance) given the values
    /// in this `MeanVariance`
    ///
    /// ## Examples
    /// ```
    /// use eqsolver::MeanVariance;
    ///
    /// let mean_variance = MeanVariance::new(0., 4.).unwrap();
    /// assert_eq!(mean_variance.standard_deviation(), 2.);
    /// ```
    pub fn standard_deviation(&self) -> T {
        self.variance.sqrt()
    }

    /// Scales the mean by a factor and updates the variance accordingly.
    ///
    /// ## Examples
    /// ```
    /// use eqsolver::MeanVariance;
    ///
    /// let mut mean_variance = MeanVariance::new(2., 8.).unwrap();
    /// let new_mean_variance = mean_variance.scale_mean(0.5);
    /// assert_eq!(new_mean_variance.mean, 1.);
    /// assert_eq!(new_mean_variance.variance, 2.);
    /// ```
    pub fn scale_mean(&self, factor: T) -> Self {
        Self {
            mean: self.mean * factor,
            variance: self.variance * factor * factor,
        }
    }

    /// Combine two `MeanVariance`s by addition, i.e. add their means and variances together;
    ///
    /// ## Examples
    /// ```
    /// use eqsolver::MeanVariance;
    ///
    /// let mean_variance1 = MeanVariance::new(1., 2.).unwrap();
    /// let mean_variance2 = MeanVariance::new(2., 3.).unwrap();
    /// let result = mean_variance1.add(&mean_variance2);
    /// assert_eq!(result.mean, 3.);
    /// assert_eq!(result.variance, 5.);
    /// ```
    pub fn add(&self, other: &Self) -> Self {
        Self {
            mean: self.mean + other.mean,
            variance: self.variance + other.variance,
        }
    }

    /// Constructs a `MeanVariance` given an iterator over samples, using [Welford's Online
    /// Algorithm](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm)
    /// This also takes a `bool` signifying whether the variance is to be Bessel-corrected, i.e.
    /// if the variance or the sample variance is to be used.
    ///
    /// This will fail if the iterator is over less than 2 items.
    ///
    /// ## Examples
    /// ```
    /// use eqsolver::MeanVariance;
    /// let samples = vec![5., 7., 9., 10.];
    /// let actual_mean = samples.iter().sum::<f64>() / 4.;
    /// let actual_sample_variance = samples.iter().map(|x| (x - actual_mean).powi(2)).sum::<f64>() / 3.;
    ///
    /// let mut sample_iterator = samples.iter().map(|x| *x);
    ///
    /// let mean_variance = MeanVariance::from_iterator_using_welford(&mut sample_iterator, true).unwrap();
    /// assert!((mean_variance.mean - actual_mean) < 1e-12);
    /// assert!((mean_variance.variance - actual_sample_variance) < 1e-12);
    /// ```
    pub fn from_iterator_using_welford(
        iterator: &mut impl Iterator<Item = T>,
        use_bessel_correction: bool,
    ) -> SolverResult<Self> {
        let mut mean = T::zero();
        let mut sum_of_squares = T::zero();
        let mut total_count: usize = 0;

        for (count, sample) in iterator
            .enumerate()
            .map(|(count, i)| (T::from(count + 1).unwrap(), i))
        {
            let delta_mean = sample - mean;
            mean = mean + delta_mean / count;
            sum_of_squares = sum_of_squares + delta_mean * (sample - mean);
            total_count += 1;
        }

        if total_count < 2 {
            Err(SolverError::IncorrectInput)
        } else {
            if use_bessel_correction {
                total_count -= 1;
            }
            let total_count_as_t = T::from(total_count).ok_or(SolverError::TypeConversionError)?;
            let variance = sum_of_squares / total_count_as_t;
            Ok(Self { mean, variance })
        }
    }
}
