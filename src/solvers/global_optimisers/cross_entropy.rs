use crate::{solvers::VectorType, SolverResult, DEFAULT_ITERMAX, DEFAULT_TOL};
use nalgebra::{allocator::Allocator, ComplexField, DefaultAllocator, Dim, Scalar, UniformNorm};
use num_traits::Float;
use rand::thread_rng;
use rand_distr::{Distribution, Normal, StandardNormal};
use std::marker::PhantomData;

const DEFAULT_SAMPLE_SIZE: usize = 100;
const DEFAULT_IMPORTANCE_SELECTION_SIZE: usize = 10;

/// # Cross-Entropy Optimiser
///
/// This struct approximates `x` where `F(x) <= F(y)` for all `y` in the domain `S` in `Rn` of `F`, where `F: S -> R` is an *objective* or *cost* function
/// to be minimised. This is done using the [Cross-Entropy Method](https://en.wikipedia.org/wiki/Cross-entropy_method) where the random values are
/// sampled from a multivariate normal distribution. More specifically it is a vector of independent normally distributed random variables, which means
/// the covariance matrix is diagonal, which is given to the optimiser as a vector of (sample) standard deviations.
///
/// **Default Tolerance:** `1e-6`
///
/// **Default Standard Deviation:** `[1.0, 1.0,..., 1.0]`
///
/// **Default Max Iterations:** `50`
///
/// **Default Sample Size:** `100`
///
/// **Default Importance Selection Size:** `10`
///
/// ## Examples
/// ```
/// use eqsolver::global_optimisers::CrossEntropy;
/// use nalgebra::SVector;
///
/// const SIZE: usize = 16;
/// let rastrigin = |v: SVector<f64, SIZE>| {
///     let mut total = 10. * SIZE as f64;

///     for &w in v.iter() {
///         total += w * w - 10. * (2. * std::f64::consts::PI * w).cos();
///     }

///     total
/// };
///

/// let guess = SVector::repeat(80.);
/// 
/// let optimised_position = CrossEntropy::new(rastrigin)
///     .solve(guess)
///     .unwrap();
/// ```
pub struct CrossEntropy<T, D, F>
where
    T: Float + Scalar,
    D: Dim,
    F: Fn(VectorType<T, D>) -> T,
    DefaultAllocator: Allocator<D>,
{
    f: F,
    std_dev: Option<VectorType<T, D>>,
    tolerance: T,
    iter_max: usize,
    sample_size: usize,
    importance_selection_size: usize,
    d_phantom: PhantomData<D>,
}

impl<T, D, F> CrossEntropy<T, D, F>
where
    T: Float + Scalar + ComplexField<RealField = T>,
    D: Dim,
    F: Fn(VectorType<T, D>) -> T,
    DefaultAllocator: Allocator<D>,
    StandardNormal: Distribution<T>,
{
    pub fn new(f: F) -> Self {
        Self {
            f,
            std_dev: None,
            tolerance: T::from(DEFAULT_TOL).unwrap(),
            iter_max: DEFAULT_ITERMAX,
            sample_size: DEFAULT_SAMPLE_SIZE,
            importance_selection_size: DEFAULT_IMPORTANCE_SELECTION_SIZE,
            d_phantom: PhantomData,
        }
    }

    /// Set the tolerance of the optimiser.
    ///
    /// **Default Tolerance:** `1e-6`
    ///## Examples
    /// ```
    /// # use eqsolver::global_optimisers::CrossEntropy;
    /// # use nalgebra::SVector;
    /// # const SIZE: usize = 4;
    /// # let f = |v: SVector<f64, SIZE>| {
    /// #   let mut total = 10. * SIZE as f64;
    /// #   for &w in v.iter() {
    /// #       total += w * w - 10. * (2. * std::f64::consts::PI * w).cos();
    /// #   }
    /// #   total
    /// # };
    /// # let guess = SVector::repeat(80.);
    /// let optimised_position = CrossEntropy::new(f)
    ///     .with_tol(1e-12)
    ///     .solve(guess)
    ///     .unwrap();
    /// ```
    pub fn with_tol(&mut self, tolerance: T) -> &mut Self {
        self.tolerance = tolerance;
        self
    }

    /// Set the maximum number of iterations of the optimiser. After that number of iterations is reached,
    /// the current mean vector (best value) is returned.
    /// 
    /// **Default Max Iterations:** `50`
    /// ## Examples
    /// ```
    /// # use eqsolver::global_optimisers::CrossEntropy;
    /// # use nalgebra::SVector;
    /// # const SIZE: usize = 4;
    /// # let f = |v: SVector<f64, SIZE>| {
    /// #   let mut total = 10. * SIZE as f64;
    /// #   for &w in v.iter() {
    /// #       total += w * w - 10. * (2. * std::f64::consts::PI * w).cos();
    /// #   }
    /// #   total
    /// # };
    /// # let guess = SVector::repeat(80.);
    /// let optimised_position = CrossEntropy::new(f)
    ///     .with_iter_max(100)
    ///     .solve(guess)
    ///     .unwrap();
    /// ```
    pub fn with_iter_max(&mut self, iter_max: usize) -> &mut Self {
        self.iter_max = iter_max;
        self
    }

    /// Sets the number of samples drawn each iteration. This value should be greater 
    /// than the importance selection size
    /// 
    /// **Default Sample Size:** `100`
    /// ## Examples
    /// ```
    /// # use eqsolver::global_optimisers::CrossEntropy;
    /// # use nalgebra::SVector;
    /// # const SIZE: usize = 4;
    /// # let f = |v: SVector<f64, SIZE>| {
    /// #   let mut total = 10. * SIZE as f64;
    /// #   for &w in v.iter() {
    /// #       total += w * w - 10. * (2. * std::f64::consts::PI * w).cos();
    /// #   }
    /// #   total
    /// # };
    /// # let guess = SVector::repeat(80.);
    /// let optimised_position = CrossEntropy::new(f)
    ///     .with_sample_size(30)
    ///     .solve(guess)
    ///     .unwrap();
    /// ```
    pub fn with_sample_size(&mut self, sample_size: usize) -> &mut Self {
        self.sample_size = sample_size;
        self
    }

    /// Sets how many of the sampled values are selected in order of smallest objective function value. 
    /// This value should be less than sample size.
    /// 
    /// **Default Importance Selection Size:** `10`
    /// ## Examples
    /// ```
    /// # use eqsolver::global_optimisers::CrossEntropy;
    /// # use nalgebra::SVector;
    /// # const SIZE: usize = 4;
    /// # let f = |v: SVector<f64, SIZE>| {
    /// #   let mut total = 10. * SIZE as f64;
    /// #   for &w in v.iter() {
    /// #       total += w * w - 10. * (2. * std::f64::consts::PI * w).cos();
    /// #   }
    /// #   total
    /// # };
    /// # let guess = SVector::repeat(80.);
    /// let optimised_position = CrossEntropy::new(f)
    ///     .with_importance_selection_size(3)
    ///     .solve(guess)
    ///     .unwrap();
    /// ```
    pub fn with_importance_selection_size(
        &mut self,
        importance_selection_size: usize,
    ) -> &mut Self {
        self.importance_selection_size = importance_selection_size;
        self
    }

    /// Sets the initial standard deviation vector used in the optimiser. The vector should contain positive values which
    /// should be *large* when the uncertainty is high.
    /// 
    /// **Default Standard Deviations:** `[1.0, 1.0,... 1.0]`
    /// ## Examples
    /// ```
    /// # use eqsolver::global_optimisers::CrossEntropy;
    /// # use nalgebra::SVector;
    /// # const SIZE: usize = 4;
    /// # let f = |v: SVector<f64, SIZE>| {
    /// #   let mut total = 10. * SIZE as f64;
    /// #   for &w in v.iter() {
    /// #       total += w * w - 10. * (2. * std::f64::consts::PI * w).cos();
    /// #   }
    /// #   total
    /// # };
    /// # let guess = SVector::repeat(80.);
    /// let std_dev = SVector::repeat(100.);
    /// let optimised_position = CrossEntropy::new(f)
    ///     .with_std_dev(std_dev)
    ///     .solve(guess)
    ///     .unwrap();
    /// ```
    pub fn with_std_dev(&mut self, std_dev: VectorType<T, D>) -> &mut Self {
        self.std_dev = Some(std_dev);
        self
    }

    /// Optimises the function using a given initial value (or guess) by returning an approximation of the global
    /// minimum of the objective function.
    /// 
    /// ## Examples
    /// ```
    /// # use eqsolver::global_optimisers::CrossEntropy;
    /// # use nalgebra::SVector;
    /// # const SIZE: usize = 4;
    /// # let f = |v: SVector<f64, SIZE>| {
    /// #   let mut total = 10. * SIZE as f64;
    /// #   for &w in v.iter() {
    /// #       total += w * w - 10. * (2. * std::f64::consts::PI * w).cos();
    /// #   }
    /// #   total
    /// # };
    /// # let guess = SVector::repeat(80.);
    /// let optimised_position = CrossEntropy::new(f)
    ///     .solve(guess)
    ///     .unwrap();
    /// ```
    pub fn solve(&self, x0: VectorType<T, D>) -> SolverResult<VectorType<T, D>> {
        let mut mus = x0.clone();
        let mut sigmas = self.std_dev.clone().unwrap_or_else(|| {
            let mut x0 = x0.clone();
            x0.fill(T::one());
            x0
        });

        let mut iter = 1;

        while sigmas.apply_norm(&UniformNorm) > self.tolerance && iter < self.iter_max {
            let distributions: Vec<_> = mus
                .iter()
                .zip(sigmas.iter())
                .map(|(&mu, &sigma)| Normal::new(mu, sigma).unwrap())
                .collect();

            let mut x_fx_pairs = vec![];

            for _ in 0..self.sample_size {
                let mut sample_x = x0.clone();
                sample_x
                    .iter_mut()
                    .zip(distributions.iter())
                    .for_each(|(x, dist)| {
                        *x = dist.sample(&mut thread_rng());
                    });

                x_fx_pairs.push((sample_x.clone(), (self.f)(sample_x)));
            }

            x_fx_pairs.sort_unstable_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());

            let mut new_mus = x0.clone();
            let mut new_sigmas = x0.clone();

            for (i, (mu, sigma)) in new_mus.iter_mut().zip(new_sigmas.iter_mut()).enumerate() {
                // Calculate sample mean for each dimension
                *mu = T::zero();
                for (v, _) in x_fx_pairs.iter().take(self.importance_selection_size) {
                    *mu += v[i];
                }
                *mu /= T::from_usize(10).unwrap();

                // Calculate sample standard deviation or each dimension
                *sigma = T::zero();
                for (v, _) in x_fx_pairs.iter().take(self.importance_selection_size) {
                    *sigma += Float::powi(v[i] - *mu, 2);
                }
                // -1 for the Bessel correction
                *sigma /= T::from_usize(self.importance_selection_size - 1).unwrap();
                *sigma = Float::sqrt(*sigma);
            }

            mus = new_mus;
            sigmas = new_sigmas;

            iter += 1;
        }

        Ok(mus)
    }
}
