use crate::{
    integrators::{OrthotopeRandomIntegrator, DEFAULT_SAMPLE_COUNT},
    MeanVariance, SolverError, SolverResult, VectorType,
};
use nalgebra::{allocator::Allocator, ComplexField, DefaultAllocator, Dim, Scalar};
use num_traits::Float;
use rand::Rng;
use rand_distr::{uniform::SampleUniform, Distribution, Uniform};
use std::marker::PhantomData;

/// # Monte Carlo Integrator
///
/// A general Monte Carlo integrator of functions `f: V -> R` using a given sampler that generates
/// points in `V ⊆ Rn` along with a volume. Specifically, it calls the sampler `sample_count`
/// number of times, evaluating `f` on each point and calculates mean and variance of the result,
/// scaling them by the volume.
///
/// **Default sample count**: 1000
///
/// NOTE! For most cases, if your integral is over the subset `[a1, b1] x [a2, b2] x ... x [an, bn]`
/// for `ai, bi ∈ Rn`, then consider using `integrate(from: V, to: V)` from the
/// `OrthotopeRngIntegrator` trait which this struct implements for `V: Float` and
/// `V: nalgebra::Vector` inputs; they set up the sampler and volume and call the general function
/// `integrate_with_sampler(sampler: FnMut() -> V, volume: T)`.
///
///
/// ## Examples
///
/// ### Using the `integrate_with_sampler`
/// We want to integrate `f(x, y) = x^2 + y^2`, i.e. the norm squared over the unit circle.
/// We choose an random number generator (here, we use a seeded one to make the test
/// deterministic), then create our sampler which samples uniformly (!) random points over the
/// volume where are integrating over. Thereafter, we create a `MonteCarlo` given our function `f`
/// and integrate using the sampler and the volume of the volume we are integrating over, which in
/// this case is π (units of area).
///
/// NOTE! This method should only be used if your integration volume is not an orthotope, in which
/// case you may use a special sample (with rejection sampling, for instance). Otherwise, it is
/// recommended that you use `integrate` or `integrate_with_rng` from the
/// `OrthotopeRandomIntegrator` trait which will simplify this procedure quite a lot. See the next
/// example.
/// ```
/// use eqsolver::integrators::MonteCarlo;
/// use nalgebra::{vector, Vector2};
/// use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng}; // or any rng you like
/// use rand_distr::{Distribution, Uniform};
/// use std::f64::consts::{FRAC_PI_2, PI};
/// let mut rng = ChaCha8Rng::seed_from_u64(1729);
/// let f = |v: Vector2<f64>| v.norm_squared();
/// let unit_uniform = Uniform::new_inclusive(0., 1.).unwrap();
/// let sampler = || {
///     let radius = f64::sqrt(unit_uniform.sample(&mut rng));
///     let angle = 2. * PI * unit_uniform.sample(&mut rng);
///     radius * (vector![angle.cos(), angle.sin()])
/// };
/// let result = MonteCarlo::new(f).integrate_with_sampler(sampler, PI).unwrap();
/// assert!((result.mean - FRAC_PI_2).abs() <= 0.1)
/// ```
///
/// ### Using `OrthotopeRandomIntegrator`
/// ```
/// use eqsolver::integrators::{MonteCarlo, OrthotopeRandomIntegrator};
/// use nalgebra::{Vector2, vector};
/// use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng}; // or any rng you like
/// let mut rng = ChaCha8Rng::seed_from_u64(1729);
/// let f = |v: Vector2<f64>| (v[1].cos() + v[0]).sin();
/// let lower = vector![0., 0.];
/// let upper = vector![1., 1.];
/// let result = MonteCarlo::new(f).integrate_with_rng(lower, upper, &mut rng);
/// // The following will use `rng()` from `rand` (formerly known as `thread_rng()`:
/// // let result = MonteCarlo::new(f).integrate(lower, upper, &mut rng);
/// assert!((result.unwrap().mean - 0.924846).abs() <= 0.1);
/// ```
pub struct MonteCarlo<F, V, T> {
    f: F,
    sample_count: usize,
    t_phantom: PhantomData<T>,
    v_phantom: PhantomData<V>,
}

impl<F, V, T> MonteCarlo<F, V, T>
where
    T: Float,
    F: Fn(V) -> T,
{
    /// Create a new instance of the algorithm
    ///
    /// Instantiates the Monte Carlo integrator using a function `f: V -> R`
    pub fn new(f: F) -> Self {
        Self {
            f,
            sample_count: DEFAULT_SAMPLE_COUNT,
            t_phantom: PhantomData,
            v_phantom: PhantomData,
        }
    }

    /// Specify the number of samples to use in the integrator
    ///
    /// ### Using `OrthotopeRandomIntegrator`
    /// ```
    /// # use eqsolver::integrators::{MonteCarlo, OrthotopeRandomIntegrator};
    /// # use nalgebra::{Vector2, vector};
    /// # let f = |v: Vector2<f64>| (v[1].cos() + v[0]).sin();
    /// # let lower = vector![0., 0.];
    /// # let upper = vector![1., 1.];
    /// let result = MonteCarlo::new(f).with_sample_count(5000)
    ///                                .integrate(lower, upper);
    /// # assert!((result.unwrap() - 0.924846).abs() <= 0.1);
    /// ```
    pub fn with_sample_count(&mut self, sample_count: usize) -> &mut Self {
        self.sample_count = sample_count;
        self
    }

    /// Run the integrator given sampler function that samples points in the region of integration,
    /// and the volume of that region.
    ///
    /// NOTE! If you are integrating over a orthotope, i.e. `[a1, b1] x [a2, b2] x ... x [an, bn]`
    /// for `ai, bi ∈ Rn`, then consider using `integrate(from: V, to: V)` from the
    /// `OrthotopeRngIntegrator`. For more details read the docstring of [`MonteCarlo`].
    ///
    /// ## Examples
    /// ```
    /// use eqsolver::integrators::MonteCarlo;
    /// use nalgebra::{vector, Vector2};
    /// use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng}; // or any rng you like
    /// use rand_distr::{Distribution, Uniform};
    /// use std::f64::consts::{FRAC_PI_2, PI};
    /// let mut rng = ChaCha8Rng::seed_from_u64(1729);
    /// let f = |v: Vector2<f64>| v.norm_squared();
    /// let unit_uniform = Uniform::new_inclusive(0., 1.).unwrap();
    /// let sampler = || {
    ///     let radius = f64::sqrt(unit_uniform.sample(&mut rng));
    ///     let angle = 2. * PI * unit_uniform.sample(&mut rng);
    ///     radius * (vector![angle.cos(), angle.sin()])
    /// };
    /// let result = MonteCarlo::new(f)
    ///     .integrate_with_sampler(sampler, PI)
    ///     .unwrap();
    /// # assert!((result.mean - FRAC_PI_2).abs() <= 0.1)
    /// ```
    pub fn integrate_with_sampler(
        &self,
        mut sampler: impl FnMut() -> V,
        volume: T,
    ) -> SolverResult<MeanVariance<T>> {
        if volume < T::zero() {
            return Err(SolverError::IncorrectInput);
        }
        let mut sample_iterator = (0..self.sample_count).map(|_| (self.f)(sampler()));

        // We use the sample variance, hence the `true`
        MeanVariance::from_iterator_using_welford(&mut sample_iterator, true)
            .map(|result| result.scale_mean(volume))
    }
}

impl<F, T> OrthotopeRandomIntegrator<T, T> for MonteCarlo<F, T, T>
where
    T: Float + SampleUniform,
    F: Fn(T) -> T,
{
    fn integrate_with_rng(
        &self,
        from: T,
        to: T,
        mut rng: &mut impl Rng,
    ) -> SolverResult<MeanVariance<T>> {
        let Ok(uniform_sampler) = Uniform::new_inclusive(from, to) else {
            return SolverResult::Err(SolverError::ExternalError);
        };
        let sampler = || uniform_sampler.sample(&mut rng);

        self.integrate_with_sampler(sampler, to - from)
    }
}

impl<F, T, D> OrthotopeRandomIntegrator<VectorType<T, D>, T> for MonteCarlo<F, VectorType<T, D>, T>
where
    T: Float + Scalar + ComplexField<RealField = T> + SampleUniform,
    D: Dim,
    F: Fn(VectorType<T, D>) -> T,
    DefaultAllocator: Allocator<D>,
{
    fn integrate_with_rng(
        &self,
        from: VectorType<T, D>,
        to: VectorType<T, D>,
        mut rng: &mut impl Rng,
    ) -> SolverResult<MeanVariance<T>> {
        let uniform_samplers = from
            .iter()
            .zip(to.iter())
            .map(|(&x, &y)| Uniform::new_inclusive(x, y))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| SolverError::ExternalError)?;

        let volume = (to - &from).product();

        let sampler = || {
            let mut vector = from.clone();
            vector
                .iter_mut()
                .zip(uniform_samplers.iter())
                .for_each(|(vector_i, sampler_i)| *vector_i = sampler_i.sample(&mut rng));
            vector
        };

        self.integrate_with_sampler(sampler, volume)
    }
}
