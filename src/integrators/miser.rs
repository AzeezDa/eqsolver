use std::marker::PhantomData;

use crate::{
    integrators::{OrthotopeRandomIntegrator, DEFAULT_SAMPLE_COUNT},
    MeanVariance, SolverError, SolverResult, VectorType,
};
use nalgebra::{allocator::Allocator, ComplexField, DefaultAllocator, Dim, Scalar};
use num_traits::Float;
use rand::Rng;
use rand_distr::{uniform::SampleUniform, Distribution, Uniform};

/// The default maximum recursion depth of [`MISER`]
const DEFAULT_MAXIMUM_DEPTH: usize = 5;

/// The default minimum of points to sample for each dimension. Read the documentation string of
/// [`MISER::with_minimum_dimensional_sample_count`] or [`MISER`] for more details.
const DEFAULT_MINIMUM_DIMENSIONAL_SAMPLE_COUNT: usize = 32;

/// Read [`MISER::with_alpha`] or [`MISER`] for more details.
const DEFAULT_ALPHA: f64 = 2.;

/// Read [`MISER::with_dither`] or [`MISER`] for more details.
const DEFAULT_DITHER: f64 = 0.05;

/// # The MISER Algorithm
///
/// The Monte Carlo integration algorithm MISER by Press and Farrar (See [1]) uses stratified
/// sampling based on variance of the integrand's value on the sampled points. More specifically,
/// the original subregion (which must be a Cartesian product of intervals `[a, b]` for real `a,
/// b`) is bisected along some dimension such that the integrand's value in those two subregions
/// gives the smallest total variance amongst all other bisections. The total samples points are
/// then distributed to those subregions according to the variance. This procedure is continued
/// recursively until either the recursion depth is reached or the number of sample points
/// allocated to that subregion fall under an (adjustable) threshold. The procedure is implemented
/// in [`miser_recurse`].
///
/// There are several adjustable parameters in this implementation of [`MISER`]. The parameters are
/// inspired by those used in [1]. The parameters in [`MISER`] are:
///
/// - **Sample Count**: (Default `1000`) A lower bound on the number of points to sample in total.
///   The reason this is a lower bound is because if the minimum threshold is fallen under then
///   the minimum will be samples. For example, if a region is allocated `5` points and the (total)
///   threshold value is `32`, then `32` values will be sampled. Thus, more than then sample count
///   may be allocated in total. For more details, read **Minimum Dimensional Sample Count** below
///   or read the docstrings of [`MISER::with_sample_count`] and
///   [`MISER::with_minimum_dimensional_sample_count`].
/// - **Maximum Depth**: (Default `5`) The maximum recursion depth, i.e. the recursive calls will
///   be no deeper than the number given by this parameter.
/// - **Minimum Dimensional Sample Count**: (Default `32`) The minimum number of points that will
///   be sampled for a subregion, per dimension. For example if this number is `32` and the input
///   is `4` dimensional (in `R^4`) then, this number of `4x32 = 128` and if a subregion is ever
///   allocated less than `128` points, then the normal Monte Carlo integration algorithm will be
///   used on that subregion with `128` samples points. Please also read the docstring of
///   [`MISER::with_minimum_dimensional_sample_count`].
/// - **Alpha** (Default `2.`): When a bisection is chosen the number of allocated points for each
///   of the two subregions is chosen based on the variance. The exact formula is for the first
///   region is `N_1 = N * (Var_1^beta / (Var_1^beta + Var_2^beta))` where `N` is the total sample
///   points, `Var_1` and `Var_2` are the variances of the integrand on the first and second
///   subregions, respectively, and `beta = 1 / (1 + alpha)`. Simiarly, for the second subregion
///   the formula is: `N_2 = N * (Var_2^beta / (Var_1^beta + Var_2^beta))`. In (1), the authors
///   recommended `alpha` being `2.` and hence this is the default used here. See also the
///   docstring of [`MISER::with_alpha`].
/// - **Dither** (Default `0.05`): Instead of bisecting exactly in on the midpoint of the boundries
///   of the chosen dimension, a small dither is added to break the symmetry of certain functions.
///   Thus, instead of having the regions `[a, 0.5*(a + b)]` and `[0.5*(a + b), b]`, a random
///   value, call it `r`, in the range [-dither, dither] is chosen and the two subregions become:
///   `[a, (0.5 + r)*(a + b)]` and `[(0.5 + r)*(a + b), b]`. NOTE, therefore that the dither should
///   be in the interval `(0, 0.5)`. See also the docstring of [`MISER::with_dither`].
///
/// ## Examples
/// [`MISER`] implements the [`OrthotopeRandomIntegrator`] and hence we can use it thusly:
/// ```
/// use eqsolver::integrators::{MISER, OrthotopeRandomIntegrator};
/// use nalgebra::{Vector2, vector};
/// use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng}; // or any rng you like
/// let mut rng = ChaCha8Rng::seed_from_u64(1729);
/// let f = |v: Vector2<f64>| (v[1].cos() + v[0]).sin();
/// let lower = vector![0., 0.];
/// let upper = vector![1., 1.];
/// let result = MISER::new(f).integrate_with_rng(lower, upper, &mut rng);
/// assert!((result.unwrap().mean - 0.924846).abs() <= 0.001);
/// ```
///
/// ## References:
/// (1) Press, W.H. and Farrar, G.R., 1990. Recursive stratified sampling for multidimensional
///     Monte Carlo integration. Computers in Physics, 4(2), pp.190-195.
/// (2) https://www.gnu.org/software/gsl/doc/html/montecarlo.html
pub struct MISER<F, V, T> {
    f: F,
    sample_count: usize,
    maximum_depth: usize,
    minimum_dimensional_sample_count: usize,
    alpha: T,
    dither: T,
    t_phantom: PhantomData<T>,
    v_phantom: PhantomData<V>,
}

impl<F, V, T> MISER<F, V, T>
where
    T: Float,
    F: Fn(V) -> T,
{
    /// Creates a new instance of the algorithm
    ///
    /// Instantiates the MISER integrator using a function `f: R -> R`
    pub fn new(f: F) -> Self {
        Self {
            f,
            sample_count: DEFAULT_SAMPLE_COUNT,
            maximum_depth: DEFAULT_MAXIMUM_DEPTH,
            alpha: T::from(DEFAULT_ALPHA).unwrap(),
            dither: T::from(DEFAULT_DITHER).unwrap(),
            minimum_dimensional_sample_count: DEFAULT_MINIMUM_DIMENSIONAL_SAMPLE_COUNT,
            t_phantom: PhantomData,
            v_phantom: PhantomData,
        }
    }

    /// Set the total number of points that will be samples in the algorithm.
    ///
    /// NOTE! This number if a lower bound on the number of sampled points.
    /// This is because when a bisection leads to a subregion having less than
    /// the minimum sample count allocated, then the minimum sample count is sampled
    /// instead. See [`MISER::with_minimum_dimensional_sample_count`].
    ///
    /// **Default sample count**: 1000
    ///
    /// ## Examples
    /// ```
    /// # use eqsolver::integrators::{MISER, OrthotopeRandomIntegrator};
    /// # use nalgebra::{Vector2, vector};
    /// # use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng}; // or any rng you like
    /// # let mut rng = ChaCha8Rng::seed_from_u64(1729);
    /// # let f = |v: Vector2<f64>| (v[1].cos() + v[0]).sin();
    /// # let lower = vector![0., 0.];
    /// # let upper = vector![1., 1.];
    /// let result = MISER::new(f)
    ///                 .with_sample_count(2000)
    ///                 .integrate_with_rng(lower, upper, &mut rng);
    /// # assert!((result.unwrap().mean - 0.924846).abs() <= 0.001);
    /// ```
    pub fn with_sample_count(&mut self, sample_count: usize) -> &mut Self {
        self.sample_count = sample_count;
        self
    }

    /// Set the minimum number of points per dimension that will be sampled in a region.
    ///
    /// In other words, if a subregion has less than this number times the number of dimensions
    /// then the recursion stops and the plain Monte Carlo algorithm is ran on that subregion using
    /// the number of samples points specified by this function times the number of dimensions.
    ///
    /// **Default minimum sample count per dimension**: 32
    ///
    /// ## Examples
    /// ```
    /// # use eqsolver::integrators::{MISER, OrthotopeRandomIntegrator};
    /// # use nalgebra::{Vector2, vector};
    /// # use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng}; // or any rng you like
    /// # let mut rng = ChaCha8Rng::seed_from_u64(1729);
    /// # let f = |v: Vector2<f64>| (v[1].cos() + v[0]).sin();
    /// # let lower = vector![0., 0.];
    /// # let upper = vector![1., 1.];
    /// let result = MISER::new(f)
    ///                 .with_minimum_dimensional_sample_count(50)
    ///                 .integrate_with_rng(lower, upper, &mut rng);
    /// # assert!((result.unwrap().mean - 0.924846).abs() <= 0.001);
    /// ```
    pub fn with_minimum_dimensional_sample_count(
        &mut self,
        minimum_dimensional_sample_count: usize,
    ) -> &mut Self {
        self.minimum_dimensional_sample_count = minimum_dimensional_sample_count;
        self
    }

    /// Set the maximum recursion depth of the algorithm.
    ///
    /// In other words, the algorithm's recursion tree is no deeper than the number specified using
    /// this function. The recursion in this algorithm happens when a region (orthotope) is
    /// bisected along an axis. When the maximum recursion depth is reached, the plain Monte Carlo
    /// algorithm is used on that subregion (which was one or more bisections of a larger region).
    ///
    /// **Default maximum depth**: 5
    /// ```
    /// # use eqsolver::integrators::{MISER, OrthotopeRandomIntegrator};
    /// # use nalgebra::{Vector2, vector};
    /// # use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng}; // or any rng you like
    /// # let mut rng = ChaCha8Rng::seed_from_u64(1729);
    /// # let f = |v: Vector2<f64>| (v[1].cos() + v[0]).sin();
    /// # let lower = vector![0., 0.];
    /// # let upper = vector![1., 1.];
    /// let result = MISER::new(f)
    ///                 .with_maximum_depth(8)
    ///                 .integrate_with_rng(lower, upper, &mut rng);
    /// # assert!((result.unwrap().mean - 0.924846).abs() <= 0.001);
    /// ```
    pub fn with_maximum_depth(&mut self, maximum_depth: usize) -> &mut Self {
        self.maximum_depth = maximum_depth;
        self
    }

    /// Set the alpha value of the algorithm.
    ///
    /// The alpha value changes the proportions of the allocated samples points to each subregion
    /// of the bisection that happens in every recursion call of [`MISER`]. More precisely, the
    /// allocated points for each subregions follow the following formulas.
    /// - `N_1 = N * (Var_1^beta / (Var_1^beta + Var_2^beta))`
    /// - `N_2 = N * (Var_2^beta / (Var_1^beta + Var_2^beta))`
    /// where `N` is the total sample points, `Var_1` and `Var_2` are the variances of the
    /// integrand on the first and second subregions, respectively, and `beta = 1 / (1 + alpha)`.
    ///
    /// For more details read the docstring of [`MISER`].
    ///
    /// **Default alpha**: 2.
    ///
    /// ```
    /// # use eqsolver::integrators::{MISER, OrthotopeRandomIntegrator};
    /// # use nalgebra::{Vector2, vector};
    /// # use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng}; // or any rng you like
    /// # let mut rng = ChaCha8Rng::seed_from_u64(1729);
    /// # let f = |v: Vector2<f64>| (v[1].cos() + v[0]).sin();
    /// # let lower = vector![0., 0.];
    /// # let upper = vector![1., 1.];
    /// let result = MISER::new(f)
    ///                 .with_alpha(1.5)
    ///                 .integrate_with_rng(lower, upper, &mut rng);
    /// # assert!((result.unwrap().mean - 0.924846).abs() <= 0.001);
    /// ```
    pub fn with_alpha(&mut self, alpha: T) -> &mut Self {
        self.alpha = alpha;
        self
    }

    /// Set the dither value of the algorithm.
    ///
    /// The dither value controls value that makes the bisection to not be cut exactly at the
    /// midpoint. This value helps break certain symmetries of functions. For example, if a region
    /// is about to be cut along an axis whose bounds are `a < b` then, instead of having the
    /// regions `[a, 0.5*(a + b)]` and `[0.5*(a + b), b]`, a random value, call it `r`, in the
    /// range [-dither, dither] is chosen and the two subregions become:
    /// - `[a, (0.5 + r)*(a + b)]`, and
    /// - `[(0.5 + r)*(a + b), b]`.
    ///
    /// NOTE! The dither must be in the range `(0, 0.5)` (this is a precondition).
    ///
    /// **Default dither**: 0.05
    ///
    /// ```
    /// # use eqsolver::integrators::{MISER, OrthotopeRandomIntegrator};
    /// # use nalgebra::{Vector2, vector};
    /// # use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng}; // or any rng you like
    /// # let mut rng = ChaCha8Rng::seed_from_u64(1729);
    /// # let f = |v: Vector2<f64>| (v[1].cos() + v[0]).sin();
    /// # let lower = vector![0., 0.];
    /// # let upper = vector![1., 1.];
    /// let result = MISER::new(f)
    ///                 .with_dither(0.01)
    ///                 .integrate_with_rng(lower, upper, &mut rng);
    /// # assert!((result.unwrap().mean - 0.924846).abs() <= 0.001);
    /// ```
    pub fn with_dither(&mut self, dither: T) -> &mut Self {
        self.dither = dither;
        self
    }
}

impl<F, T, D> OrthotopeRandomIntegrator<VectorType<T, D>, T> for MISER<F, VectorType<T, D>, T>
where
    T: Float + Scalar + ComplexField<RealField = T> + SampleUniform,
    D: Dim,
    F: Fn(VectorType<T, D>) -> T,
    DefaultAllocator: Allocator<D>,
{
    fn integrate_with_rng(
        &self,
        mut from: VectorType<T, D>,
        mut to: VectorType<T, D>,
        rng: &mut impl rand::Rng,
    ) -> SolverResult<MeanVariance<T>> {
        miser_recurse(
            &self,
            &mut from,
            &mut to,
            rng,
            self.maximum_depth,
            self.sample_count,
        )
    }
}

// === PRIVATE HELPER FUNCTIONS ===

fn uniform<T: SampleUniform>(lower: T, upper: T) -> SolverResult<Uniform<T>> {
    Uniform::new_inclusive(lower, upper).map_err(|_| SolverError::IncorrectInput)
}

fn miser_recurse<F, T, D>(
    miser: &MISER<F, VectorType<T, D>, T>,
    from: &mut VectorType<T, D>,
    to: &mut VectorType<T, D>,
    rng: &mut impl Rng,
    depth_remaining: usize,
    sample_count: usize,
) -> SolverResult<MeanVariance<T>>
where
    T: Float + Scalar + ComplexField<RealField = T> + SampleUniform,
    D: Dim,
    F: Fn(VectorType<T, D>) -> T,
    DefaultAllocator: Allocator<D>,
{
    let half = T::from(0.5).unwrap();

    let dimensions = from.len();
    let minimum_sample_count = miser.minimum_dimensional_sample_count * dimensions;
    let samplers = from
        .iter()
        .zip(to.iter())
        .map(|(&lower, &upper)| uniform(lower, upper))
        .collect::<Result<Vec<_>, _>>()?;

    let mut vector_sampler = || {
        let mut vector = from.clone();
        for (element, sampler) in vector.iter_mut().zip(&samplers) {
            *element = sampler.sample(rng);
        }
        vector
    };

    if depth_remaining == 0 || sample_count < minimum_sample_count {
        let sample_count = sample_count.max(minimum_sample_count);
        let volume = from
            .iter()
            .zip(to.iter())
            .map(|(&lower, &upper)| upper - lower)
            .fold(T::one(), T::mul);
        // Normal Monte Carlo
        return MeanVariance::from_iterator_using_welford(
            &mut (0..sample_count).map(|_| (miser.f)(vector_sampler())),
            false,
        )
        .map(|result| result.scale_mean(volume));
    }

    let sample_count = sample_count.max(2); // To avoid having less than 2 sample count

    debug_assert!(2 <= sample_count && sample_count >= minimum_sample_count);

    // Find bisection with smallest total variance
    let mut best_variance = T::max_value();
    let mut best_lower_variance = T::max_value();
    let mut best_upper_variance = T::max_value();
    let mut best_dimension = uniform(0, dimensions - 1)?.sample(rng);
    let mut best_mid = from[best_dimension] + (to[best_dimension] - from[best_dimension]) * half;
    let dither_sampler = uniform(-miser.dither, miser.dither)?;
    for d in 0..dimensions {
        let delta = to[d] - from[d];
        let mid = from[d] + delta * (half + dither_sampler.sample(rng));
        let lower_d_sampler = uniform(from[d], mid)?;
        let upper_d_sampler = uniform(mid, to[d])?;

        let mut sampler = |d_sampler: &Uniform<T>| {
            let mut vector = from.clone();
            for (inner_d, element) in vector.iter_mut().enumerate() {
                if inner_d == d {
                    *element = d_sampler.sample(rng);
                } else {
                    *element = samplers[inner_d].sample(rng);
                }
            }
            vector
        };

        let variance_lower = MeanVariance::from_iterator_using_welford(
            &mut (0..sample_count).map(|_| (miser.f)(sampler(&lower_d_sampler))),
            true,
        )?
        .variance;

        let variance_upper = MeanVariance::from_iterator_using_welford(
            &mut (0..sample_count).map(|_| (miser.f)(sampler(&upper_d_sampler))),
            true,
        )?
        .variance;

        let total_variance = variance_lower + variance_upper;

        if total_variance < best_variance {
            best_variance = total_variance;
            best_lower_variance = variance_lower;
            best_upper_variance = variance_upper;
            best_dimension = d;
            best_mid = mid;
        }
    }

    let old_from = from[best_dimension];
    let old_to = to[best_dimension];
    let beta = T::one() / (T::one() + miser.alpha);
    let best_lower_variance_alpha = Float::powf(best_lower_variance, beta);
    let best_upper_variance_alpha = Float::powf(best_upper_variance, beta);
    let best_variance_alpha = best_lower_variance_alpha + best_upper_variance_alpha;
    let sample_count_as_t = T::from(sample_count).ok_or(SolverError::TypeConversionError)?;
    let lower_sample_fraction = best_lower_variance_alpha / best_variance_alpha;
    let upper_sample_fraction = best_upper_variance_alpha / best_variance_alpha;

    let lower_sample_count = (sample_count_as_t * lower_sample_fraction)
        .to_usize()
        .ok_or(SolverError::TypeConversionError)?;

    let upper_sample_count = (sample_count_as_t * upper_sample_fraction)
        .to_usize()
        .ok_or(SolverError::TypeConversionError)?;

    // Recurse on lower bisection
    to[best_dimension] = best_mid;
    let lower_result = miser_recurse(
        miser,
        from,
        to,
        rng,
        depth_remaining - 1,
        lower_sample_count,
    )?;
    to[best_dimension] = old_to;

    // Recurse on upper bisection
    from[best_dimension] = best_mid;
    let upper_result = miser_recurse(
        miser,
        from,
        to,
        rng,
        depth_remaining - 1,
        upper_sample_count,
    )?;
    from[best_dimension] = old_from;

    Ok(lower_result.add(&upper_result))
}
