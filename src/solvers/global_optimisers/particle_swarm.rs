use super::VectorType;
use crate::{SolverResult, DEFAULT_ITERMAX, DEFAULT_TOL};
use nalgebra::{allocator::Allocator, ComplexField, DefaultAllocator, Dim, Scalar};
use num_traits::Float;
use rand::thread_rng;
use rand_distr::{uniform::SampleUniform, Distribution, Uniform};
use std::{marker::PhantomData, slice::Iter};

const DEFAULT_INERTIA_WEIGHT: f64 = 0.5;
const DEFAULT_COGNIIVE_COEFFICIENT: f64 = 1.0;
const DEFAULT_SOCIAL_COEFFICIENT: f64 = 1.0;
const DEFAULT_PARTICLE_COUNT: usize = 1000;
const DEFAULT_STALL_ITERATIONS: usize = 50;

struct Particle<T: Scalar, D: Dim>
where
    DefaultAllocator: Allocator<D>,
{
    pub position: VectorType<T, D>,
    pub velocity: VectorType<T, D>,
    pub best_position: VectorType<T, D>,
    pub best_cost: T,
}

struct CircularArray<T, const N: usize> {
    array: [T; N],
    i: usize,
}

impl<T: Copy, const N: usize> CircularArray<T, N> {
    pub fn fill(value: T) -> Self {
        Self {
            array: [value; N],
            i: 0,
        }
    }

    pub fn push(&mut self, value: T) {
        self.array[self.i] = value;
        self.i += 1;
        self.i %= N;
    }

    pub fn iter(&self) -> Iter<T> {
        self.array.iter()
    }
}

/// # Particle Swarm Global Optimiser
///
/// This struct approximates `x` where `F(x) <= F(y)` for all `y` in the domain `S in Rn` of F, where `F: S -> R` is an *objective* or *cost* function
/// to be minimised. This is done using the [Particle Swarm Optimization](https://en.wikipedia.org/wiki/Particle_swarm_optimization) method.
///
/// **Default Tolerance:** `1e-6`
///
/// **Default Max Iterations:** `50`
///
/// **Default Particle Count:** `1000`
///
/// **Default Inertia Weight:** `0.5`
///
/// **Default Cognitive Coefficient:** `1`
///
/// **Default Social Coefficient:** `1`
///
/// ## Examples
/// ```
/// use eqsolver::global_optimisers::ParticleSwarm;
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
/// let bounds = SVector::repeat(100.);

/// let optimised_position = ParticleSwarm::new(rastrigin, -bounds, bounds)
///     .solve(guess)
///     .unwrap();
/// ```
pub struct ParticleSwarm<T: Scalar + SampleUniform, D: Dim, F>
where
    F: Fn(VectorType<T, D>) -> T,
    DefaultAllocator: Allocator<D>,
{
    f: F,
    inertia_weight: T,
    cognitive_coefficient: T,
    social_coefficient: T,
    particle_count: usize,
    tolerance: T,
    iter_max: usize,
    position_distributions: Vec<Uniform<T>>,
    velocity_distributions: Vec<Uniform<T>>,
    d_phantom: PhantomData<D>,
}

impl<T, D, F> ParticleSwarm<T, D, F>
where
    T: Scalar + Float + SampleUniform + ComplexField<RealField = T>,
    D: Dim,
    F: Fn(VectorType<T, D>) -> T,
    DefaultAllocator: Allocator<D>,
{
    /// Sets up the optimiser given the function `f` to optimise and the bounds of the hyperrectangle that contains the global minimum.
    /// The hyperrectangle is given as two vectors `lower_bounds` and `upper_bounds` containing the lower and upper bounds of each dimension,
    /// respectively.
    pub fn new(f: F, lower_bounds: VectorType<T, D>, upper_bounds: VectorType<T, D>) -> Self {
        let position_distributions = lower_bounds
            .iter()
            .zip(upper_bounds.iter())
            .map(|(&low, &up)| Uniform::new_inclusive(low, up))
            .collect();

        let velocity_distributions = lower_bounds
            .iter()
            .zip(upper_bounds.iter())
            .map(|(&low, &up)| {
                let distance = Float::abs(up - low);
                Uniform::new_inclusive(-distance, distance)
            })
            .collect();

        Self {
            f,
            inertia_weight: T::from(DEFAULT_INERTIA_WEIGHT).unwrap(),
            cognitive_coefficient: T::from(DEFAULT_COGNIIVE_COEFFICIENT).unwrap(),
            social_coefficient: T::from(DEFAULT_SOCIAL_COEFFICIENT).unwrap(),
            tolerance: T::from(DEFAULT_TOL).unwrap(),
            iter_max: DEFAULT_ITERMAX,
            particle_count: DEFAULT_PARTICLE_COUNT,
            position_distributions,
            velocity_distributions,
            d_phantom: PhantomData,
        }
    }

    /// Set the inertia weight of the optimiser. For more information about the effect of this parameter,
    /// see [Particle Swarm Optimization](https://en.wikipedia.org/wiki/Particle_swarm_optimization).
    ///
    /// ## Examples
    /// ```
    /// # use eqsolver::global_optimisers::ParticleSwarm;
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
    /// # let bounds = SVector::repeat(100.);
    /// let optimised_position = ParticleSwarm::new(f, -bounds, bounds)
    ///     .with_inertia_weight(0.8)
    ///     .solve(guess)
    ///     .unwrap();
    /// ```
    pub fn with_inertia_weight(&mut self, inertia_weight: T) -> &mut Self {
        self.inertia_weight = inertia_weight;
        self
    }

    /// Set the cognitive coefficient of the optimiser. For more information about the effect of this parameter,
    /// see [Particle Swarm Optimization](https://en.wikipedia.org/wiki/Particle_swarm_optimization).
    /// ## Examples
    /// ```
    /// # use eqsolver::global_optimisers::ParticleSwarm;
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
    /// # let bounds = SVector::repeat(100.);
    /// let optimised_position = ParticleSwarm::new(f, -bounds, bounds)
    ///     .with_cognitive_coefficient(1.5)
    ///     .solve(guess)
    ///     .unwrap();
    /// ```
    pub fn with_cognitive_coefficient(&mut self, cognitive_coefficient: T) -> &mut Self {
        self.cognitive_coefficient = cognitive_coefficient;
        self
    }

    /// Set the social coefficient of the optimiser. For more information about the effect of this parameter,
    /// see [Particle Swarm Optimization](https://en.wikipedia.org/wiki/Particle_swarm_optimization).
    /// ## Examples
    /// ```
    /// # use eqsolver::global_optimisers::ParticleSwarm;
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
    /// # let bounds = SVector::repeat(100.);
    /// let optimised_position = ParticleSwarm::new(f, -bounds, bounds)
    ///     .with_social_coefficient(1.5)
    ///     .solve(guess)
    ///     .unwrap();
    /// ```
    pub fn with_social_coefficient(&mut self, social_coefficient: T) -> &mut Self {
        self.social_coefficient = social_coefficient;
        self
    }

    /// Set the tolerance of the optimiser. If many (exact amount specified in implementation) of the previous
    /// best global minimums found are in absolve value less than the tolerance then the optimiser terminates
    /// and returns the best value found
    /// ## Examples
    /// ```
    /// # use eqsolver::global_optimisers::ParticleSwarm;
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
    /// # let bounds = SVector::repeat(100.);
    /// let optimised_position = ParticleSwarm::new(f, -bounds, bounds)
    ///     .with_tolerance(1e-3)
    ///     .solve(guess)
    ///     .unwrap();
    /// ```
    pub fn with_tolerance(&mut self, tolerance: T) -> &mut Self {
        self.tolerance = tolerance;
        self
    }

    /// Set the number of particles in the optimiser. For more information about the effect of this parameter,
    /// see [Particle Swarm Optimization](https://en.wikipedia.org/wiki/Particle_swarm_optimization).
    /// ## Examples
    /// ```
    /// # use eqsolver::global_optimisers::ParticleSwarm;
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
    /// # let bounds = SVector::repeat(100.);
    /// let optimised_position = ParticleSwarm::new(f, -bounds, bounds)
    ///     .with_particle_count(100)
    ///     .solve(guess)
    ///     .unwrap();
    /// ```
    pub fn with_particle_count(&mut self, particle_count: usize) -> &mut Self {
        self.particle_count = particle_count;
        self
    }

    /// Set the maximum number of iterations of the optimiser. The optimiser returns the best value found after
    /// it has iterated that amount of number of iterations.
    /// ## Examples
    /// ```
    /// # use eqsolver::global_optimisers::ParticleSwarm;
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
    /// # let bounds = SVector::repeat(100.);
    /// let optimised_position = ParticleSwarm::new(f, -bounds, bounds)
    ///     .with_iter_max(100)
    ///     .solve(guess)
    ///     .unwrap();
    /// ```
    pub fn with_iter_max(&mut self, iter_max: usize) -> &mut Self {
        self.iter_max = iter_max;
        self
    }

    /// Optimises the function using a given initial value (or guess) by returning an approximation of the global
    /// minimum of the objective function within the bounds.
    /// ## Examples
    /// ```
    /// # use eqsolver::global_optimisers::ParticleSwarm;
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
    /// # let bounds = SVector::repeat(100.);
    /// let optimised_position = ParticleSwarm::new(f, -bounds, bounds)
    ///     .solve(guess)
    ///     .unwrap();
    /// ```
    pub fn solve(&self, x0: VectorType<T, D>) -> SolverResult<VectorType<T, D>> {
        let mut global_best_position = x0.clone();
        let mut global_best_cost = (self.f)(global_best_position.clone());
        let mut previous_best = CircularArray::fill(T::infinity());

        let mut particles: Vec<Particle<T, D>> = (0..self.particle_count)
            .map(|_| {
                let mut position = x0.clone();
                position
                    .iter_mut()
                    .zip(self.position_distributions.iter())
                    .for_each(|(v, dist)| *v = dist.sample(&mut thread_rng()));

                let mut velocity = x0.clone();
                velocity
                    .iter_mut()
                    .zip(self.velocity_distributions.iter())
                    .for_each(|(v, dist)| *v = dist.sample(&mut thread_rng()));

                let cost = (self.f)(position.clone());
                if cost < global_best_cost {
                    previous_best.push(global_best_cost);
                    global_best_position = position.clone();
                    global_best_cost = cost;
                }

                Particle {
                    position: position.clone(),
                    velocity,
                    best_position: position,
                    best_cost: cost,
                }
            })
            .collect();

        let u01 = Uniform::new_inclusive(T::zero(), T::one());

        for _ in 0..self.iter_max {
            if self.stalled_too_long(global_best_cost, &previous_best) {
                break;
            }

            for Particle {
                position,
                velocity,
                best_position,
                best_cost,
            } in particles.iter_mut()
            {
                for (d, v_d) in velocity.iter_mut().enumerate() {
                    let rp = u01.sample(&mut thread_rng());
                    let rg = u01.sample(&mut thread_rng());
                    *v_d = self.inertia_weight * *v_d
                        + self.cognitive_coefficient * rp * (best_position[d] - position[d])
                        + self.social_coefficient * rg * (global_best_position[d] - position[d]);
                }
                *position += &*velocity;

                let cost = (self.f)(position.clone());

                if cost < *best_cost {
                    *best_position = position.clone();
                    *best_cost = cost;
                    if *best_cost < global_best_cost {
                        previous_best.push(global_best_cost);

                        global_best_position = position.clone();
                        global_best_cost = *best_cost;
                    }
                }
            }
        }

        Ok(global_best_position)
    }

    fn stalled_too_long(
        &self,
        global_best: T,
        previous: &CircularArray<T, DEFAULT_STALL_ITERATIONS>,
    ) -> bool {
        previous
            .iter()
            .all(|&x| Float::abs(x - global_best) < self.tolerance)
    }
}
