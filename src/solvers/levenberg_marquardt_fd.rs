use std::marker::PhantomData;

use crate::{SolverResult, DEFAULT_TOL};

use super::levenberg_marquardt::{DEFAULT_DAMPING_DECAY_FACTOR, DEFAULT_DAMPING_INITIAL_VALUE};
use super::{SolverError, VectorType, DEFAULT_ITERMAX};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim};
use nalgebra::{ComplexField, U1};
use num_traits::{Float, Signed};

/// # Levenberg-Marquardt
///
/// Solves `x` in a system of equation `F(x) = 0` (where `F: Rm ⟶ Rn`) by minimizing `||F(x)||` using the Levenberg-Marquardt Algorithm (LMA)
///
/// **Default Tolerance:** 1e-6
///
/// **Default Max Iterations:** 50
///
/// The damping in LMA is done using two parameters, the intial value μ_0 and decaying factor β, the next damping factor is calculated as
///
/// μ_{k+1} = μ_k / β if ||F(x_k + Δx_k)|| < ||F(x_k)||
///
/// μ_{k+1} = μ_k * β if otherwise
///
/// In other words, the damping factor decreases when the change in the objective function is lower than the previous step
/// and otherwise it increases. The default parameters are usually set to μ_0 = 0.01, and β = 10. For more details see this
/// [article](https://www.mathworks.com/help/optim/ug/least-squares-model-fitting-algorithms.html) by MathWorks.
pub struct LevenbergMarquardtFD<T, R, C, F> {
    f: F,
    tolerance: T,
    h: T,
    iter_max: usize,
    mu_0: T,
    beta: T,
    r_phantom: PhantomData<R>,
    c_phantom: PhantomData<C>,
}

impl<T, R, C, F> LevenbergMarquardtFD<T, R, C, F>
where
    T: Float + ComplexField<RealField = T> + Signed,
    R: Dim,
    C: Dim,
    F: Fn(VectorType<T, C>) -> VectorType<T, R>,
    DefaultAllocator: Allocator<C, R>
        + Allocator<C>
        + Allocator<R>
        + Allocator<U1, R>
        + Allocator<C, U1>
        + Allocator<U1, C>
        + Allocator<R, C>
        + Allocator<C, C>,
{
    /// Create a new instance of the algorithm
    ///
    /// Instantiates the Levenberg-Marquardt algorithm using the system of equation `F` and its Jacobian `J`.
    pub fn new(f: F) -> Self {
        Self {
            f,
            tolerance: T::from(DEFAULT_TOL).unwrap(),
            h: Float::sqrt(T::epsilon()),
            mu_0: T::from(DEFAULT_DAMPING_INITIAL_VALUE).unwrap(),
            beta: T::from(DEFAULT_DAMPING_DECAY_FACTOR).unwrap(),
            iter_max: DEFAULT_ITERMAX,
            r_phantom: PhantomData,
            c_phantom: PhantomData,
        }
    }

    /// Updates the solver's tolerance (Magnitude of Error).
    ///
    /// **Default Tolerance:** `1e-6`
    pub fn with_tol(&mut self, tol: T) -> &mut Self {
        self.tolerance = tol;
        self
    }

    /// Updates the solver's amount of iterations done before terminating the iteration
    ///
    /// **Default Max Iterations:** `50`
    pub fn with_itermax(&mut self, max: usize) -> &mut Self {
        self.iter_max = max;
        self
    }

    /// Updates the solver's initial damping factor
    ///
    /// **Default Initial Damping Factor:** `0.1`
    pub fn with_intial_damping_factor(&mut self, mu_0: T) -> &mut Self {
        self.mu_0 = mu_0;
        self
    }

    /// Updates the solver's damping decay factor
    ///
    /// **Default Damping Decay Factor:** `10`
    pub fn with_damping_decay_factor(&mut self, beta: T) -> &mut Self {
        self.beta = beta;
        self
    }

    /// Updates the step length used in the finite difference
    ///
    /// **Default Step length for Finite Difference:** `√(Machine Epsilon)`
    pub fn with_fd_step_length(&mut self, h: T) -> &mut Self {
        self.h = h;
        self
    }

    /// Run the algorithm
    ///
    /// Finds `x` such that `||F(x)||` is minimized where `F` is the overdetermined system of equations.
    pub fn solve(&self, mut x0: VectorType<T, C>) -> SolverResult<VectorType<T, C>> {
        let mut dv = x0.clone().add_scalar(T::max_value()); // We assume error vector is infinitely long at the start
        let mut identity = x0.clone() * x0.clone().transpose();
        identity.fill_with_identity();
        let mut iter = 1;
        let mut damping = self.mu_0;
        let mut fx = (self.f)(x0.clone());
        let zero = (fx.clone() * x0.clone().transpose()).scale(T::zero());

        // Levenberg-Marquardt Iteration
        while dv.abs().max() > self.tolerance && iter <= self.iter_max {
            let mut j = zero.clone(); // Jacobian, will be approximated below

            // Approximate the Jacobian using forward finite difference
            for i in 0..j.ncols() {
                let mut x_h = x0.clone();
                x_h[i] = x_h[i] + self.h; // Add derivative step to specific parameter
                let df = ((self.f)(x_h) - fx.clone()) / self.h; // Derivative of F with respect to x_i
                for k in 0..j.nrows() {
                    j[(k, i)] = df[k];
                }
            }

            let jt = j.clone().transpose();
            let d = jt.clone() * j + identity.clone() * damping;
            let Some(j_inv) = d.clone().try_inverse() else {
                return Err(SolverError::BadJacobian);
            };

            dv = j_inv * -jt * fx.clone();
            x0 += dv.clone();
            let fx_next = (self.f)(x0.clone());

            if fx_next.norm() < fx.norm() {
                damping /= self.beta;
            } else {
                damping *= self.beta;
            }

            fx = fx_next;

            iter += 1;
        }

        if iter >= self.iter_max {
            return Err(SolverError::MaxIterReached);
        }

        Ok(x0)
    }
}
