use std::marker::PhantomData;

use super::{SolverError, VectorType, DEFAULT_ITERMAX, DEFAULT_TOL};
#[allow(dead_code)]
use nalgebra::ComplexField;
use nalgebra::{allocator::Allocator, Const, DefaultAllocator, Dim};
use num_traits::{Float, Signed};

/// # Gauss-Newton with Finite Differences
///
/// Solves x in a system of equation F(x) = 0 (where F: Rm ⟶ Rn) by minimizing ||F(x)|| in a Least Square Sense using The Gauss-Newton algorithm ([Wikipedia](https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm)). This struct uses a closure that takes a nalgebra vector (size m) and outputs a nalgebra vector (size n) to represent the overdetermined system of equations. The struct approximates the Jacobian using finite differences.
///
/// **Default Tolerance:** 1e-6
///
/// **Default Max Iterations:** 50
pub struct GaussNewtonFD<T, R, C, F>
where
    T: Float + ComplexField<RealField = T> + Signed,
    R: Dim,
    C: Dim,
    F: Fn(VectorType<T, C>) -> VectorType<T, R>,
    DefaultAllocator: Allocator<T, C, R>
        + Allocator<T, C>
        + Allocator<T, R>
        + Allocator<T, Const<1>, R>
        + Allocator<T, C, Const<1>>
        + Allocator<T, Const<1>, C>
        + Allocator<T, R, C>
        + Allocator<T, C, C>
{
    f: F,
    h: T,
    tolerance: T,
    iter_max: usize,
    r_phantom: PhantomData<R>,
    c_phantom: PhantomData<C>,
}

impl<T, R, C, F> GaussNewtonFD<T, R, C, F>
where
    T: Float + ComplexField<RealField = T> + Signed,
    R: Dim,
    C: Dim,
    F: Fn(VectorType<T, C>) -> VectorType<T, R>,
    DefaultAllocator: Allocator<T, C, R>
        + Allocator<T, C>
        + Allocator<T, R>
        + Allocator<T, Const<1>, R>
        + Allocator<T, C, Const<1>>
        + Allocator<T, Const<1>, C>
        + Allocator<T, R, C>
        + Allocator<T, C, C>
{
    /// Create a new instance of the algorithm
    ///
    /// Instantiates the Gauss-Newton algorithm using the system of equation F and its Jacobian J.
    pub fn new(f: F) -> Self {
        Self {
            f,
            h: Float::sqrt(T::epsilon()),
            tolerance: T::from(DEFAULT_TOL).unwrap(),
            iter_max: DEFAULT_ITERMAX,
            r_phantom: PhantomData,
            c_phantom: PhantomData,
        }
    }

    /// Updates the solver's tolerance (Magnitude of Error).
    ///
    /// **Default Tolerance:** 1e-6
    pub fn with_tol(&mut self, tol: T) -> &mut Self {
        self.tolerance = tol;
        self
    }

    /// Updates the solver's amount of iterations done before terminating the iteration
    ///
    /// **Default Max Iterations:** 50
    pub fn with_itermax(&mut self, max: usize) -> &mut Self {
        self.iter_max = max;
        self
    }

    /// Updates the step length used in the finite difference
    ///
    /// **Default Step length for Finite Difference:** √(Machine Epsilon)
    pub fn with_fd_step_length(&mut self, h: T) -> &mut Self {
        self.h = h;
        self
    }

    /// Run the algorithm
    ///
    /// Finds x such that ||F(x)|| is minimized where F is the overdetermined system of equations.
    pub fn solve(&self, mut x0: VectorType<T, C>) -> Result<VectorType<T, C>, SolverError> {
        let mut dv = x0.clone().add_scalar(T::max_value()); // We assume error vector is infinitely long at the start
        let mut iter = 1;
        let fx = (self.f)(x0.clone());
        let zero = (fx * x0.clone().transpose()).scale(T::zero());

        while dv.abs().max() > self.tolerance && iter < self.iter_max {
            let mut j = zero.clone(); // Jacobian, will be approximated below
            let fx = (self.f)(x0.clone());

            // Approximate Jacobi using forward finite difference
            for i in 0..j.ncols() {
                let mut x_h = x0.clone();
                x_h[i] = x_h[i] + self.h; // Add derivative step to specific parameter
                let df = ((self.f)(x_h) - fx.clone()) / self.h; // Derivative of F with respect to x_i
                for k in 0..j.nrows() {
                    j[(k, i)] = df[k];
                }
            }

            // Gauss Newton Iteration
            let jt = j.transpose();
            if let Some(jjt_inv) = (jt.clone() * j).try_inverse() {
                dv = jjt_inv * jt * fx;
                x0 = x0 - dv.clone();
                iter += 1;
            } else {
                return Err(SolverError::BadJacobian);
            }
        }

        if iter > self.iter_max {
            return Err(SolverError::MaxIterReached);
        }

        Ok(x0)
    }
}
