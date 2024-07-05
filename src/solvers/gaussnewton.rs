use std::marker::PhantomData;

use crate::SolverResult;

use super::{MatrixType, SolverError, VectorType, DEFAULT_ITERMAX, DEFAULT_TOL};
use nalgebra::ComplexField;
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim};
use num_traits::{Float, Signed};

/// # Gauss-Newton
///
/// Solves `x` in a system of equation `F(x) = 0` (where `F: Rm ⟶ Rn`) by minimizing `||F(x)||` in a least square sense using The Gauss-Newton algorithm ([Wikipedia](https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm)). This struct uses a closure that takes a nalgebra vector (size `m`) and outputs a nalgebra vector (size `n`) to represent the overdetermined system of equations. The struct also uses the Jacobian of `F`, that is represented by a closure taking a nalgebra vector (size `m`) and outputs a nalgebra matrix of size `n x m`.
///
/// **Default Tolerance:** 1e-6
///
/// **Default Max Iterations:** 50
pub struct GaussNewton<T, R, C, F, J> {
    f: F,
    j: J,
    tolerance: T,
    iter_max: usize,
    r_phantom: PhantomData<R>,
    c_phantom: PhantomData<C>,
}

impl<T, R, C, F, J> GaussNewton<T, R, C, F, J>
where
    T: Float + ComplexField<RealField = T> + Signed,
    R: Dim,
    C: Dim,
    F: Fn(VectorType<T, C>) -> VectorType<T, R>,
    J: Fn(VectorType<T, C>) -> MatrixType<T, R, C>,
    DefaultAllocator:
        Allocator<C> + Allocator<R> + Allocator<R, C> + Allocator<C, R> + Allocator<C, C>,
{
    /// Create a new instance of the algorithm
    ///
    /// Instantiates the Gauss-Newton algorithm using the system of equation `F` and its Jacobian `J`.
    pub fn new(f: F, j: J) -> Self {
        Self {
            f,
            j,
            tolerance: T::from(DEFAULT_TOL).unwrap(),
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

    /// Run the algorithm
    ///
    /// Finds `x` such that `||F(x)||` is minimized where `F` is the overdetermined system of equations.
    pub fn solve(&self, mut x0: VectorType<T, C>) -> SolverResult<VectorType<T, C>> {
        let mut dv = x0.clone().add_scalar(T::max_value()); // We assume error vector is infinitely long at the start
        let mut iter = 1;

        // Gauss-Newton Iteration
        while dv.abs().max() > self.tolerance && iter <= self.iter_max {
            let j = (self.j)(x0.clone());
            let jt = j.transpose();
            if let Some(jtj_inv) = (jt.clone() * j).try_inverse() {
                dv = jtj_inv * jt * (self.f)(x0.clone());
                x0 = x0 - dv.clone();
                iter += 1;
            } else {
                return Err(SolverError::BadJacobian);
            }
        }

        if iter >= self.iter_max {
            return Err(SolverError::MaxIterReached);
        }

        Ok(x0)
    }
}
