#[allow(dead_code)]
use nalgebra::ComplexField;
use nalgebra::{SMatrix, SVector};
use num_traits::{Signed, Float};
use super::{SolverError, DEFAULT_ITERMAX, DEFAULT_TOL};

/// # Gauss-Newton
/// 
/// Solves x in a system of equation F(x) = 0 (where F: Rm ⟶ Rn) by minimizing ||F(x)|| in a Least Square Sense using The Gauss-Newton algorithm ([Wikipedia](https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm)). This struct uses a closure that takes a nalgebra vector (size m) and outputs a nalgebra vector (size n) to represent the overdetermined system of equations. The struct also uses the Jacobian of F, that is represented by a closure taking a nalgebra vector (size m) and outputs a nalgebra matrix of size n x m.
/// 
/// **Default Tolerance:** 1e-6
/// 
/// **Default Max Iterations:** 50
pub struct GaussNewton<T, F, J, const R: usize, const C: usize>
where
    T: Float + ComplexField + Signed,
    F: Fn(SVector<T, C>) -> SVector<T, R>,
    J: Fn(SVector<T, C>) -> SMatrix<T, R, C>,
{
    f: F,
    j: J,
    tolerance: T,
    iter_max: usize,
}

impl<T, F, J, const R: usize, const C: usize> GaussNewton<T, F, J, R, C>
where
    T: Float + ComplexField + Signed,
    F: Fn(SVector<T, C>) -> SVector<T, R>,
    J: Fn(SVector<T, C>) -> SMatrix<T, R, C>,
{
    /// Create a new instance of the algorithm
    /// 
    /// Instantiates the Gauss-Newton algorithm using the system of equation F and its Jacobian J.
    pub fn new(f: F, j: J) -> Self {
        Self {
            f,
            j,
            tolerance: T::from(DEFAULT_TOL).unwrap(),
            iter_max: DEFAULT_ITERMAX,
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

    /// Run the algorithm
    /// 
    /// Finds x such that ||F(x)|| is minimized where F is the overdetermined system of equations. 
    pub fn solve(&self, mut x0: SVector<T, C>) -> Result<SVector<T, C>, SolverError> {
        let mut dv: SVector<T, C> = SVector::repeat(T::max_value()); // We assume error vector is infinitely long at the start
        let mut iter = 1;

        // Gauss Newton Iteration
        while dv.abs().max() > self.tolerance && iter < self.iter_max {
            let j = (self.j)(x0);
            let jt = j.transpose();
            if let Some(jtj_inv) = (jt*j).try_inverse() {
                dv = jtj_inv * jt * (self.f)(x0);
                x0 = x0 - dv;
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