#[allow(dead_code)]
use nalgebra::ComplexField;
use nalgebra::{SMatrix, SVector};
use num_traits::{Float, Signed};
use super::{SolverError, DEFAULT_ITERMAX, DEFAULT_TOL};

/// # Gauss-Newton with Finite Differences 
/// 
/// Solves x in a system of equation F(x) = 0 (where F: Rm ⟶ Rn) by minimizing ||F(x)|| in a Least Square Sense using The Gauss-Newton algorithm ([Wikipedia](https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm)). This struct uses a closure that takes a nalgebra vector (size m) and outputs a nalgebra vector (size n) to represent the overdetermined system of equations. The struct approximates the Jacobian using finite differences.
/// 
/// **Default Tolerance:** 1e-6
/// 
/// **Default Max Iterations:** 50
pub struct GaussNewtonFD<T, F, const R: usize, const C: usize>
where
    T: Float + ComplexField + Signed,
    F: Fn(SVector<T, R>) -> SVector<T, C>,
{
    f: F,
    h: T,
    tolerance: T,
    iter_max: usize,
}

impl<T, F, const R: usize, const C: usize> GaussNewtonFD<T, F, R, C>
where
    T: Float + ComplexField + Signed,
    F: Fn(SVector<T, R>) -> SVector<T, C>,
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
    pub fn solve(&self, mut x0: SVector<T, R>) -> Result<SVector<T, R>, SolverError> {
        let mut dv: SVector<T, R> = SVector::repeat(T::max_value()); // We assume error vector is infinitely long at the start
        let mut iter = 1;

        while dv.abs().max() > self.tolerance && iter < self.iter_max {
            let mut j: SMatrix<T, C, R> = SMatrix::zeros(); // Jacobian, will be approximated below
            let fx = (self.f)(x0);

            // Approximate Jacobi using forward finite difference
            for i in 0..R {
                let mut x_h = x0;
                x_h[i] = x_h[i] + self.h; // Add derivative step to specific parameter
                let df = ((self.f)(x_h) - fx)/self.h; // Derivative of F with respect to x_i
                for k in 0..C {
                    j[(k, i)] = df[k];
                }
            }

            // Gauss Newton Iteration
            let jt = j.transpose();
            if let Some(jjt_inv) = (jt*j).try_inverse() {
                dv = jjt_inv*jt*fx;
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
