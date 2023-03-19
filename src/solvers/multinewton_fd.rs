use std::marker::PhantomData;

use super::{MatrixType, SolverError, VectorType, DEFAULT_ITERMAX, DEFAULT_TOL};
#[allow(dead_code)]
use nalgebra::ComplexField;
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, DimName};
use num_traits::{Float, Signed};

/// # Multivarite Newton-Raphson with Finite Differences
/// 
/// This struct finds x such that F(x) = 0 where F: Rn ⟶ Rn is a vectorial function. The vector x is given as a nalgebra vector and the solution will be of the same dimension as the input vector. This struct approximates the Jacobian using finite differences. This struct uses the Newton-Raphson method for system of equations ([Wikipedia](https://en.wikipedia.org/wiki/Newton%27s_method#k_variables,_k_functions)).
///
/// **Default Tolerance:** 1e-6
///
/// **Default Max Iterations:** 50
/// 
/// ## Examples
/// 
/// ```
/// use eqsolver::multivariable::MultiVarNewtonFD;
/// use nalgebra::{Vector2, Matrix2};
/// // Vectorial Function (x, y) ↦ (x^2-y-1, xy - 2). Want to solve x^2 - y = 1 and xy = 2
/// let F = |v: Vector2<f64>| Vector2::new(v[0].powi(2) - v[1] - 1., v[0] * v[1] - 2.);
///
/// // Solved analytically but form was ugly so here is approximation
/// const SOLUTION: Vector2<f64> = Vector2::new(1.521379706804567569604081, 1.314596212276751981650111);
///
/// let solution = MultiVarNewtonFD::new(F)
///                 .with_tol(1e-6)
///                 .solve(Vector2::new(1., 1.))
///                 .unwrap();
///
/// assert!((solution - SOLUTION).norm() <= 1e-6);
/// ```
pub struct MultiVarNewtonFD<T, D, F>
where
    T: Float + ComplexField + Signed,
    D: Dim + DimName,
    F: Fn(VectorType<T, D>) -> VectorType<T, D>,
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
{
    f: F,
    h: T,
    tolerance: T,
    iter_max: usize,
    d_phantom: PhantomData<D>,
}

impl<T, D, F> MultiVarNewtonFD<T, D, F>
where
    T: Float + ComplexField + Signed,
    D: Dim + DimName,
    F: Fn(VectorType<T, D>) -> VectorType<T, D>,
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
{
    /// Set up the solver
    /// 
    /// Instantiate the solver using the given vectorial function `F` that is closure that takes a nalgebra vector and outputs a nalgebra vector of the same size.
    /// 
    /// ## Examples
    /// ```
    /// use eqsolver::multivariable::MultiVarNewtonFD;
    /// use nalgebra::{Vector2, Matrix2};
    /// // Vectorial Function (x, y) ↦ (x^2-y-1, xy - 2). Want to solve x^2 - y = 1 and xy = 2
    /// let F = |v: Vector2<f64>| Vector2::new(v[0].powi(2) - v[1] - 1., v[0] * v[1] - 2.);
    ///
    /// let solution = MultiVarNewtonFD::new(F);
    /// ```
    pub fn new(f: F) -> Self {
        Self {
            f,
            h: Float::sqrt(T::epsilon()),
            tolerance: T::from(DEFAULT_TOL).unwrap(),
            iter_max: DEFAULT_ITERMAX,
            d_phantom: PhantomData
        }
    }

    /// Updates the solver's tolerance (Magnitude of Error).
    ///
    /// **Default Tolerance:** 1e-6
    /// 
    /// ## Examples
    /// ```
    /// use eqsolver::multivariable::MultiVarNewtonFD;
    /// use nalgebra::{Vector2, Matrix2};
    /// // Vectorial Function (x, y) ↦ (x^2-y-1, xy - 2). Want to solve x^2 - y = 1 and xy = 2
    /// let F = |v: Vector2<f64>| Vector2::new(v[0].powi(2) - v[1] - 1., v[0] * v[1] - 2.);
    ///
    /// // Solved analytically but form was ugly so here is approximation
    /// const SOLUTION: Vector2<f64> = Vector2::new(1.521379706804567569604081, 1.314596212276751981650111);
    ///
    /// let solution = MultiVarNewtonFD::new(F)
    ///                 .with_tol(1e-12)
    ///                 .solve(Vector2::new(1., 1.))
    ///                 .unwrap();
    ///
    /// assert!((solution - SOLUTION).norm() <= 1e-12);
    /// ```
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
    /// 
    /// ## Examples
    /// ```
    /// # use eqsolver::multivariable::MultiVarNewtonFD;
    /// # use nalgebra::{Vector2, Matrix2};
    /// // Vectorial Function (x, y) ↦ (x^2-y-1, xy - 2). Want to solve x^2 - y = 1 and xy = 2
    /// # let F = |v: Vector2<f64>| Vector2::new(v[0].powi(2) - v[1] - 1., v[0] * v[1] - 2.);
    ///
    /// // Solved analytically but form was ugly so here is approximation
    /// # const SOLUTION: Vector2<f64> = Vector2::new(1.521379706804567569604081, 1.314596212276751981650111);
    ///
    /// let solution = MultiVarNewtonFD::new(F)
    ///                 .with_fd_step_length(1e-3)
    ///                 .solve(Vector2::new(1., 1.));
    ///
    /// # assert!((solution.unwrap() - SOLUTION).norm() <= 1e-6);
    pub fn with_fd_step_length(&mut self, h: T) -> &mut Self {
        self.h = h;
        self
    }

    /// Solves x in f(x) = 0 where f is the stored function.
    /// 
    /// ## Examples
    /// 
    /// ### Solution working
    /// 
    /// ```
    /// use eqsolver::multivariable::MultiVarNewtonFD;
    /// use nalgebra::{Vector2, Matrix2};
    /// // Vectorial Function (x, y) ↦ (x^2-y-1, xy - 2). Want to solve x^2 - y = 1 and xy = 2
    /// let F = |v: Vector2<f64>| Vector2::new(v[0].powi(2) - v[1] - 1., v[0] * v[1] - 2.);
    ///
    /// // Solved analytically but form was ugly so here is approximation
    /// const SOLUTION: Vector2<f64> = Vector2::new(1.521379706804567569604081, 1.314596212276751981650111);
    ///
    /// let solution = MultiVarNewtonFD::new(F)
    ///                 .with_tol(1e-6)
    ///                 .solve(Vector2::new(1., 1.))
    ///                 .unwrap();
    ///
    /// assert!((solution - SOLUTION).norm() <= 1e-6);
    /// ```
    /// 
    /// ### Bad Jacobian Error
    /// 
    /// ```
    /// use eqsolver::{multivariable::MultiVarNewtonFD, SolverError};
    /// use nalgebra::{Vector2, Matrix2};
    /// // Vectorial Function (x, y) ↦ (x^2-y-1, xy - 2). Want to solve x^2 - y = 1 and xy = 2
    /// let F = |v: Vector2<f64>| Vector2::new(v[0] + v[1], v[0]*v[1]);
    ///
    /// let solution = MultiVarNewtonFD::new(F)
    ///                 .with_tol(1e-6)
    ///                 .solve(Vector2::new(0., 0.)); // This will make the Jacobian very close to singular.  [1 1]
    ///                                                                                                    // [0 0]
    ///
    /// assert_eq!(solution.err().unwrap(), SolverError::BadJacobian);
    /// ```
    pub fn solve(&self, mut x0: VectorType<T, D>) -> Result<VectorType<T, D>, SolverError> {
        let mut dv = VectorType::<T, D>::repeat(T::max_value()); // We assume error vector is infinitely long at the start
        let mut iter = 1;
        let dim = x0.nrows();

        while dv.abs().max() > self.tolerance && iter < self.iter_max {
            let mut j: MatrixType<T, D> = MatrixType::<T, D>::zeros(); // Jacobian, will be approximated below
            let fx = (self.f)(x0.clone());

            // Approximate Jacobi using forward finite difference
            for i in 0..dim {
                let mut x_h = x0.clone();
                x_h[i] = x_h[i] + self.h; // Add derivative step to specific parameter
                let df = ((self.f)(x_h) - fx.clone()) / self.h; // Derivative of F with respect to x_i
                for k in 0..dim {
                    j[(k, i)] = df[k];
                }
            }

            // Newton-Raphson iteration
            if let Some(j_inv) = j.try_inverse() {
                dv = j_inv * fx;
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
