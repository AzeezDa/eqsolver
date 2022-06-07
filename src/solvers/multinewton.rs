use nalgebra::{SVector, SMatrix, Scalar, ComplexField};
use num_traits::{Float, Signed};

use super::{DEFAULT_TOL, DEFAULT_ITERMAX, SolverError};

/// # `MultiVarNewton`
/// 
/// This struct finds x such that F(x) = 0 where F: Rn ⟶ Rn is a vectorial function. The vector x is given as a nalgebra vector and the solution will be of the same dimension as the input vector. This struct requires the Jacobian Matrix of F this is given as nalgebra Matrix. This struct uses the Newton-Raphson method for system of equations ([Wikipedia](https://en.wikipedia.org/wiki/Newton%27s_method#k_variables,_k_functions)).
/// 
/// **Default Tolerance:** 1e-6
/// 
/// **Default Max Iterations:** 50
/// 
/// # Examples
/// 
/// ```
/// use eqsolver::solvers::multivariable::MultiVarNewton;
/// use nalgebra::{Vector2, Matrix2};
/// // Vectorial Function (x, y) ↦ (x^2-y-1, xy - 2). Want to solve x^2 - y = 1 and xy = 2
/// let F = |v: Vector2<f64>| Vector2::new(v[0].powi(2) - v[1] - 1., v[0] * v[1] - 2.);
/// 
/// // Jacobian of F
/// let J = |v: Vector2<f64>| Matrix2::new(2. * v[0], -1., 
///                                             v[1], v[0]);
/// 
/// // Solved analytically but form was ugly so here is approximation
/// const SOLUTION: Vector2<f64> = Vector2::new(1.521379706804567569604081, 1.314596212276751981650111);
/// 
/// let solution = MultiVarNewton::new(F, J)
///                 .with_tol(1e-6)
///                 .solve(Vector2::new(1., 1.))
///                 .unwrap();
/// 
/// assert!((solution - SOLUTION).norm() <= 1e-6);
/// ```
pub struct MultiVarNewton<T, F, J, const S: usize>
where
    T: Float + Scalar + ComplexField + Signed,
    J: Fn(SVector<T, S>) -> SMatrix<T, S, S>,
    F: Fn(SVector<T, S>) -> SVector<T, S>,
{
    f: F,
    j: J,
    tolerance: T,
    iter_max: usize,
}

impl<T, F, J, const S: usize> MultiVarNewton<T, F, J, S>
where
    T: Float + Scalar + ComplexField + Signed,
    J: Fn(SVector<T, S>) -> SMatrix<T, S, S>,
    F: Fn(SVector<T, S>) -> SVector<T, S>,
{

    /// # `new`
    /// Instantiate the solver using the given vectorial function `F` that is closure that takes a nalgebra vector and outputs a nalgebra vector of the same size.
    /// This also takes the Jacobian of `F` as `J` that is a closure that takes a nalgebra vector and outputs its jacobian as nalgebra matrix.
    /// 
    /// # Examples
    /// ```
    /// use eqsolver::solvers::multivariable::MultiVarNewton;
    /// use nalgebra::{Vector2, Matrix2};
    /// // Vectorial Function (x, y) ↦ (x^2-y-1, xy - 2). Want to solve x^2 - y = 1 and xy = 2
    /// let F = |v: Vector2<f64>| Vector2::new(v[0].powi(2) - v[1] - 1., v[0] * v[1] - 2.);
    /// 
    /// // Jacobian of F
    /// let J = |v: Vector2<f64>| Matrix2::new(2. * v[0], -1., 
    ///                                             v[1], v[0]);
    /// 
    /// let solution = MultiVarNewton::new(F, J);
    /// ```
    pub fn new(f: F, j: J) -> Self {
        Self {
            f,
            j,
            tolerance: T::from(DEFAULT_TOL).unwrap(),
            iter_max: DEFAULT_ITERMAX,
        }
    }

    /// # 'with_tol`
    /// Updates the solver's tolerance (Magnitude of Error).
    /// 
    /// **Default Tolerance:** 1e-6
    /// 
    /// # Examples
    /// ```
    /// use eqsolver::solvers::multivariable::MultiVarNewton;
    /// use nalgebra::{Vector2, Matrix2};
    /// // Vectorial Function (x, y) ↦ (x^2-y-1, xy - 2). Want to solve x^2 - y = 1 and xy = 2
    /// let F = |v: Vector2<f64>| Vector2::new(v[0].powi(2) - v[1] - 1., v[0] * v[1] - 2.);
    /// 
    /// // Jacobian of F
    /// let J = |v: Vector2<f64>| Matrix2::new(2. * v[0], -1., 
    ///                                             v[1], v[0]);
    /// 
    /// // Solved analytically but form was ugly so here is approximation
    /// const SOLUTION: Vector2<f64> = Vector2::new(1.521379706804567569604081, 1.314596212276751981650111);
    /// 
    /// let solution = MultiVarNewton::new(F, J)
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

    /// # `with_itermax`
    /// Updates the solver's amount of iterations done before terminating the iteration
    /// 
    /// **Default Max Iterations:** 50
    pub fn with_itermax(&mut self, max: usize) -> &mut Self {
        self.iter_max = max;
        self
    }

    /// # `solve`
    /// Solves x in F(x) = 0 where F is the stored function.
    /// 
    /// # Examples
    /// 
    /// ## Solution working
    /// 
    /// ```
    /// use eqsolver::solvers::multivariable::MultiVarNewton;
    /// use nalgebra::{Vector2, Matrix2};
    /// // Vectorial Function (x, y) ↦ (x^2-y-1, xy - 2). Want to solve x^2 - y = 1 and xy = 2
    /// let F = |v: Vector2<f64>| Vector2::new(v[0].powi(2) - v[1] - 1., v[0] * v[1] - 2.);
    /// 
    /// // Jacobian of F
    /// let J = |v: Vector2<f64>| Matrix2::new(2. * v[0], -1., 
    ///                                             v[1], v[0]);
    /// 
    /// // Solved analytically but form was ugly so here is approximation
    /// const SOLUTION: Vector2<f64> = Vector2::new(1.521379706804567569604081, 1.314596212276751981650111);
    /// 
    /// let solution = MultiVarNewton::new(F, J)
    ///                 .with_tol(1e-6)
    ///                 .solve(Vector2::new(1., 1.))
    ///                 .unwrap();
    /// 
    /// assert!((solution - SOLUTION).norm() <= 1e-6);
    /// ```
    /// 
    /// ## Bad Jacobian Error
    /// ```
    /// use eqsolver::solvers::{multivariable::MultiVarNewton, SolverError};
    /// use nalgebra::{Vector2, Matrix2};
    /// // Vectorial Function (x, y) ↦ (x^2-y-1, xy - 2). Want to solve x^2 - y = 1 and xy = 2
    /// let F = |v: Vector2<f64>| Vector2::new(v[0] + v[1], v[0]*v[1]);
    /// 
    /// // Jacobian of F
    /// let J = |v: Vector2<f64>| Matrix2::new(1., 1., 
    ///                                        v[1], v[0]);
    /// 
    /// let solution = MultiVarNewton::new(F, J)
    ///                 .with_tol(1e-6)
    ///                 .solve(Vector2::new(0., 0.)); // This will make the Jacobian singular.  [1 1]
    ///                                                                                     // [0 0]
    /// 
    /// assert_eq!(solution.err().unwrap(), SolverError::BadJacobian);
    /// ```
    pub fn solve(&self, mut x0: SVector<T, S>) -> Result<SVector<T, S>, SolverError> {
        let mut dv: SVector<T, S> = SVector::repeat(T::max_value()); // We assume error vector is infinitely long at the start
        let mut iter = 1;

        // Newton-Raphson Iteration
        while dv.abs().max() > self.tolerance && iter < self.iter_max {
            if let Some(j_inv) = (self.j)(x0).try_inverse() {
                dv = j_inv * (self.f)(x0);
                x0 = x0 - dv;
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
