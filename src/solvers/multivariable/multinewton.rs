use crate::{SolverError, SolverResult, DEFAULT_ITERMAX, DEFAULT_TOL, MatrixType, VectorType};
use nalgebra::{allocator::Allocator, ComplexField, DefaultAllocator, Dim, Scalar, UniformNorm};
use num_traits::{Float, Signed};
use std::marker::PhantomData;

/// # Multivariate Newton-Raphson
///
/// This struct finds `x` such that `F(x) = 0` where `F: Rn ⟶ Rn` is a vectorial function. The vector x is given as a nalgebra vector and the solution will be of the same dimension as the input vector. This struct requires the Jacobian Matrix of `F` this is given as nalgebra Matrix. This struct uses the Newton-Raphson method for system of equations ([Wikipedia](https://en.wikipedia.org/wiki/Newton%27s_method#k_variables,_k_functions)).
///
/// **Default Tolerance:** `1e-6`
///
/// **Default Max Iterations:** `50`
///
/// ## Examples
///
/// ```
/// use eqsolver::multivariable::MultiVarNewton;
/// use nalgebra::{Vector2, Matrix2};
/// // Vectorial Function (x, y) ↦ (x^2-y-1, xy - 2). Want to solve x^2 - y = 1 and xy = 2
/// let F = |v: Vector2<f64>| Vector2::new(v[0].powi(2) - v[1] - 1., v[0] * v[1] - 2.);
///
/// // Jacobian of F
/// let J = |v: Vector2<f64>| Matrix2::new(2. * v[0], -1.,
///                                             v[1], v[0]);
///
/// // Solved analytically but its form was ugly so here is an approximation
/// const SOLUTION: Vector2<f64> = Vector2::new(1.521379706804567569604081, 1.314596212276751981650111);
///
/// let solution = MultiVarNewton::new(F, J)
///                 .with_tol(1e-6)
///                 .solve(Vector2::new(1., 1.))
///                 .unwrap();
///
/// assert!((solution - SOLUTION).norm() <= 1e-6);
/// ```
pub struct MultiVarNewton<T, D, F, J> {
    f: F,
    j: J,
    tolerance: T,
    iter_max: usize,
    d_phantom: PhantomData<D>,
}

impl<T, D, F, J> MultiVarNewton<T, D, F, J>
where
    T: Float + Scalar + ComplexField<RealField = T> + Signed,
    D: Dim,
    J: Fn(VectorType<T, D>) -> MatrixType<T, D, D>,
    F: Fn(VectorType<T, D>) -> VectorType<T, D>,
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
{
    /// Set up the solver
    ///
    /// Instantiate the solver using the given vectorial function `F` that is closure that takes a nalgebra vector and outputs a nalgebra vector of the same size.
    /// This also takes the Jacobian of `F` as `J` that is a closure that takes a nalgebra vector and outputs its jacobian as nalgebra matrix.
    ///
    /// ## Examples
    /// ```
    /// use eqsolver::multivariable::MultiVarNewton;
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
            d_phantom: PhantomData,
        }
    }

    /// Updates the solver's tolerance (Magnitude of Error).
    ///
    /// **Default Tolerance:** `1e-6`
    ///
    /// ## Examples
    /// ```
    /// use eqsolver::multivariable::MultiVarNewton;
    /// use nalgebra::{Vector2, Matrix2};
    /// // Vectorial Function (x, y) ↦ (x^2-y-1, xy - 2). Want to solve x^2 - y = 1 and xy = 2
    /// let F = |v: Vector2<f64>| Vector2::new(v[0].powi(2) - v[1] - 1., v[0] * v[1] - 2.);
    ///
    /// // Jacobian of F
    /// let J = |v: Vector2<f64>| Matrix2::new(2. * v[0], -1.,
    ///                                             v[1], v[0]);
    ///
    /// // Solved analytically but its form was ugly so here is an approximation
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

    /// Updates the solver's amount of iterations done before terminating the iteration
    ///
    /// **Default Max Iterations:** `50`
    pub fn with_itermax(&mut self, max: usize) -> &mut Self {
        self.iter_max = max;
        self
    }

    /// Solves for `x` in `F(x) = 0` where `F` is the stored function.
    ///
    /// ## Examples
    ///
    /// ### Working solution
    ///
    /// ```
    /// use eqsolver::multivariable::MultiVarNewton;
    /// use nalgebra::{Vector2, Matrix2};
    /// // Vectorial Function (x, y) ↦ (x^2-y-1, xy - 2). Want to solve x^2 - y = 1 and xy = 2
    /// let F = |v: Vector2<f64>| Vector2::new(v[0].powi(2) - v[1] - 1., v[0] * v[1] - 2.);
    ///
    /// // Jacobian of F
    /// let J = |v: Vector2<f64>| Matrix2::new(2. * v[0], -1.,
    ///                                             v[1], v[0]);
    ///
    /// // Solved analytically but its form was ugly so here is an approximation
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
    /// ### Bad Jacobian Error
    /// ```
    /// use eqsolver::{multivariable::MultiVarNewton, SolverError};
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
    pub fn solve(&self, mut x0: VectorType<T, D>) -> SolverResult<VectorType<T, D>> {
        let mut dv = x0.clone().add_scalar(T::max_value()); // We assume error vector is infinitely long at the start
        let mut iter = 1;

        // Newton-Raphson Iteration
        while dv.apply_norm(&UniformNorm) > self.tolerance && iter <= self.iter_max {
            if let Some(j_inv) = (self.j)(x0.clone()).try_inverse() {
                dv = j_inv * (self.f)(x0.clone());
                x0 -= &dv;
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
