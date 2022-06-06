
pub struct MultiVarNewton<F, J, const S: usize>
where
    J: Fn(nalgebra::SVector<f64, S>) -> nalgebra::SMatrix<f64, S, S>,
    F: Fn(nalgebra::SVector<f64, S>) -> nalgebra::SVector<f64, S>,
{
    f: F,
    j: J,
    tolerance: f64,
    iter_max: usize,
}

impl<F, J, const S: usize> MultiVarNewton<F, J, S>
where
    J: Fn(nalgebra::SVector<f64, S>) -> nalgebra::SMatrix<f64, S, S>,
    F: Fn(nalgebra::SVector<f64, S>) -> nalgebra::SVector<f64, S>,
{
    pub fn new(f: F, j: J) -> Self {
        Self {
            f,
            j,
            tolerance: crate::solvers::DEFAULT_TOL,
            iter_max: crate::solvers::DEFAULT_ITERMAX,
        }
    }

    pub fn with_tol(&mut self, tol: f64) -> &mut Self {
        self.tolerance = tol;
        self
    }

    pub fn with_itermax(&mut self, max: usize) -> &mut Self {
        self.iter_max = max;
        self
    }

    pub fn solve(&self, mut x0: nalgebra::SVector<f64, S>) -> Result<nalgebra::SVector<f64, S>, crate::solvers::SolverError> {
        let mut dv: nalgebra::SVector<f64, S> = nalgebra::SVector::repeat(f64::MAX);
        let mut iter = 1;

        while dv.abs().max() > self.tolerance && iter < self.iter_max {
            if let Some(j_inv) = (self.j)(x0).try_inverse() {
                dv = j_inv * (self.f)(x0);
                x0 = x0 - dv;
                iter += 1;
            } else {
                return Err(crate::solvers::SolverError::BadJacobian);
            }
        }

        Ok(x0)
    }
}
