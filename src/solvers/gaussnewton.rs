#[allow(dead_code)]
use nalgebra::{SMatrix, SVector};
use super::{SolverError, DEFAULT_ITERMAX, DEFAULT_TOL};


pub struct GaussNewton<F, J, const R: usize, const C: usize>
where
    F: Fn(SVector<f64, R>) -> SVector<f64, C>,
    J: Fn(SVector<f64, R>) -> SMatrix<f64, C, R>,
{
    f: F,
    j: J,
    tolerance: f64,
    iter_max: usize,
}

impl<F, J, const R: usize, const C: usize> GaussNewton<F, J, R, C>
where
    F: Fn(SVector<f64, R>) -> SVector<f64, C>,
    J: Fn(SVector<f64, R>) -> SMatrix<f64, C, R>,
{
    pub fn new(f: F, j: J) -> Self {
        Self {
            f,
            j,
            tolerance: DEFAULT_TOL,
            iter_max: DEFAULT_ITERMAX,
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

    pub fn solve(&self, mut x0: SVector<f64, R>) -> Result<SVector<f64, R>, SolverError> {
        let mut dv: SVector<f64, R> = SVector::repeat(f64::MAX);
        let mut iter = 1;

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

        Ok(x0)
    }
}