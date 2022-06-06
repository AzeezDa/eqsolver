#[allow(dead_code)]
use nalgebra::{SMatrix, SVector};
use super::{SolverError, DEFAULT_ITERMAX, DEFAULT_TOL};

pub struct GaussNewtonFD<F, const R: usize, const C: usize>
where
    F: Fn(SVector<f64, R>) -> SVector<f64, C>,
{
    f: F,
    h: f64,
    tolerance: f64,
    iter_max: usize,
}

impl<F, const R: usize, const C: usize> GaussNewtonFD<F, R, C>
where
    F: Fn(SVector<f64, R>) -> SVector<f64, C>,
{
    pub fn new(f: F) -> Self {
        Self {
            f,
            h: f64::EPSILON.sqrt(),
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

    pub fn with_fd_step_length(&mut self, h: f64) -> &mut Self {
        self.h = h;
        self
    }

    pub fn solve(&self, mut x0: SVector<f64, R>) -> Result<SVector<f64, R>, SolverError> {
        let mut dv: SVector<f64, R> = SVector::repeat(f64::MAX);
        let mut iter = 1;

        while dv.abs().max() > self.tolerance && iter < self.iter_max {
            let mut j: SMatrix<f64, C, R> = SMatrix::zeros();
            let fx = (self.f)(x0);

            for i in 0..R {
                let mut x_h = x0;
                x_h[i] = x_h[i] + self.h;
                let df = ((self.f)(x_h) - fx)/self.h;
                for k in 0..C {
                    j[(k, i)] = df[k];
                }
            }

            let jt = j.transpose();
            if let Some(jjt_inv) = (jt*j).try_inverse() {
                dv = jjt_inv*jt*fx;
                x0 = x0 - dv;
                iter += 1;
            } else {
                return Err(SolverError::BadJacobian);
            }
        }

        Ok(x0)
    }
}
