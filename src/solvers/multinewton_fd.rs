#[allow(dead_code)]
use nalgebra::{SMatrix, SVector};
use super::{SolverError, DEFAULT_ITERMAX, DEFAULT_TOL};

pub struct MultiVarNewtonFD<F, const S: usize>
where
    F: Fn(SVector<f64, S>) -> SVector<f64, S>,
{
    f: F,
    h: f64,
    tolerance: f64,
    iter_max: usize,
}

impl<F, const S: usize> MultiVarNewtonFD<F, S>
where
    F: Fn(SVector<f64, S>) -> SVector<f64, S>,
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

    pub fn solve(&self, mut x0: SVector<f64, S>) -> Result<SVector<f64, S>, SolverError> {
        let mut dv: SVector<f64, S> = SVector::repeat(f64::MAX);
        let mut iter = 1;

        while dv.abs().max() > self.tolerance && iter < self.iter_max {
            let mut j: SMatrix<f64, S, S> = SMatrix::zeros();
            let fx = (self.f)(x0);

            for i in 0..S {
                let mut x_h = x0;
                x_h[i] = x_h[i] + self.h;
                let df = ((self.f)(x_h) - fx)/self.h;
                for k in 0..S {
                    j[(k, i)] = df[k];
                }
            }

            if let Some(j_inv) = j.try_inverse() {
                dv = j_inv * fx;
                x0 = x0 - dv;
                iter += 1;
            } else {
                return Err(SolverError::BadJacobian);
            }
        }

        Ok(x0)
    }
}
