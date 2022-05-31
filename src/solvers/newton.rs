use super::{SolverError, DEFAULT_ITERMAX, DEFAULT_TOL};
use std::ops::Fn;

pub struct NewtonSolver<F, D>
where
    F: Fn(f64) -> f64,
    D: Fn(f64) -> f64,
{
    f: F,
    df: D,
    tolerance: f64,
    iter_max: usize,
}

impl<F, D> NewtonSolver<F, D>
where
    F: Fn(f64) -> f64,
    D: Fn(f64) -> f64,
{
    pub fn new(f: F, df: D) -> Self {
        Self {
            f,
            df,
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

    pub fn solve(&self, mut x0: f64) -> Result<f64, SolverError> {
        let mut dx = f64::MAX;
        let mut iter = 1;
        while dx.abs() > self.tolerance && iter <= self.iter_max {
            dx = (self.f)(x0)/(self.df)(x0);
            x0 = x0 - dx;
            iter += 1;
        }

        if iter >= self.iter_max {
            return Err(SolverError::MaxIterReached);
        }

        Ok(x0)
    }
}
