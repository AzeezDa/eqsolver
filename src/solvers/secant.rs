use super::{SolverError, DEFAULT_ITERMAX, DEFAULT_TOL};
use std::ops::Fn;

pub struct SecantSolver<F>
where
    F: Fn(f64) -> f64,
{
    f: F,
    tolerance: f64,
    iter_max: usize,
}

impl<F> SecantSolver<F>
where
    F: Fn(f64) -> f64,
{
    pub fn new(f: F) -> Self {
        Self {
            f,
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

    pub fn solve(&self, mut x0: f64, mut x1: f64) -> Result<f64, SolverError> {
        if x0 == x1 {
            return Err(SolverError::IncorrectInput);
        };

        let mut dx = f64::MAX;
        let mut iter = 1;
        let mut f0 = (self.f)(x0);
        let mut f1 = (self.f)(x1);
        while dx.abs() > self.tolerance && iter <= self.iter_max {
            dx = f1 * (x1 - x0) / (f1 - f0);
            x0 = x1;
            x1 = x1 - dx;
            f0 = f1;
            f1 = (self.f)(x1);
            iter += 1;
        }

        if iter >= self.iter_max {
            return Err(SolverError::MaxIterReached);
        }

        Ok(x1)
    }
}
