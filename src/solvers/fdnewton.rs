use std::ops::Fn;

use crate::math::finite_differences::{central, FiniteDifferenceType, forward, backward};

use super::{DEFAULT_ITERMAX, DEFAULT_TOL, SolverError};

pub struct FDNewton<F>
where
    F: Fn(f64) -> f64 + Copy,
{
    f: F,
    fd: fn(F, f64, f64) -> f64,
    h: f64,
    tolerance: f64,
    iter_max: usize,
}

impl<F> FDNewton<F>
where
    F: Fn(f64) -> f64 + Copy,
{
    pub fn new(f: F) -> Self {
        Self {
            f,
            fd: central,
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

    pub fn with_finite_difference(&mut self, fd_type: FiniteDifferenceType) -> &mut Self {
        match fd_type {
            FiniteDifferenceType::Central => {self.fd = central},
            FiniteDifferenceType::Foward => {self.fd = forward},
            FiniteDifferenceType::Backward => {self.fd = backward},
        }
        self
    }

    pub fn solve(&self, mut x0: f64) -> Result<f64, SolverError> {
        let mut dx = f64::MAX;
        let mut iter = 1;
        while dx.abs() > self.tolerance && iter <= self.iter_max {
            dx = (self.f)(x0)/(self.fd)(self.f, x0, self.h);
            x0 = x0 - dx;
            iter += 1;
        }

        if iter >= self.iter_max {
            return Err(SolverError::MaxIterReached);
        }

        Ok(x0)
    }
}
