macro_rules! gen_multivarnewton {
    ($name:ident, $dim:expr) => {
        pub struct $name<F, J>
        where
            J: Fn(nalgebra::SVector<f64, $dim>) -> nalgebra::SMatrix<f64, $dim, $dim>,
            F: Fn(nalgebra::SVector<f64, $dim>) -> nalgebra::SVector<f64, $dim>,
        {
            f: F,
            j: J,
            tolerance: f64,
            iter_max: usize,
        }

        impl<F, J> $name<F, J>
        where
            J: Fn(nalgebra::SVector<f64, $dim>) -> nalgebra::SMatrix<f64, $dim, $dim>,
            F: Fn(nalgebra::SVector<f64, $dim>) -> nalgebra::SVector<f64, $dim>,
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

            pub fn solve(&self, mut x0: nalgebra::SVector<f64, $dim>) -> Result<nalgebra::SVector<f64, $dim>, crate::solvers::SolverError> {
                let mut dv: nalgebra::SVector<f64, $dim> = nalgebra::SVector::repeat(f64::MAX);
                let mut iter = 1;

                while dv.abs().max() > self.tolerance && iter < self.iter_max {
                    let lu_decomp = (self.j)(x0).lu();
                    if let Some(solution) = lu_decomp.solve(&(self.f)(x0)) {
                        dv = solution;
                        x0 = x0 - dv;
                        iter += 1;
                    } else {
                        return Err(crate::solvers::SolverError::BadJacobian);
                    }
                }

                Ok(x0)
            }
        }
    };
}

pub (crate) use gen_multivarnewton;
