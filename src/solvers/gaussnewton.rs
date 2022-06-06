#[macro_export]
macro_rules! gen_gaussnewton {
    ($name:ident, $dim_n:expr, $dim_m:expr) => {
        pub struct $name<F, J>
        where
            F: Fn(nalgebra::SVector<f64, $dim_m>) -> nalgebra::SVector<f64, $dim_n>,
            J: Fn(nalgebra::SVector<f64, $dim_m>) -> SMatrix<f64, $dim_n, $dim_m>,
        {
            f: F,
            j: J,
            tolerance: f64,
            iter_max: usize,
        }

        impl<F, J> $name<F, J>
        where
            F: Fn(nalgebra::SVector<f64, $dim_m>) -> nalgebra::SVector<f64, $dim_n>,
            J: Fn(nalgebra::SVector<f64, $dim_m>) -> nalgebra::SMatrix<f64, $dim_n, $dim_m>,
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

            pub fn solve(&self, mut x0: nalgebra::SVector<f64, $dim_m>) -> Result<nalgebra::SVector<f64, $dim_m>, crate::solvers::SolverError> {
                let mut dv: nalgebra::SVector<f64, $dim_m> = nalgebra::SVector::repeat(f64::MAX);
                let mut iter = 1;

                while dv.abs().max() > self.tolerance && iter < self.iter_max {
                    let j = (self.j)(x0);
                    let jt = j.transpose();
                    let lu_decomp = (jt*j).lu();
                    if let Some(solution) = lu_decomp.solve(&(jt*(self.f)(x0))) {
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

pub (crate) use gen_gaussnewton;
