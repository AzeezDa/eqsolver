macro_rules! gen_gaussnewton_fd {
    ($name:ident, $dim_n:expr, $dim_m:expr) => {
        pub struct $name<F>
        where
            F: Fn(nalgebra::SVector<f64, $dim_m>) -> nalgebra::SVector<f64, $dim_n>,
        {
            f: F,
            h: f64,
            tolerance: f64,
            iter_max: usize,
        }

        impl<F> $name<F>
        where
            F: Fn(nalgebra::SVector<f64, $dim_m>) -> nalgebra::SVector<f64, $dim_n>,
        {
            pub fn new(f: F) -> Self {
                Self {
                    f,
                    h: f64::EPSILON.sqrt(),
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

            pub fn with_fd_step_length(&mut self, h: f64) -> &mut Self {
                self.h = h;
                self
            }

            pub fn solve(&self, mut x0: nalgebra::SVector<f64, $dim_m>) -> Result<nalgebra::SVector<f64, $dim_m>, crate::solvers::SolverError> {
                let mut dv: nalgebra::SVector<f64, $dim_m> = nalgebra::SVector::repeat(f64::MAX);
                let mut iter = 1;

                while dv.abs().max() > self.tolerance && iter < self.iter_max {
                    let mut j: SMatrix<f64, $dim_n, $dim_m> = SMatrix::zeros();
                    let fx = (self.f)(x0);

                    for i in 0..$dim_m {
                        let mut x_h = x0;
                        x_h[i] = x_h[i] + self.h;
                        let df = ((self.f)(x_h) - fx)/self.h;
                        for k in 0..$dim_n {
                            j[(k, i)] = df[k];
                        }
                    }

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

pub (crate) use gen_gaussnewton_fd;
