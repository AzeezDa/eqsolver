macro_rules! gen_multivarnewton_fd {
    ($name:ident, $dim:expr) => {
        pub struct $name<F>
        where
            F: Fn(nalgebra::SVector<f64, $dim>) -> nalgebra::SVector<f64, $dim>,
        {
            f: F,
            h: f64,
            tolerance: f64,
            iter_max: usize,
        }

        impl<F> $name<F>
        where
            F: Fn(nalgebra::SVector<f64, $dim>) -> nalgebra::SVector<f64, $dim>,
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

            pub fn solve(&self, mut x0: nalgebra::SVector<f64, $dim>) -> Result<nalgebra::SVector<f64, $dim>, crate::solvers::SolverError> {
                let mut dv: nalgebra::SVector<f64, $dim> = nalgebra::SVector::repeat(f64::MAX);
                let mut iter = 1;

                while dv.abs().max() > self.tolerance && iter < self.iter_max {
                    let mut j: SMatrix<f64, $dim, $dim> = SMatrix::zeros();
                    let fx = (self.f)(x0);

                    for i in 0..$dim {
                        let mut x_h = x0;
                        x_h[i] = x_h[i] + self.h;
                        let df = ((self.f)(x_h) - fx)/self.h;
                        for k in 0..$dim {
                            j[(k, i)] = df[k];
                        }
                    }

                    let lu_decomp = j.lu();
                    if let Some(solution) = lu_decomp.solve(&fx) {
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

pub (crate) use gen_multivarnewton_fd;

