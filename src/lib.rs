pub mod math;
pub mod solvers;
pub extern crate nalgebra;

#[cfg(test)]
mod tests {
    use crate::math::finite_differences::{backward, central, forward};
    use crate::solvers::{FDNewton, NewtonSolver, SecantSolver};
    use crate::{gen_gaussnewton, gen_gaussnewton_fd, gen_multivarnewton, gen_multivarnewton_fd};
    use nalgebra::{Vector2, Matrix2, Vector3, SMatrix, Matrix3x2, vector};

    #[test]
    fn solve_secant() {
        let f = |x: f64| x * x - 2.;

        let solution = SecantSolver::new(f)
            .with_tol(1e-3)
            .solve(0., 2.)
            .ok()
            .unwrap();

        assert!((solution - 2_f64.sqrt()).abs() <= 1e-3);
        assert!((solution - 2_f64.sqrt()).abs() > 1e-12);
    }

    #[test]
    fn solve_newton() {
        let f = |x: f64| x.sin() * 2. - x;
        let df = |x: f64| x.cos() * 2. - 1.;
        const SOLUTION: f64 = 1.8954942670339809471; // From Wolfram Alpha

        let solution = NewtonSolver::new(f, df)
            .with_tol(1e-3)
            .solve(2.)
            .ok()
            .unwrap();

        assert!((solution - SOLUTION).abs() <= 1e-3);
        assert!((solution - SOLUTION).abs() > 1e-12);
    }

    #[test]
    fn finite_differences() {
        let f = |x: f64| x.powi(3);
        let x = -1.;

        const SOLUTION: f64 = 3.;
        let dx_c = central(f, x, f64::EPSILON.sqrt());
        let dx_f = forward(f, x, f64::EPSILON);
        let dx_b = backward(f, x, f64::EPSILON);

        assert!((dx_c - SOLUTION).abs() <= f64::EPSILON);
        assert!((dx_f - SOLUTION).abs() <= f64::EPSILON);
        assert!((dx_b - SOLUTION).abs() <= f64::EPSILON);
    }

    #[test]
    fn newton_with_finite_differences() {
        let f = |x: f64| x.exp().sin() / (1. + x * x) - (-x).exp();

        const SOLUTION: f64 = 0.1168941457861853920; // From Wolfram Alpha

        let solution = FDNewton::new(f).with_tol(1e-3).solve(0.).ok().unwrap();

        assert!((solution - SOLUTION).abs() <= 1e-3);
        assert!((solution - SOLUTION).abs() > 1e-12);
    }

    #[test]
    fn multi_var_newton() {
        // Vectorial Function (x, y) ↦ (x^2-y-1, xy - 2)
        let f = |v: Vector2<f64>| Vector2::new(v[0].powi(2) - v[1] - 1., v[0] * v[1] - 2.);

        // Jacobian of F
        let j = |v: Vector2<f64>| Matrix2::new(2. * v[0], -1., v[1], v[0]);

        // Solved analytically but form was ugly so here is approximation
        const SOLUTION: Vector2<f64> = Vector2::new(1.521379706804567569604081, 1.314596212276751981650111);

        // Generate struct to solve non-linear system of equation of 2 equations and 2 variables (assuming Jacobian can be analytically found)
        gen_multivarnewton!(MVN2, 2);
        gen_multivarnewton_fd!(MVNFD2, 2);

        // Use struct generated above to solve the problem defined in f
        let solution = MVN2::new(f, j)
            .with_tol(1e-3)
            .with_itermax(50)
            .solve(Vector2::repeat(1.)).ok().unwrap();

            

        // Use struct generated above to solve the problem defined in f
        let solution_fd = MVNFD2::new(f)
            .with_tol(1e-3)
            .with_itermax(50)
            .solve(Vector2::repeat(1.)).ok().unwrap();


        assert!((SOLUTION - solution).norm() <= 1e-3);
        assert!((SOLUTION - solution).norm() > 1e-12);
        assert!((SOLUTION - solution_fd).norm() <= 1e-3);
        assert!((SOLUTION - solution_fd).norm() > 1e-12);
    }

    #[test]
    fn gauss_newton() {
        // Test is about finding point closest to three circles in R2

        // [Center_x, Center_y, Radius]
        let c0 = [3., 5., 3.];
        let c1 = [1., 0., 4.];
        let c2 = [6., 2., 2.];

        // Want to (x, y) such that F(x, y) = (x - X)^2 + (y - Y) - R^2 is minimized in a Least Square sense for data in c0, c1, c2
        let f = |v: Vector2<f64>| Vector3::new(
                                                                            (v[0] - c0[0]).powi(2) + (v[1] - c0[1]).powi(2) - c0[2] * c0[2],
                                                                            (v[0] - c1[0]).powi(2) + (v[1] - c1[1]).powi(2) - c1[2] * c1[2],
                                                                            (v[0] - c2[0]).powi(2) + (v[1] - c2[1]).powi(2) - c2[2] * c2[2],
                                                                        );

        let j = |v: Vector2<f64>| Matrix3x2::new(
                                                                            2. * (v[0] - c0[0]), 2. * (v[1] - c0[1]),
                                                                            2. * (v[0] - c1[0]), 2. * (v[1] - c1[1]),
                                                                            2. * (v[0] - c2[0]), 2. * (v[1] - c2[1]),
                                                                        );
        
        // Solved using Octave (Can also be checked visually in Desmos or similar)
        const SOLUTION: Vector2<f64> = vector![4.217265312839526, 2.317879970005811];
        
        gen_gaussnewton!(GN3x2, 3, 2);
        gen_gaussnewton_fd!(GNFD3x2, 3, 2);

        let solution_gn = GN3x2::new(f, j)
                        .with_tol(1e-3)
                        .solve(vector![4.5, 2.5]).ok().unwrap();

        let solution_gn_fd = GNFD3x2::new(f)
                        .with_tol(1e-3)
                        .solve(vector![4.5, 2.5]).ok().unwrap();

        assert!((SOLUTION - solution_gn).norm() <= 1e-3);
        assert!((SOLUTION - solution_gn).norm() > 1e-12);
        assert!((SOLUTION - solution_gn_fd).norm() <= 1e-3);
        assert!((SOLUTION - solution_gn_fd).norm() > 1e-12);
    }

}
