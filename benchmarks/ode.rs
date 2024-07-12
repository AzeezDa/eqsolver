use criterion::{black_box, criterion_group, Criterion};
use eqsolver::{ODESolver, ODESolverMethod};
use nalgebra::{vector, SVector, Vector2};

criterion_group!(
    benches,
    bench_1st_order_ode,
    bench_higher_order_ode,
    bench_cauchy_euler
);

macro_rules! bench_with_all_methods {
    ($c:ident, $f:expr, $x0:expr, $y0:expr, $step_size:expr, $x_end:expr) => {{
        let mut solver = ODESolver::new($f, $x0, $y0, $step_size);

        solver.with_method(ODESolverMethod::EulerForward);
        $c.bench_function("Euler", |bh| {
            bh.iter(|| {
                solver.solve(black_box($x_end)).unwrap();
            })
        });

        solver.with_method(ODESolverMethod::Heun);
        $c.bench_function("Heun", |bh| {
            bh.iter(|| {
                solver.solve(black_box($x_end)).unwrap();
            })
        });

        solver.with_method(ODESolverMethod::RungeKutta4);
        $c.bench_function("RK4", |bh| {
            bh.iter(|| {
                solver.solve(black_box($x_end)).unwrap();
            })
        });
    }};
}

fn bench_1st_order_ode(c: &mut Criterion) {
    let mut group = c.benchmark_group("1st order ODE");

    let f = |t: f64, y: f64| t * y; // y' = f(t, y) = ty
    let (x0, y0) = (0., 0.2);
    let x_end = 2.;
    let step_size = 1e-3;

    bench_with_all_methods!(group, f, x0, y0, step_size, x_end);

    group.finish()
}

fn bench_higher_order_ode(c: &mut Criterion) {
    let mut group = c.benchmark_group("2nd Order ODE");
    let f = |t: f64, y: Vector2<f64>| Vector2::new(y[1], t - y[0]); // y'' = f(t, y) = ty
    let (x0, y0) = (0., vector![1., 1.]);
    let x_end = 2.;
    let step_size = 1e-3;

    bench_with_all_methods!(group, f, x0, y0, step_size, x_end);
}

fn bench_cauchy_euler(c: &mut Criterion) {
    const ORDER: usize = 1000;
    let mut group = c.benchmark_group(format!("Order {ORDER} Cauchy-Euler"));

    const SIZE: usize = ORDER + 1;
    let f = |t: f64, y: SVector<f64, SIZE>| {
        let mut v = SVector::zeros();
        for i in 0..ORDER {
            v[i] = y[i + 1];
            v[ORDER] -= t.powi(i as i32) * v[i];
        }

        v[ORDER] /= t.powi(ORDER as i32);
        v
    };

    let (x0, y0) = (1., SVector::repeat(1.));
    let x_end = 2.;
    let step_size = 1e-3;

    bench_with_all_methods!(group, f, x0, y0, step_size, x_end);
}
