use criterion::{criterion_group, Criterion};
use std::hint::black_box;
use eqsolver::single_variable::*;
use std::f64::consts::PI;

criterion_group!(
    benches,
    bench_single_variable_heavy,
    bench_single_variable_rastrigin
);

macro_rules! bench_with_all_methods {
    ($c:ident, $f:expr, $d:expr, $newton_guess:expr, $secant_x0:expr, $secant_x1:expr) => {
        $c.bench_function("newton", |bh| {
            bh.iter(|| Newton::new($f, $d).solve(black_box($newton_guess)));
        });

        $c.bench_function("newtonfd", |bh| {
            bh.iter(|| FDNewton::new($f).solve(black_box($newton_guess)));
        });

        $c.bench_function("secant", |bh| {
            bh.iter(|| Secant::new($f).solve(black_box($secant_x0), black_box($secant_x1)));
        });
    };
}

fn bench_single_variable_heavy(c: &mut Criterion) {
    let mut group = c.benchmark_group("1D Heavy");
    let f = |x: f64| {
        let mut sum = 1.;
        for i in 0..1000 {
            sum += (-1f64).powi(i) * i as f64 * x.powi(i);
        }
        sum
    };

    let d = |x: f64| {
        let mut sum = 1.;
        for i in 1..1000 {
            sum += (-1f64).powi(i) * (i * i) as f64 * x.powi(i - 1);
        }
        sum
    };

    bench_with_all_methods!(group, f, d, 1., 0.5, 1.);
}

fn bench_single_variable_rastrigin(c: &mut Criterion) {
    let mut group = c.benchmark_group("1D Rastrigin");
    let f = |x: f64| 10. + x * x - 10. * (2. * PI * x).cos();
    let d = |x: f64| 2. * x + 20. * PI * (2. * PI * x).sin();

    bench_with_all_methods!(group, f, d, 0.5, -0.5, 0.5);
}
