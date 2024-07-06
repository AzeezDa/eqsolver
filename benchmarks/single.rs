use criterion::{black_box, criterion_group, Criterion};
use eqsolver::single_variable::*;
use std::f64::consts::PI;

criterion_group!(
    benches,
    bench_single_variable_heavy,
    bench_single_variable_rastrigin
);

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

    group.bench_function("newton", |bh| {
        bh.iter(|| Newton::new(f, d).solve(black_box(1.)));
    });

    group.bench_function("newtonfd", |bh| {
        bh.iter(|| FDNewton::new(f).solve(black_box(1.)));
    });

    group.bench_function("secant", |bh| {
        bh.iter(|| Secant::new(f).solve(black_box(0.5), black_box(1.)));
    });
}

fn bench_single_variable_rastrigin(c: &mut Criterion) {
    let mut group = c.benchmark_group("1D Rastrigin");
    let f = |x: f64| 10. + x * x - 10. * (2. * PI * x).cos();
    let d = |x: f64| 2. * x + 20. * PI * (2. * PI * x).sin();

    group.bench_function("newton", |bh| {
        bh.iter(|| Newton::new(f, d).solve(black_box(0.5)));
    });

    group.bench_function("newtonfd", |bh| {
        bh.iter(|| FDNewton::new(f).solve(black_box(0.5)));
    });

    group.bench_function("secant", |bh| {
        bh.iter(|| Secant::new(f).solve(black_box(-0.5), black_box(0.5)));
    });
}
