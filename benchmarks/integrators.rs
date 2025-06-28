use criterion::{criterion_group, Criterion};
use std::hint::black_box;
use eqsolver::integrators::*;

criterion_group!(
    benches,
    bench_integrators_heavy,
);

macro_rules! bench_with_all_methods {
    ($c:ident, $f:expr, $from:expr, $to:expr) => {
        $c.bench_function("Trapezium", |bh| {
            bh.iter(|| NewtonCotes::new($f).with_formula(Formula::Trapezium).integrate(black_box($from), black_box($to)));
        });

        $c.bench_function("Simpson's 1/3", |bh| {
            bh.iter(|| NewtonCotes::new($f).with_formula(Formula::SimpsonsOneThird).integrate(black_box($from), black_box($to)));
        });

        $c.bench_function("Simpson's 3/8", |bh| {
            bh.iter(|| NewtonCotes::new($f).with_formula(Formula::SimpsonsThreeEighths).integrate(black_box($from), black_box($to)));
        });

        $c.bench_function("Adaptive Trapezium", |bh| {
            bh.iter(|| AdaptiveNewtonCotes::new($f).with_formula(Formula::Trapezium).integrate(black_box($from), black_box($to)));
        });

        $c.bench_function("Adaptive Simpson's 1/3", |bh| {
            bh.iter(|| AdaptiveNewtonCotes::new($f).with_formula(Formula::SimpsonsOneThird).integrate(black_box($from), black_box($to)));
        });

        $c.bench_function("Adaptive Simpson's 3/8", |bh| {
            bh.iter(|| AdaptiveNewtonCotes::new($f).with_formula(Formula::SimpsonsThreeEighths).integrate(black_box($from), black_box($to)));
        });
    };
}

fn bench_integrators_heavy(c: &mut Criterion) {
    let mut group = c.benchmark_group("1D Heavy Integrators");
    let f = |x: f64| {
        let mut sum = 0.;
        for i in 0..1000 {
            sum += (-1f64).powi(i) * x.powi(i);
        }
        sum
    };

    bench_with_all_methods!(group, f, 0., 0.5);
}
