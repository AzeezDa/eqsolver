use criterion::{black_box, criterion_group, Criterion};
use eqsolver::multivariable::*;
use eqsolver::global_optimisers::*;
use nalgebra::{vector, DMatrix, DVector, Matrix, SVector};
use std::f64::consts::PI;

macro_rules! bench_lm {
    ($c:ident, $name:literal, $f:expr, $j:expr, $guess:expr) => {
        $c.bench_function(format!("LM {}", $name).as_str(), |bh| {
            bh.iter(|| {
                LevenbergMarquardt::new($f, $j)
                    .solve(black_box($guess))
                    .unwrap()
            })
        });

        $c.bench_function(format!("LMFD {}", $name).as_str(), |bh| {
            bh.iter(|| {
                LevenbergMarquardtFD::new($f)
                    .solve(black_box($guess))
                    .unwrap()
            })
        });
    };
    ($c:ident, $f:expr, $j:expr, $guess:expr) => {
        bench_lm!($c, "", $f, $j, $guess);
    };
}

macro_rules! bench_gn {
    ($c:ident, $name:literal, $f:expr, $j:expr, $guess:expr) => {
        $c.bench_function(format!("GN {}", $name).as_str(), |bh| {
            bh.iter(|| GaussNewton::new($f, $j).solve(black_box($guess)).unwrap())
        });

        $c.bench_function(format!("GNFD {}", $name).as_str(), |bh| {
            bh.iter(|| GaussNewtonFD::new($f).solve(black_box($guess)).unwrap())
        });
    };
    ($c:ident, $f:expr, $j:expr, $guess:expr) => {
        bench_gn!($c, "", $f, $j, $guess);
    };
}

macro_rules! bench_newton {
    ($c:ident, $name:literal, $f:expr, $j:expr, $guess:expr) => {
        $c.bench_function(format!("NR {}", $name).as_str(), |bh| {
            bh.iter(|| {
                MultiVarNewton::new($f, $j)
                    .solve(black_box($guess))
                    .unwrap()
            })
        });

        $c.bench_function(format!("NRFD {}", $name).as_str(), |bh| {
            bh.iter(|| MultiVarNewtonFD::new($f).solve(black_box($guess)).unwrap())
        });
    };
    ($c:ident, $f:expr, $j:expr, $guess:expr) => {
        bench_newton!($c, "", $f, $j, $guess);
    };
}

macro_rules! bench_pso {
    ($c:ident, $name:literal, $f:expr, $guess:expr, $lbounds:expr, $ubounds:expr) => {
        $c.bench_function(format!("PSO {}", $name).as_str(), |bh| {
            bh.iter(|| {
                ParticleSwarm::new($f, $lbounds, $ubounds)
                    .solve(black_box($guess))
                    .unwrap()
            })
        });
    };
    ($c:ident, $f:expr, $guess:expr, $lbounds:expr, $ubounds:expr) => {
        bench_pso!($c, "", $f, $guess, $lbounds, $ubounds);
    };
}

fn bench_multi_variable_heavy(c: &mut Criterion) {
    const SIZE: usize = 100;
    let mut group = c.benchmark_group(format!("{SIZE}D Heavy"));

    let f = |v: DVector<f64>| {
        let mut out = DVector::zeros(SIZE);
        for (i, o) in Matrix::iter_mut(&mut out).enumerate() {
            for x in v.iter().skip(i) {
                *o += x.powi(2)
            }
        }

        out
    };

    let j = |v: DVector<f64>| {
        let mut out = DMatrix::<f64>::zeros(SIZE, SIZE);

        for (i, mut r) in out.row_iter_mut().enumerate() {
            for (j, c) in r.iter_mut().enumerate().skip(i) {
                *c = 2. * v[j];
            }
        }

        out
    };

    let init = DVector::<f64>::repeat(SIZE, 0.3);

    bench_lm!(group, f, j, init.clone());
    bench_gn!(group, f, j, init.clone());
    bench_newton!(group, f, j, init.clone());
    group.finish()
}

fn bench_multi_variable_lm_rastrigin(c: &mut Criterion) {
    const SIZE: usize = 30;
    let mut group = c.benchmark_group(format!("{SIZE}D Rastrigin"));

    let f = |v: SVector<f64, SIZE>| {
        let mut total = 10. * SIZE as f64;

        for &w in v.iter() {
            total += w * w - 10. * (2. * PI * w).cos();
        }

        total
    };

    let f_vec = |v: SVector<f64, SIZE>| {
        vector![f(v)]
    };

    let j = |mut v: SVector<f64, SIZE>| {
        for w in v.iter_mut() {
            let w_ = *w;
            *w = 2. * w_ + 20. * PI * (2. * PI * w_).sin();
        }

        v.transpose()
    };

    let init = SVector::<f64, SIZE>::repeat(0.4);
    let bounds = SVector::repeat(1.);

    bench_lm!(group, f_vec, j, init);
    bench_pso!(group, f, init, -bounds, bounds);
}

criterion_group!(
    benches,
    bench_multi_variable_lm_rastrigin,
    bench_multi_variable_heavy
);
