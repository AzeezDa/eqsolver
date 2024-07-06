use criterion::{black_box, criterion_group, Criterion};
use eqsolver::multivariable::*;
use nalgebra::{vector, DMatrix, DVector, SVector};
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
}

fn bench_multi_variable_heavy(c: &mut Criterion) {
    const SIZE: usize = 100;
    let f = |v: DVector<f64>| {
        let mut out = DVector::zeros(SIZE);

        for (i, o) in out.iter_mut().enumerate() {
            for x in v.iter().skip(i) {
                *o += x.powi(2)
            }
        }

        out
    };

    let j = |v: DVector<f64>| {
        let mut out = DMatrix::zeros(SIZE, SIZE);

        for (i, mut r) in out.row_iter_mut().enumerate() {
            for (j, c) in r.iter_mut().enumerate().skip(i) {
                *c = 2. * v[j];
            }
        }

        out
    };

    let init = DVector::<f64>::repeat(SIZE, 0.3);

    bench_lm!(c, "heavy", f, j, init.clone());
    bench_gn!(c, "heavy", f, j, init.clone());
    bench_newton!(c, "heavy", f, j, init.clone());
}

fn bench_multi_variable_lm_rastrigin(c: &mut Criterion) {
    let f = |v: SVector<f64, 30>| {
        let mut total = 10. * 30.;

        for &w in v.iter() {
            total += w * w - 10. * (2. * PI * w).cos();
        }

        vector![total]
    };

    let j = |mut v: SVector<f64, 30>| {
        for w in v.iter_mut() {
            let w_ = *w;
            *w = 2. * w_ + 20. * PI * (2. * PI * w_).sin();
        }

        v.transpose()
    };

    let init = SVector::<f64, 30>::repeat(0.4);

    bench_lm!(c, "rastrigin", f, j, init);
}

criterion_group!(
    benches,
    bench_multi_variable_lm_rastrigin,
    bench_multi_variable_heavy
);
