use std::ops::Fn;
use nalgebra::{SMatrix, SVector};

pub enum FiniteDifferenceType {
    Central, 
    Foward,
    Backward
}

pub fn central<F: Fn(f64) -> f64>(f: F, x: f64, h: f64) -> f64 {
    (f(x + h) - f(x - h)) / h / 2.
}

pub fn forward<F: Fn(f64) -> f64>(f: F, x: f64, h: f64) -> f64 {
    (f(x + h) - f(x)) / h
}

pub fn backward<F: Fn(f64) -> f64>(f: F, x: f64, h: f64) -> f64 {
    (f(x) - f(x - h)) / h
}

