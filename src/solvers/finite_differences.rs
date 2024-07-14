use num_traits::Float;
use std::ops::Fn;

/// Finite difference types for derivative approximation
///
/// Enum type that is used in the solvers to determine what type of finite difference to use.
pub enum FiniteDifferenceType {
    /// Central Difference: `(f(x+h) - f(x-h))/(2h)`. This has Order of Accuracy 2.
    Central,

    /// Forward Difference: `(f(x+h) - f(x))/h`. This has Order of Accuracy 1.
    Forward,

    // Backward Difference: `(f(x) - f(x-h))/h`. This has Order of Accuracy 1.
    Backward,
}

/// Central difference
///
/// Given the closure `F(T) -> T` where `T` is a floating number, a value `x`, and the step length h, approximate the derivative of `F` at `x` using `(f(x+h) - f(x-h))/(2h)`. This has the Order of Accuracy 2.
pub fn central<T: Float, F: Fn(T) -> T>(f: F, x: T, h: T) -> T {
    (f(x + h) - f(x - h)) / (h + h)
}

/// Forward difference
///
/// Given the closure `F(T) -> T` where `T` is a floating number, a value `x`, and the step length h, approximate the derivative of `F` at `x` using `(f(x+h) - f(x))/h`. This has the Order of Accuracy 1.
pub fn forward<T: Float, F: Fn(T) -> T>(f: F, x: T, h: T) -> T {
    (f(x + h) - f(x)) / h
}

/// Backward difference
///
/// Given the closure `F(T) -> T` where `T` is a floating number, a value `x`, and the step length h, approximate the derivative of `F` at `x` using `(f(x) - f(x-h))/h`. This has the Order of Accuracy 1.
pub fn backward<T: Float, F: Fn(T) -> T>(f: F, x: T, h: T) -> T {
    (f(x) - f(x - h)) / h
}
