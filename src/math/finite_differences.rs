use num_traits::Float;
use std::ops::Fn;

/// # `FiniteDifferenceType`
/// 
/// Enum type that is used in the solvers to determine what type of finite difference to use.
pub enum FiniteDifferenceType {
    Central,
    Forward,
    Backward,
}

/// # `central`
/// Given the closure F(T) -> T where T is a floating number, point x to derivate at and the step length h.
/// This approximates the derivative using (f(x+h) - f(x-h))/(2h). This has Order of Accuracy 2.
pub fn central<T: Float, F: Fn(T) -> T>(f: F, x: T, h: T) -> T {
    (f(x + h) - f(x - h)) / (h + h)
}

/// # `forward`
/// Given the closure F(T) -> T where T is a floating number, point x to derivate at and the step length h.
/// This approximates the derivative using (f(x+h) - f(x))/h. This has Order of Accuracy 1.
pub fn forward<T: Float, F: Fn(T) -> T>(f: F, x: T, h: T) -> T {
    (f(x + h) - f(x)) / h
}

/// # `backward`
/// Given the closure F(T) -> T where T is a floating number, point x to derivate at and the step length h.
/// This approximates the derivative using (f(x) - f(x-h))/h. This has Order of Accuracy 1.
pub fn backward<T: Float, F: Fn(T) -> T>(f: F, x: T, h: T) -> T {
    (f(x) - f(x - h)) / h
}
