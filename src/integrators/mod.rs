mod newton_cotes;
mod adaptive_newton_cotes;

pub use newton_cotes::{Formula, NewtonCotes, DEFAULT_SUBDIVISIONS};
pub use adaptive_newton_cotes::{AdaptiveNewtonCotes, DEFAULT_MAXIMUM_CUT_AMOUNT};
