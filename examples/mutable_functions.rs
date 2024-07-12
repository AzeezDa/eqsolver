use eqsolver::single_variable::Secant;
use std::cell::RefCell;

fn main() {
    /*
    Sometimes the function to be solved requires mutation of its environment,
    for example saving the values that function was called with, or for memoization.
    This mutation behaviour can be done using a `std::cell::RefCell` as in this example.
    */

    let trace = RefCell::new(vec![]);
    let f = |x: f64| {
        trace.borrow_mut().push(x);
        x * x - 2.
    };

    Secant::new(f).solve(0., 2.).unwrap();

    println!("{:?}", trace.borrow());

    // See this issue, https://github.com/AzeezDa/eqsolver/issues/5, for the reasoning behind this technicality.
}
