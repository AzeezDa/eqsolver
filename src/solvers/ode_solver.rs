use super::ODESolverMethod;
use crate::solvers::{SolverError, SolverResult};
use nalgebra::{ClosedAddAssign, ClosedMulAssign};
use num_traits::Float;
#[allow(dead_code)]

/// # General ODE solver for Initial Value Problems
///
/// Solves first order ODE systems of equations in form of Initial Value Problem.
/// Given a system of equations `F(x)` that is a closure of the form `|x, y|` where `x` is a `Float` and `y` is either a `Float` or nalgebra `Vector` representing the system of first order ODEs.
///
/// This solver includes 3 explicit numerical methods for solving ODEs that are: Euler's Forward Method, Heun's Method (Runge-Kutta 2) and Runge-Kutta 4. The default is Runge-Kutta 4.
///
/// ## Examples
///
/// ### First Order ODE (no system)
///
/// ```rust
/// use eqsolver::ODESolver;
/// // ODE: y' = -y. where y(0) = 1
/// let f = |x: f64, y: f64| -y;
/// let (x0, y0) = (0., 1.);
/// let h = 1e-3; // step size
///
/// let solution = ODESolver::new(f, x0, y0, h)
///                 .solve(1.).unwrap();
///
/// assert!((solution - (-1_f64).exp()) <= 1e-3);
/// ```
///
/// ### System of ODEs
///
/// ```
/// use eqsolver::ODESolver;
/// use nalgebra::Vector2;
/// // ODE: y' = -y. where y(0) = 1
/// let f = |t: f64, v: Vector2<f64>| Vector2::new(v[1], v[1]-v[0]); // System {v0 =  y1
///                                                                  //        {v1 =  y1-y0
/// let (x0, v0) = (0., Vector2::new(1., 1.));
/// let h = 1e-3; // step size
///
/// const SOLUTION: f64 = 1.7932509391963074; // Exact solution for y(x)
///
/// let solution = ODESolver::new(f, x0, v0, h)
///                 .solve(1.).unwrap();
///
/// assert!((solution[0] - SOLUTION) <= 1e-3);
/// ```
pub struct ODESolver<T, V, F> {
    f: F,
    x0: T,
    y0: V,
    h: T,
    half_h: T,
    method: fn(&Self, T, V) -> V,
}

impl<T, V, F> ODESolver<T, V, F>
where
    T: Float,
    V: Copy + ClosedAddAssign + ClosedMulAssign<T>,
    F: Fn(T, V) -> V,
{
    /// Set up the solver with the initial value problem
    ///
    /// Instantiate the ODESolver given the derivative function `F(x, Y)` that represents the equation or the system, the initial values and the step size.
    ///
    /// ## Examples
    ///
    /// Given the equation `y' = t*y`, where `y(0) = 1`.
    /// ```
    /// # use eqsolver::ODESolver;
    ///
    /// let f = |t: f64, y: f64| t*y; // f(t, y) = y' = t*y
    /// let (x0, y0) = (0., 1.); // y0 = y(x0) = y(0) = 1
    /// let h = 1e-3; // Step size. Lower give more accurate solution but take longer to compute
    ///
    /// let solver = ODESolver::new(f, x0, y0, h);
    /// ```
    pub fn new(f: F, x0: T, y0: V, h: T) -> Self {
        Self {
            f,
            x0,
            y0,
            h,
            half_h: h / T::from(2_f64).unwrap(),
            method: Self::rk4_step,
        }
    }

    /// Solve the Initial Value Problem
    ///
    /// Solve the equation (or system) at `x_end` using the numerical method of the solver.
    ///
    /// ## Examples
    ///
    /// ## First Order ODE (no system)
    ///
    /// ```
    /// use eqsolver::ODESolver;
    /// // ODE: y' = -y. where y(0) = 1
    /// let f = |x: f64, y: f64| -y;
    /// let (x0, y0) = (0., 1.);
    /// let h = 1e-3; // step size
    ///
    /// let solution = ODESolver::new(f, x0, y0, h)
    ///                 .solve(1.).unwrap();
    ///
    /// assert!((solution - (-1_f64).exp()) <= 1e-3);
    /// ```
    ///
    /// ## System of ODEs
    ///
    /// ```
    /// use eqsolver::ODESolver;
    /// use nalgebra::Vector2;
    /// // ODE: y' = -y. where y(0) = 1
    /// let f = |t: f64, v: Vector2<f64>| Vector2::new(v[1], v[1]-v[0]); // System {v0 =  y1
    ///                                                                  //        {v1 =  y1-y0
    /// let (x0, v0) = (0., Vector2::new(1., 1.));
    /// let h = 1e-3; // step size
    ///
    /// const SOLUTION: f64 = 1.7932509391963074; // Exact solution for y(x)
    ///
    /// let solution = ODESolver::new(f, x0, v0, h)
    ///                 .solve(1.).unwrap();
    ///
    /// assert!((solution[0] - SOLUTION) <= 1e-3);
    /// ```
    pub fn solve(&self, x_end: T) -> SolverResult<V> {
        let mut x = self.x0;
        let mut y = self.y0;
        let steps = T::to_usize(&((x_end - self.x0) / self.h)).unwrap_or(0);
        if steps == 0 {
            return Err(SolverError::IncorrectInput);
        }

        for _ in 1..steps {
            y = (self.method)(&self, x, y);
            x = x + self.h;
        }

        Ok(y)
    }

    /// Change the solver's step size
    ///
    /// ## Examples
    /// ```
    /// use eqsolver::ODESolver;
    /// // ODE: y' = -y. where y(0) = 1
    /// let f = |x: f64, y: f64| -y;
    /// let (x0, y0) = (0., 1.);
    /// let x_end = 1.;
    /// let h = 0.1; // step size
    ///
    /// let mut solver = ODESolver::new(f, x0, y0, h);
    ///
    /// # let solution = solver.solve(x_end).unwrap();
    /// # assert!((solution - (-1_f64).exp()) > 1e-3); // Error too big!
    /// let solution = solver
    ///                 .with_step_size(0.001)
    ///                 .solve(x_end); // This changes solver's step size until changed again
    /// # assert!((solution.unwrap() - (-1_f64).exp()) <= 1e-3);
    /// ```
    pub fn with_step_size(&mut self, h: T) -> &mut Self {
        self.h = h;
        self.half_h = h / T::from(2.).unwrap();
        self
    }

    /// Change step size given amount of steps
    ///
    /// Specify the end value you want to evaluate at and the amount steps to be taken to that end value.
    /// This will change the inner step size according to that given data.
    ///
    /// ## Examples
    /// ```
    /// use eqsolver::ODESolver;
    /// // ODE: y' = -y. where y(0) = 1
    /// let f = |x: f64, y: f64| -y;
    /// let (x0, y0) = (0., 1.);
    /// let x_end = 1.;
    /// let h = 0.1; // step size
    ///
    /// let mut solver = ODESolver::new(f, x0, y0, h);
    /// # let solution = solver.solve(x_end).unwrap();
    /// # assert!((solution - (-1_f64).exp()) > 1e-3); // Error too big!
    ///
    /// let solution = solver
    ///                 .with_steps(x_end, 1000)
    ///                 .solve(x_end).unwrap(); // This changes solver's step size until changed again
    /// # assert!((solution - (-1_f64).exp()) <= 1e-3);
    /// ```
    pub fn with_steps(&mut self, x_end: T, steps: usize) -> &mut Self {
        self.h = (x_end - self.x0) / T::from(steps).unwrap();
        self.half_h = self.h / T::from(2.).unwrap();
        self
    }

    /// Specify the method to use for solving the ODE.
    ///
    /// There are 3 methods available: Euler Forward, Heun and Runge-Kutta 4.
    /// The default is Runge-Kutta 4.
    ///
    /// ```
    /// use eqsolver::{ODESolver, ODESolverMethod};
    /// // ODE: y' = -y. where y(0) = 1
    /// let f = |x: f64, y: f64| -y;
    /// let (x0, y0) = (0., 1.);
    /// let x_end = 1.;
    /// let h = 0.1; // step size
    /// # let exact = (-1_f64).exp();
    ///
    /// let mut solver = ODESolver::new(f, x0, y0, h);
    ///
    /// # let rk4_err = (solver.solve(x_end).unwrap() - exact).abs();
    /// let solution = solver
    ///                 .with_method(ODESolverMethod::EulerForward)
    ///                 .solve(x_end);
    ///
    /// # assert!((solution.unwrap() - (-1_f64).exp()) <= 0.1);
    /// ```
    pub fn with_method(&mut self, method: ODESolverMethod) -> &mut Self {
        match method {
            ODESolverMethod::EulerForward => {
                self.method = Self::euler_step;
            }
            ODESolverMethod::Heun => {
                self.method = Self::heun_step;
            }
            ODESolverMethod::RungeKutta4 => {
                self.method = Self::rk4_step;
            }
        }
        self
    }

    /// === PRIVATE FUNCTIONS: A step in the different methods available ===
    fn euler_step(&self, x: T, y: V) -> V {
        y + (self.f)(x, y) * self.h
    }

    fn heun_step(&self, x: T, y: V) -> V {
        let y1 = y + (self.f)(x, y) * self.h;
        y + (y1 + (self.f)(x + self.h, y1)) * self.half_h
    }

    fn rk4_step(&self, x: T, y: V) -> V {
        let k1 = (self.f)(x, y);
        let k2 = (self.f)(x + self.half_h, y + k1 * self.half_h);
        let k3 = (self.f)(x + self.half_h, y + k2 * self.half_h);
        let k4 = (self.f)(x + self.h, y + k3 * self.h);

        y + (k1 + k2 + k2 + k3 + k3 + k4) * (self.h / T::from(6_f64).unwrap())
    }
}
