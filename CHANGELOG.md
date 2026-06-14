# Change Log
All notable changes to this project will be documented in this file, starting from version 0.2.0.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [0.4.1] - 2026-06-14

### Changed
Upgraded the following dependencies
- `nalgebra` to `0.35.0`
- `rand` to `0.10.1`
- `rand_distr` to `0.6.0`
- `thiserror` to `2.0.18`
- `criterion` to `0.8.2`
- `rand_chacha` to `0.10.0`

## [0.4.0] - 2025-12-19

### Added
- More workflow checks:
  - Correction of formatting via `cargo fmt -- --check`;
  - clippy via `cargo clippy`;
  - Documentation generation with `cargo doc`;
  - Build via `cargo build --verbose`;
  - Tests via `cargo test`; and
  - Examples running in debug and release mode.

### Changed
- Upgraded to `rand 0.9.2`, `nalgebra 0.34.1`, and `critrion 0.8.1`
- `SolverError` now derives `std::error::Error` using the [`thiserror`](https://docs.rs/thiserror/latest/thiserror/) crate.
- Added variant fields to some enum fields of `SolverError`:
  - `MaxIterReached(usize)`: Contains the number maximum iteration count plus 1;
     it is the number of iterations the solver was at when it output the error.
  - `IncorrectInput { details: &'static str }`: Contains an informative string about why the input was incorrect; and
  - `ExternalError(String)`: Contains a `String` representation (using `to_string()`) of the external error.
- The `cargo fmt` and `cargo clippy` were ran and the suggestions from the latter were fixed.
- Workflows now work in the `dev` branch in addition to `master`


## [0.3.0] - 2025-07-19

### Added
- [Newton-Cotes Integration](https://en.wikipedia.org/wiki/Newton%E2%80%93Cotes_formulas) for
  integrating function with single-variable domains.
- [Monte Carlo Integration](https://en.wikipedia.org/wiki/Monte_Carlo_integration) for integrating
  functions with single or multiple variable domains.
- [MISER Integration](https://doi.org/10.1063/1.4822899) for integrating
  functions with single or multiple variable domains.
- A new output type (`MeanVariance`) for algorithms whose output are random/stochastic.
- APL2.0 is now a license of `eqsolver` together with MIT.

### Changed
- Upgraded to `rand 0.9.1`, `rand_distr 0.5.1`, and `criterion 0.6.0`.
- Moved [`/tests`](./tests) to be in the root of the project.
- Restructured where in the modules global definitions (constants, types, etc.) are.
- Updated dates in licenses


## [0.2.0] - 2024-07-14

### Added
- [Levenberg-Marquardt method](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm) for non-linear least squared problems
- [Particle Swarm Optimisation](https://en.wikipedia.org/wiki/Particle_swarm_optimization) and [Cross-Entropy method](https://en.wikipedia.org/wiki/Cross-entropy) for finding global minimums of objective functions
- Benchmarks using the Rust library `criterion`
- A directory for examples

### Changed
- Upgraded to `nalgebra 0.33.0` and `num_traits 0.2.19`
- Better module/file structure for the source code

### Fixed
- Replaced unnecessary `.clone()`s with borrowing
