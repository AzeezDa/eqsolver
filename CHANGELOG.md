# Change Log
All notable changes to this project will be documented in this file, starting from version 0.2.0.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

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

## [0.3.0] - 2025-07-19

### Added
- [Newton-Cotes Integration](https://en.wikipedia.org/wiki/Newton%E2%80%93Cotes_formulas) for
  integrating function with single-variable domains.
- [Monte Carlo Integration](https://en.wikipedia.org/wiki/Monte_Carlo_integration) for integrating
  functions with single or multiple variable domains.
- [MISER Integration](https://doi.org/10.1063/1.4822899) for integrating
  functions with single or multiple variable domains.
- A new output type (`MeanVariance`) for algorithms whose output are random/stochastic.
- APL2.0 is now a licence of `eqsolver` together with MIT.

### Changed
- Upgraded to `rand 0.9.1`, `rand_distr 0.5.1`, and `criterion 0.6.0`.
- Moved [`/tests`](./tests) to be in the root of the project.
- Restructured where in the modules global definitions (constants, types, etc.) are.
- Updated dates in licences
