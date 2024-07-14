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
- Replaced unncessary `.clone()`s with borrowing