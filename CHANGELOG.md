# Changelog

## [v0.11.1]

### Added

- Added planar quadrotor
- Added LQR and convex MPC example for planar quadrotor

### Changed

- Changed all state, action, observation to be numpy array by default (cont).
- Separated is_state_terminal() from get_state_info().

## [v0.11.0]

### Added

- Added convex MPC.
- Added inverted pendulum environment.
- Added a robot parent class for different robots.
- Added LQR and convex MPC for path follow and inverted pendulum.

### Changed

- Refactored folder structure to better group different components.
- Changed all state, action, observation to be numpy array by default.

### Fixed

- Fixed diff_drive_2d_planning traversal cost calculation so euclidean distance is properly underestimating true cost.

## [v0.10.0]

### Added

- Added cost-aware differential drive path planning

### Fixed

- Fixed a bug in Hybrid A-star

### Changed

- Split continuous 2d env into separate classes for planning, control and localization, each with its own file.

## [v0.9.1]

### Added

- Added changelog

### Changed

- Minor comment and code change

### Removed

- Trademark sign previously shown after the project description in version
  0.3.0

## [v0.9.0]

### Added

- Added AMCL

## [0.8.3]

### Added

- Added RL tabular methods
