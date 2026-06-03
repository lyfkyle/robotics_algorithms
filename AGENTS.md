# Project Context

This repository is a pure-Python collection of representative robotics algorithms. It is intended to be both a
reference implementation and a runnable playground for algorithms used in planning, control, state estimation,
reinforcement learning, robot dynamics, and robotics environments.

## Repository Shape

- `robotics_algorithm/` contains the importable Python package.
- `examples/` contains runnable scripts that demonstrate and lightly test individual algorithms.
- `robotics_algorithm/env/` defines shared environment abstractions and concrete robotics/MDP environments.
- `robotics_algorithm/robot/` contains robot dynamics and kinematics models.
- `robotics_algorithm/planning/` contains path planning, motion planning, and MDP planning algorithms.
- `robotics_algorithm/control/` contains classical, optimal, and path-following controllers.
- `robotics_algorithm/state_estimation/` contains Bayes/Kalman/particle-filter localization algorithms.
- `robotics_algorithm/learning/` contains reinforcement learning algorithms.
- `robotics_algorithm/utils/` contains shared math, MDP, tree, and model helpers.

## Design Intent

The codebase separates algorithms from problems. Algorithms should generally consume a `BaseEnv` or robot model rather
than baking in a specific world. The central abstractions are:

- `BaseEnv` for state/action/observation spaces, transitions, rewards, observability, and Gymnasium-style stepping.
- `DiscreteSpace` and `ContinuousSpace` for declaring state, action, and observation domains.
- `Robot` subclasses for dynamics simulation and local linearization.

When adding new algorithms, prefer matching the existing pattern: validate the environment capabilities in `__init__`,
store the environment/model, and expose a small `run(...)` or controller-style method.

## Development Commands

- Install locally with `pip install -e .`.
- Run examples directly, such as `python examples/planning/path_planning/test_a_star.py`.
- There is no central test runner configured in `pyproject.toml`; examples are the main executable checks.
- Formatting style is Ruff-compatible with line length `120`, single quotes, and space indentation.

## General coding guid

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## #3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## Dependency Notes

The package targets Python `>=3.10`. Core dependencies are scientific Python and robotics/demo tooling:

- `numpy`
- `matplotlib`
- `scikit-learn`
- `networkx`
- `pygame`
- `cvxpy`
- `typing_extensions`

Avoid introducing heavyweight dependencies unless the algorithm strongly justifies it and the rest of the repository has
no suitable pattern already.

## Implementation Guidelines

- Keep algorithm implementations readable and educational; this repo favors clear reference code over highly optimized
  framework code.
- Use NumPy arrays for states, actions, observations, and trajectories unless an existing local API requires otherwise.
- Preserve the existing modular boundary between algorithm logic, robot dynamics, and environment-specific details.
- Prefer structured environment methods such as `sample_state_transition`, `state_transition_func`, `reward_func`,
  `observation_func`, and `linearize_state_transition` instead of duplicating environment mechanics inside algorithms.
- If an algorithm requires assumptions, assert or check them near construction, following existing files such as A\*,
  Dijkstra, LQR, and MPC implementations.
- Add or update an example script when adding a new algorithm or behavior that benefits from visual/manual validation.
