# robotics_algorithms

![logo](/doc/logo.jpg "logo")

This repository contains pure-python implementation for essential robotics algorithms.

The main benefits are:

1. Have a single source of truth of various algorithms with clear explanation.
2. Implemented with clear separation between dynamics, environment and algorithm, emphasizing that a lot of algorithms, e.g planing under uncertainties, optimal control, share the same underlying problem formulation，eg. MDP.

## Scope

It should include popular and representative algorithms from robot dynamics, state estimation, planning, control and
learning.

## Requirement

python 3.10

## How to use

- Run `pip install -e .`
- Run various scripts inside examples folder.

For example, to run a\* to find the shortest path between start and goal in a grid world

```python
python examples/planning/test_a_star.py
```

## News

- 09/02/2025: Added cost-aware differential drive path planning
- Added [AMCL](https://docs.nav2.org/configuration/packages/configuring-amcl.html) from v0.9.0 onwards.

## Status

This repository is undergoing significant development. Here is the status checklist.

Algorithms

- [ ] Robot Dynamics
  - [x] Differential drive
  - [x] Cartpole
  - [x] Double Integrator
  - [ ] Arm
  - [ ] Car
  - [ ] Quadrotor
  - [ ] Quadruped
- [x] State Estimation
  - [x] Localizaion
    - [x] Discrete bayes filter
    - [x] Kalman filter
    - [x] Extended Kalman filter
    - [x] Particle filter (MCL)
    - [x] AMCL
  - [ ] SLAM
    - [ ] EKF SLAM
    - [ ] Fast SLAM
    - [ ] Graph SLAM
- [ ] Planning
  - [x] Path Planning
    - [x] Dijkstra
    - [x] A-star
    - [x] Hybrid A-star
  - [ ] Motion Planning
    - [x] RRT / RRT-Connect
    - [x] RRT\*
    - [ ] RRT\*-Connect
    - [x] PRM
    - [ ] Informed RRT\*
    - [ ] BIT\*
  - [x] MDP
    - [x] Value iteration
    - [x] policy iteration
    - [x] Policy tree search
    - [x] MCTS
  - [ ] POMDP
    - [ ] Belief tree search
    - [ ] SARSOP
    - [ ] DESPOT
- [ ] Control
  - [x] Classical control
    - [x] PID
  - [ ] Optimal Control
    - [x] LQR
    - [x] MPPI
    - [x] CEM-MPC
    - [ ] Convex-MPC
  - [ ] Other
    - [ ] Time-elastic Band
    - [ ] Regulated Pure Pursuit
- [ ] Imitation learning
  - [ ] ACT
  - [ ] Diffusion-policy
- [ ] Reinforcement learning
  - [ ] Tabular
    - [x] On-policy MC
    - [ ] Off-policy MC
    - [x] On-policy TD (SARSA)
    - [x] Off-policy TD (Q-learning)
  - [ ] Function approximation
- [ ] Environments
  - [x] Frozen lake (MDP)
  - [x] Cliff walking (MDP)
  - [x] Windy gridworld (MDP)
  - [x] 1D navigation with double integrator
    - [x] Deterministic and fully-observable
    - [x] Stochastic and partially-observable
  - [x] 2D navigation with omni-directional robot
    - [x] Deterministic and fully-observable
  - [x] 2D navigation with differential drive
    - [x] Deterministic and fully-observable
    - [x] Stochastic and partially-observable
    - [ ] With obstacle-distance cost
  - [ ] 2D localization
  - [ ] 2D SLAM
  - [ ] Multi-arm bandits (POMDP)
  - [ ] Tiger (POMDP)

In addition to the algorithm itself, also implement several realistic robotics problems that often require additional
domain-specific components and strategies.

- [x] Path planning for differential drive robot using Hybrid A Star with original heuristics and cost weighted distance measure.

## Known issues

- [ ] EKF gives high localisation error at some instances.
