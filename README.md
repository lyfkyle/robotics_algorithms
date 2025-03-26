# robotics_algorithms

![logo](/doc/TIAGo_05.jpg "logo")

This repository contains my pure-python implementation of essential algorithms that make a mobile manipulator (and other robots) move.

While it is true that a lot of algorithms have been implemented by other projects, this repo serves these main benefits:

1. It achieves a good balance of width and depth. It covers a wide range of topics, while focusing on only key algorithms in each field.
2. It is implemented with modular structure that cleanly separates algorithms from problems (like OMPL), at the same time emphasizing connection between different algorithms. For example, the design reflects that planing under uncertainties, optimal control and RL share the same underlying MDP formulation.
3. Serves as a single source of truth of various algorithms so that I no longer need to search all over the Internet.

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
python examples/planning/path_planning/test_a_star.py
```

## News

- 26/03/2025: Added DWA (v0.11.2)
- 20/02/2025: Added LQR and convex MPC for planar quadrotor (v0.11.1).
- 17/02/2025: Added convex MPC, inverted pendulum and more path following examples (v0.11.0).
- 09/02/2025: Added cost-aware differential drive path planning (v0.10.0).

## Status

This repository is undergoing significant development. Here is the status checklist.

Algorithms

- [ ] Robot Kinematic and Dynamics
  - [x] Differential drive
  - [x] Cartpole
  - [x] Double Integrator
  - [x] Inverted Pendulum
  - [ ] Arm
    - [ ] FK and IK
  - [ ] Car
  - [x] Planar Quadrotor
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
    - [ ] Graph-based SLAM
- [ ] Planning
  - [x] Path Planning
    - [x] Dijkstra
    - [x] A-star
    - [x] Hybrid A-star
  - [ ] Motion Planning
    - [x] PRM
    - [x] RRT / RRT-Connect
    - [x] RRT\*
    - [ ] RRT\*-Connect
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
    - [x] Convex-MPC
  - [ ] Trajectory optimization
    - [ ] iLQR
  - [ ] Domain-specific Path Follow Control
    - [ ] Regulated Pure Pursuit
    - [x] Dynamic Window Approach
    - [ ] Time-elastic Band
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
    - [x] With obstacle-distance cost
  - [x] 2D localization
  - [ ] 2D SLAM
  - [ ] Multi-arm bandits (POMDP)
  - [ ] Tiger (POMDP)

In addition to the algorithm itself, also implement several realistic robotics problems that often require additional
domain-specific components and strategies.

- [x] Path planning for differential drive robot using Hybrid A Star with original heuristics and cost weighted distance measure.

## Known issues

- [ ] EKF gives high localisation error at some instances.
- [ ] MCTS is not stable.
- [ ] Recursive feasibility in convex MPC
