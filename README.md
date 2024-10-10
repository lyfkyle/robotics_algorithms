# robotics_algorithms

![logo](/doc/logo.jpg "logo")

This repository contains pure-python implementation for essential robotics algorithms. 

The main benefits are:
1. Have a single source of truth of various algorithms with clear explanation.
2. Implemented with clear separation between dynamics, environment and algorithm, emphasizing that a lot of algorithms, e.g planing under uncertainties, optimal control, share the same underlying problem formulation，eg. MDP. 

## Scope

It should include popular and representative algorithms from robot dynamics, state estimation, planning, control and
learning.

## How to use

- Run `pip install -e .`
- Run various scripts inside examples folder.

## Status
This repository is undergoing significant development. Here is the status checklist.

- [ ] Robot Dynamics
  - [x] Differential drive
  - [x] Cartpole
  - [x] Double Integrator
  - [ ] Arm
  - [ ] Car
  - [ ] Quadrotor
  - [ ] Quadruped
- [ ] State Estimation
  - [x] Discrete bayes filter
  - [x] Kalman filter
  - [ ] Extended Kalman filter
  - [ ] Particle filter
- [ ] Planing
  - [x] Discrete Planning
    - [x] Dijkstra
    - [x] A-star
  - [x] Motion Planning
    - [x] RRT / RRT-Connect / RRT*
    - [x] PRM
    - [ ] BIT*
  - [x] MDP
    - [x] Value iteration
    - [x] policy iteration
    - [x] Policy tree search
    - [x] MCTS
  - [ ] POMDP
    - [ ] SARSOP
    - [ ] DESPOT
- [ ] Control
  - [x] Classical control (PID)
  - [x] LQR
  - [x] MPPI
  - [x] CEM-MPC
- [ ] Imitation learning
- [ ] Reinforcement learning
  - [ ] Tabular
  - [ ] Function approximation 
- [ ] Other Environment
  - [x] Frozen lake
  - [x] Cliff walking
  - [x] Windy gridworld 
  - [x] 1D navigation
    - [x] Deterministic and fully-observable
    - [x] Stochastic and partially-observable
  - [x] 2D navigation
    - [x] Deterministic and fully-observable
    - [ ] Stochastic and partially-observable
  - [ ] 2D localization
  - [ ] 2D SLAM  
  - [ ] Multi-arm bandits
  - [ ] Tiger 
