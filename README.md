# robotics_algorithms

![logo](/doc/logo.jpg "logo")

This repository contains algorithms related to robotics for personal learning purpose. The motivation is to have
a single source of truth with clear documentation.

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
- [ ] Learning