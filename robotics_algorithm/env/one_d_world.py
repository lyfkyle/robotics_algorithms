from typing_extensions import override

import matplotlib.pyplot as plt
import random
import numpy as np
import math
import numpy as np
from numpy.random import randn

from robotics_algorithm.robot.double_integrator import DoubleIntegrator
from robotics_algorithm.env.base_env import (
    ContinuousEnv,
    StochasticEnv,
    PartiallyObservableEnv,
    FunctionType,
    NoiseType,
)


class DoubleIntegratorEnv(ContinuousEnv, StochasticEnv, PartiallyObservableEnv):
    """A DoubleIntegrator robot is tasked to stop at the goal state which is at zero position  with zero velocity.

    State: [pos, vel]
    Action: [acceleration]

    Continuous state space.
    Continuous action space.
    Deterministic transition.
    Fully observable.

    Linear state transition function.
    Quadratic cost.
    """

    def __init__(
        self, size=10, use_discrete_time_model=True, observation_noise_std=0.01, state_transition_noise_std=0.01
    ):
        super().__init__()

        self.size = size

        self.robot_model = DoubleIntegrator(use_discrete_time_model=use_discrete_time_model)
        self.state_space = [[-self.size / 2, float("-inf")], [self.size / 2, float("inf")]]  # pos and vel
        self.action_space = [float("-inf"), float("inf")]  # accel

        # declare linear state transition
        # x_dot = Ax + Bu
        self.state_transition_type = FunctionType.LINEAR.value
        self.A = self.robot_model.A
        self.B = self.robot_model.B

        # declare quadratic cost
        # L = x.T @ Q @ x + u.T @ R @ u
        self.reward_func_type = FunctionType.QUADRATIC.value
        self.Q = np.array([[1, 0], [0, 1]])
        self.R = np.array([[1]])

        # state transition noise model
        self.state_transition_noise_type = NoiseType.GAUSSIAN.value
        self.state_transition_noise_std = state_transition_noise_std

        # Measurement noise model
        # self.R = np.array([[process_var, 0], [0, process_var]], dtype=np.float32)
        self.observation_noise_type = NoiseType.GAUSSIAN.value
        self.observation_noise_std = observation_noise_std
        self.H = np.eye(2, dtype=np.float32)
        # self.Q = np.array([[observation_noise_std]], dtype=np.float32)

        self.path = None

    @override
    def reset(self):
        self.start_state = [random.uniform(self.state_space[0][0], self.state_space[1][0]), 0]
        self.goal_state = [0, 0]  # fixed
        self.cur_state = self.start_state

    @override
    def sample_state_transition(self, state: list, action: list) -> tuple[list, float, bool, bool, dict]:
        new_state = np.array(self.robot_model.control(state, action, dt=0.01))
        new_state = new_state + self.state_transition_noise_std * np.random.randn(*new_state.shape)  # add noise
        new_state = new_state.reshape(-1).tolist()

        # Check bounds
        if (
            new_state[0] <= self.state_space[0][0]
            or new_state[0] >= self.state_space[1][0]
            or new_state[1] <= self.state_space[0][1]
            or new_state[1] >= self.state_space[1][1]
        ):
            return state, -100, True, False, {"success": False}

        # Check goal state reached for termination
        term = False
        if np.allclose(np.array(new_state), np.array(self.goal_state), atol=1e-4):
            term = True

        self.cur_state = new_state

        return new_state, self.reward_func(state, action), term, False, {}

    @override
    def sample_observation(self, state):
        # simulate measuring the position with noise
        state = np.array(state).reshape(-1, 1)
        meas = self.H @ state + self.observation_noise_std * np.random.randn(*state.shape)
        return meas.reshape(-1).tolist()

    @override
    def reward_func(self, state: list, action: list | None = None) -> float:
        # quadratic reward func
        state = np.array(state)
        action = np.array(action)

        cost = state.T @ self.Q @ state + action.T @ self.R @ action
        return -cost

    @override
    def render(self):
        fig, ax = plt.subplots(2, 1, dpi=100)
        ax[0].plot(0, self.start_state[0], "o")
        ax[1].plot(0, self.start_state[1], "o")

        if self.path:
            for i, state in enumerate(self.path):
                ax[0].plot(i, state[0], "o")
                ax[1].plot(i, state[1], "o")

        ax[0].set_ylim(-self.size / 2, self.size / 2)
        plt.show()

    def add_path(self, path):
        self.path = path
