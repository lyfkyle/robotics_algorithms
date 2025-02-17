from typing_extensions import override

import matplotlib.pyplot as plt
import numpy as np

from robotics_algorithm.robot.double_integrator import DoubleIntegrator
from robotics_algorithm.env.base_env import (
    ContinuousSpace,
    StochasticEnv,
    PartiallyObservableEnv,
    FunctionType,
    DistributionType,
)


class DoubleIntegratorEnv(StochasticEnv, PartiallyObservableEnv):
    """A DoubleIntegrator robot is tasked to stop at the goal state which is at zero position with zero velocity.

    State: [pos, vel]
    Action: [acceleration]

    Continuous state space.
    Continuous action space.
    Stochastic transition.
    Partially observable.

    Linear state transition function.
    Quadratic cost.
    """

    def __init__(self, size=10, observation_noise_std=0.01, state_transition_noise_std=0.01):
        super().__init__()

        self.size = size

        self.robot_model = DoubleIntegrator()
        self.state_space = ContinuousSpace(low=[-self.size / 2, -1e10], high=[self.size / 2, 1e10])  # pos and vel
        self.action_space = ContinuousSpace(low=[-1e10], high=[1e10])  # accel

        # declare linear state transition
        # x_dot = Ax + Bu
        self.state_transition_func_type = FunctionType.LINEAR.value
        self.A = self.robot_model.A
        self.B = self.robot_model.B

        # declare quadratic cost
        # L = x.T @ Q @ x + u.T @ R @ u
        self.reward_func_type = FunctionType.QUADRATIC.value
        self.Q = np.array([[1, 0], [0, 1]])  # state cost matrix
        self.R = np.array([[1]])  # control cost matrix

        # declare linear observation
        self.observation_func_type = FunctionType.LINEAR.value
        self.H = np.eye(2, dtype=np.float32)

        # state transition noise model
        self.state_transition_dist_type = DistributionType.GAUSSIAN.value
        self.state_transition_noise_std = state_transition_noise_std
        self.state_transition_noise_var = state_transition_noise_std * state_transition_noise_std
        self.state_transition_covariance_matrix = np.array(
            [
                [self.state_transition_noise_var, 0],  # only have noise on position
                [0, 1e-10],
            ],
            dtype=np.float32,
        )

        # observation noise model
        self.observation_dist_type = DistributionType.GAUSSIAN.value
        self.observation_noise_std = observation_noise_std
        var = observation_noise_std * observation_noise_std
        self.observation_covariance_matrix = np.array(
            [
                [var, 0],  # only have noise on position
                [0, 1e-10],
            ],
            dtype=np.float32,
        )

        self.path = None

    @override
    def reset(self):
        self.start_state = self.random_state()
        self.start_state[1] = 0.0  # zero velocity
        self.goal_state = [0, 0]  # fixed
        self.cur_state = np.copy(self.start_state)

        return self.sample_observation(self.cur_state), {}

    @override
    def state_transition_func(self, state: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        new_state_mean = self.robot_model.control(state, action, dt=0.01)
        new_state_var = [self.state_transition_noise_var, 1e-10]
        return new_state_mean, new_state_var

    @override
    def linearize_state_transition(self, state: np.ndarray, action: np.ndarray):
        return self.robot_model.linearize_state_transition(state, action)

    @override
    def linearize_observation(self, state, observation):
        return self.H

    @override
    def sample_observation(self, state):
        # simulate measuring the position with noise
        state = np.array(state).reshape(-1, 1)

        # noisy observation
        obs = self.H @ state + np.random.multivariate_normal(
            mean=[0, 0], cov=self.observation_covariance_matrix
        ).reshape(2, 1)

        return obs.reshape(-1)

    @override
    def reward_func(self, state: np.ndarray, action: np.ndarray = None, new_state: np.ndarray = None) -> float:
        # check bounds
        if (
            state[0] <= self.state_space.space[0][0]
            or state[0] >= self.state_space.space[1][0]
            or state[1] <= self.state_space.space[0][1]
            or state[1] >= self.state_space.space[1][1]
        ):
            return -100

        # quadratic reward func
        state = np.array(state)
        action = np.array(action)

        cost = state.T @ self.Q @ state + action.T @ self.R @ action
        return -cost

    @override
    def get_state_info(self, state):
        term = False
        info = {"success": False}
        # Check bounds
        if (
            state[0] <= self.state_space.space[0][0]
            or state[0] >= self.state_space.space[1][0]
            or state[1] <= self.state_space.space[0][1]
            or state[1] >= self.state_space.space[1][1]
        ):
            term = True

            return state, -100, True, False, {"success": False}

        # Check goal state reached for termination
        if np.allclose(np.array(state), np.array(self.goal_state), atol=1e-4):
            term = True
            info["success"] = True

        return term, False, info

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
