import math

import matplotlib.pyplot as plt
import numpy as np
from typing_extensions import override

from robotics_algorithm.env.base_env import (
    ContinuousSpace,
    DeterministicEnv,
    FullyObservableEnv,
    FunctionType
)
from robotics_algorithm.robot.pendulum import Pendulum


class InvertedPendulumEnv(DeterministicEnv, FullyObservableEnv):
    """Inverted pendulum environment.

    State: [theta, theta_dot]
    Action: [torque]

    """

    def __init__(self, quadratic_reward=True):
        super().__init__()

        self.state_transition_func_type = FunctionType.LINEAR.value

        # state and action space
        self.state_space = ContinuousSpace(low=[-float('inf'), -float('inf')], high=[float('inf'), float('inf')])
        self.action_space = ContinuousSpace(low=[-1e10], high=[1e10])  # torque

        # reward
        self.quadratic_reward = quadratic_reward
        if quadratic_reward:
            self.reward_func_type = FunctionType.QUADRATIC.value
            self.Q = np.array([[10, 0], [0, 10]])
            self.R = np.array([[1]])

        # robot model
        self.robot_model = Pendulum()
        self.action_dt = self.robot_model.dt
        self._ref_state = np.array([-np.pi, 0])

        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.max_steps = 1000

        # others
        self._fig_created = False

    @override
    def reset(self):
        self.cur_state = np.random.randn(2) * 0.1  # near-upright position
        self.goal_state = np.array([0, 0])  # upright position
        self.goal_action = np.zeros(1)

        self.step_cnt = 0

        return self.sample_observation(self.cur_state), {}

    @override
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        new_state, reward, term, trunc, info = super().step(action)

        self.step_cnt += 1
        trunc = self.step_cnt > self.max_steps  # set truncated flag

        return new_state, reward, term, trunc, info

    @override
    def state_transition_func(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        # compute next state
        new_state = self.robot_model.control(state, action)
        return new_state

    @override
    def reward_func(self, state: np.ndarray, action: np.ndarray = None, new_state: np.ndarray = None) -> float:
        theta = new_state[0]

        terminated = bool(theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians)

        if not terminated:
            if self.quadratic_reward:
                reward = -new_state.T @ self.Q @ new_state - action.T @ self.R @ action
            else:
                reward = 1.0
        else:
            reward = -1.0

        return reward

    @override
    def get_state_transition_info(self, state, action, new_state):
        theta = new_state[0]

        terminated = bool(theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians)

        return terminated, False, {}

    @override
    def linearize_state_transition(self, state, action):
        return self.robot_model.linearize_state_transition(state, action)

    @override
    def render(self):
        if not self._fig_created:
            # Create an animation of the pendulum
            plt.ion()
            plt.figure()

            self._fig_created = True

        plt.clf()
        theta = self.cur_state[0] + np.pi
        x_pendulum = self.robot_model.L * np.sin(theta)
        y_pendulum = -self.robot_model.L * np.cos(theta)
        plt.plot([0, x_pendulum], [0, y_pendulum], 'o-', lw=2)
        plt.xlim(-self.robot_model.L - 0.1, self.robot_model.L + 0.1)
        plt.ylim(-self.robot_model.L - 0.1, self.robot_model.L + 0.1)
        plt.pause(0.01)