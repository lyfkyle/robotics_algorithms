import math

import matplotlib.pyplot as plt
import numpy as np
from typing_extensions import override

from robotics_algorithm.env.base_env import ContinuousSpace, DeterministicEnv, FullyObservableEnv, FunctionType
from robotics_algorithm.robot.pendulum import Pendulum


class InvertedPendulumEnv(DeterministicEnv, FullyObservableEnv):
    """Inverted pendulum environment.

    It has two modes, swing up or keep upright.
    - In swing up mode, the goal is to swing up the pendulum from downward position to upright position. The reward is a quadratic cost that penalizes deviation from the upright position and control effort, similar to the one used in OpenAI Gym Pendulum environment.
    - In keep upright mode, the goal is to keep the pendulum upright. The reward is 1 for every step the pendulum is upright, and -1 if the pendulum falls down. The episode terminates when the pendulum falls down.

    State: [theta, theta_dot]
    Action: [torque]

    """

    def __init__(self, mode: str = 'keep_upright', dt=0.01):
        super().__init__()

        self.state_transition_func_type = FunctionType.LINEAR.value

        # state and action space
        self.state_space = ContinuousSpace(low=[-float('inf'), -float('inf')], high=[float('inf'), float('inf')])
        self.action_space = ContinuousSpace(low=[-1e10], high=[1e10])  # torque

        # reward
        assert mode in ['swing_up', 'keep_upright'], "mode must be either 'swing_up' or 'keep_upright'"
        self.mode = mode
        if self.mode == 'swing_up':
            # Follows https://gymnasium.farama.org/environments/classic_control/pendulum/
            self.reward_func_type = FunctionType.QUADRATIC.value
            self.Q = np.array([[1.0, 0], [0, 0.1]])
            self.R = np.array([[0.001]])

        # robot model
        self.robot_model = Pendulum()
        # self.action_dt = self.robot_model.dt
        self.action_dt = dt
        self._implicit_step_cnt = int(
            self.action_dt / self.robot_model.dt
        )  # number of internal robot steps per environment step
        self._ref_state = np.array([-np.pi, 0])

        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.max_steps = 500

        # others
        self._fig_created = False

    @override
    def reset(self):
        if self.mode == 'keep_upright':
            self.cur_state = np.random.randn(2) * 0.1  # near-upright position
        else:
            self.cur_state = np.array([3.1, 0.0])  # downward position
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
        for _ in range(self._implicit_step_cnt):
            state = self.robot_model.control(state, action)

        return state

    @override
    def reward_func(self, state: np.ndarray, action: np.ndarray = None, new_state: np.ndarray = None) -> float:
        terminated = self.is_state_terminal(new_state)

        # ! Should operate on state error. However, since the goal state is [0, 0], we can directly use new_state.
        if self.mode == 'swing_up':
            reward = -new_state.T @ self.Q @ new_state - action.T @ self.R @ action
        else:  # keep upright
            if not terminated:
                reward = 1.0
            else:
                reward = -1.0

        return reward

    @override
    def is_state_terminal(self, state):
        theta = state[0]

        term = False
        if self.mode == 'keep_upright':
            term = bool(theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians)

        return term

    @override
    def state_transition_jacobian(self, state, action):
        self.f_x, self.f_u = self.robot_model.state_transition_jacobian(state, action)
        return self.f_x, self.f_u

    @override
    def reward_jacobian(self, state, action):
        if self.mode == 'swing_up':
            self.l_x = -2 * self.Q @ state
            self.l_u = -2 * self.R @ action
            return self.l_x, self.l_u
        else:
            raise NotImplementedError('First order approximation of reward is not implemented for keep_upright mode')

    @override
    def reward_hessian(self, state, action):
        if self.mode == 'swing_up':
            self.l_xx = -2 * self.Q
            self.l_uu = -2 * self.R
            return self.l_xx, self.l_uu
        else:
            raise NotImplementedError('Second order approximation of reward is not implemented for keep_upright mode')

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
