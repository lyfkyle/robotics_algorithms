import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from typing_extensions import override

from robotics_algorithm.env.base_env import (
    DistributionType,
    PartiallyObservableEnv,
    StochasticEnv,
)
from robotics_algorithm.env.continuous_2d.diff_drive_2d_env_base import DiffDrive2DEnv


class DiffDrive2DLocalization(DiffDrive2DEnv, StochasticEnv, PartiallyObservableEnv):
    """A differential drive robot must reach goal state in a 2d maze with obstacles.
       Adds transition stochasticity and partial observability to the original DiffDrive2DEnv.

    State: [x, y, theta]
    Action: [lin_vel, ang_vel]

    Continuous state space.
    Continuous action space.
    Stochastic transition.
    Partially observable.

    In this mode, user commands the robot to move arbitrarily, the environment will provide
    noisy measurement of robot's position.
    """

    def __init__(
        self,
        size=10,
        robot_radius=0.2,
        action_dt=1.0,
        ref_path=None,
        discrete_action=False,
        state_transition_noise_std=[0.01, 0.01, 0.01],
        obs_noise_std=[0.05, 0.05, 0.05],
    ):
        # NOTE: MRO ensures DiffDrive2DEnv methods are always checked first. However, during init, we manually init
        #       DiffDrive2DEnv last.
        StochasticEnv.__init__(self)
        PartiallyObservableEnv.__init__(self)
        DiffDrive2DEnv.__init__(self, size, robot_radius, action_dt, ref_path, discrete_action)

        self.state_transition_dist_type = DistributionType.GAUSSIAN.value
        self.state_transition_noise_var = np.array(state_transition_noise_std) ** 2
        self.state_transition_cov_matrix = np.eye(self.state_space.state_size)
        np.fill_diagonal(self.state_transition_cov_matrix, self.state_transition_noise_var)

        self.observation_dist_type = DistributionType.GAUSSIAN.value
        self.observation_var = np.array(obs_noise_std) ** 2
        self.obs_cov_matrix = np.eye(self.state_space.state_size)
        np.fill_diagonal(self.obs_cov_matrix, self.observation_var)

    @override
    def state_transition_func(self, state, action) -> tuple[np.ndarray, np.ndarray]:
        mean_new_state = DiffDrive2DEnv.state_transition_func(self, state, action)
        return mean_new_state, self.state_transition_noise_var.

    @override
    def observation_func(self, state) -> tuple[np.ndarray, np.ndarray]:
        return state, self.observation_var.

    @override
    def reward_func(self, state, action = None, new_state = None):
        # Reward function is not defined in localization environment.
        return 0

    def state_transition_jacobian(self, state, action):
        """
        Return jocobian matrix of transition function wrt state at control
        @state, state
        @action, action
        @return A, jacobian matrix
        """
        lin_vel = action[0]
        theta = state[2]

        self.F = np.array(
            [
                [1, 0, -lin_vel * np.sin(theta) * self.action_dt],
                [0, 1, lin_vel * np.cos(theta) * self.action_dt],
                [0, 0, 1],
            ]
        )
        return self.F

    def observation_jacobian(self, state, observation):
        """
        Return observation_jacobian matrix.

        @state, state
        @observation, observation
        """
        self.H = np.eye(3, dtype=np.float32)

        return self.H

    @override
    def get_state_transition_prob(self, state: np.ndarray, action: np.ndarray, new_state: np.ndarray) -> float:
        # NOTE: since the state is continuous, we can only returns the pdf value. It is up to the consumer to
        #       normalize to get a probability.
        mean, var = self.state_transition_func(state, action)
        return multivariate_normal.pdf(new_state, mean=mean, cov=np.diag(var))

    @override
    def get_observation_prob(self, state: np.ndarray, observation: np.ndarray) -> float:
        # NOTE: since the state is continuous, we can only returns the pdf value. It is up to the consumer to
        #       normalize to get a probability.
        mean, var = self.observation_func(state)
        return multivariate_normal.pdf(observation, mean=mean, cov=np.diag(var))

    def add_state_path(self, path, id=None):
        """Add a path for visualization.

        Args:
            path (np.ndarray(np.ndarray)): the path.
        """
        if id is None:
            self.path = path
        else:
            self.path_dict[id] = path

    def render(self, draw_start=True, draw_goal=False):
        if self.interactive_viz:
            if not self._fig_created:
                plt.ion()
                plt.figure(figsize=(10, 10), dpi=100)
                self._fig_created = True
        else:
            plt.figure(figsize=(10, 10), dpi=100)

        plt.clf()
        s = 1000 / self.size / 2
        for obstacle in self.obstacles:
            x, y, r = obstacle
            plt.scatter(
                x,
                y,
                s=(s * r * 2) ** 2,
                c='black',
                marker='o',
            )
        if draw_goal:
            plt.scatter(
                self.goal_state[0],
                self.goal_state[1],
                s=(s * self.robot_radius * 2) ** 2,
                c='red',
                marker='o',
            )
        if draw_start:
            plt.scatter(
                self.start_state[0],
                self.start_state[1],
                s=(s * self.robot_radius * 2) ** 2,
                c='yellow',
                marker='o',
            )
        if self.path is not None:
            plt.plot(
                [s[0] for s in self.path],
                [s[1] for s in self.path],
                # s=(s * self.robot_radius * 2) ** 2,
                ms=(s * 0.01 * 2),
                c='green',
                marker='o',
            )

        for key, path in self.path_dict.items():
            plt.plot(
                [s[0] for s in path],
                [s[1] for s in path],
                # s=(s * self.robot_radius * 2) ** 2,
                ms=(s * 0.01 * 2),
                marker='o',
                label=key,
            )

        plt.xlim(0, self.size)
        plt.xticks(np.arange(self.size))
        plt.ylim(0, self.size)
        plt.yticks(np.arange(self.size))
        plt.legend()

        if self.interactive_viz:
            plt.pause(0.01)
        else:
            plt.show()