from typing import Any
from typing_extensions import override

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

from robotics_algorithm.robot.double_integrator import DoubleIntegrator


class OneDWorld:
    def __init__(self, size=10):
        super().__init__()

        self.size = size

        self.robot_model = DoubleIntegrator()
        self.state_space = [[0, self.size], [float('-inf'), float('inf')]]  # pos and vel
        self.action_space = [float('-inf'), float('inf')]  # accel

    def reset(self):
        self.start_state = self._random_valid_state()
        self.goal_state = self._random_valid_state()

    @override
    def state_transition_func(self, state: Any, action: Any) -> tuple[Any, float, bool, bool, dict]:
        new_state = self.robot_model.control(state, action[0], action[1], dt=0.01).tolist()

        if new_state[0] <= 0 or new_state[1] >= self.size or new_state[1] <= 0 or new_state[1] >= self.size:
            return state, 0, True, False, {"success": False}

        return tuple(new_state), 1, False, False, {}

    @override
    def reward_func(self, state: tuple, new_state: tuple | None = None) -> float:



    def render(self):
        plt.figure(figsize=(10, 10), dpi=100)
        s = 1000 / self.size / 2
        plt.scatter(
            self.goal_state[0],
            self.goal_state[1],
            s=(s * self.robot_radius * 2) ** 2,
            c="red",
            marker="s",
        )
        plt.scatter(
            self.start_state[0],
            self.start_state[1],
            s=(s * self.robot_radius * 2) ** 2,
            c="yellow",
            marker="s",
        )

        if self.state_samples:
            for state in self.state_samples:
                plt.scatter(
                    state[0],
                    state[1],
                    s=(s * self.robot_radius * 2) ** 2,
                    # s=(s * 0.01 * 2) ** 2,
                    c="blue",
                    marker="o",
                )

        if self.path:
            for state in self.path:
                plt.scatter(
                    state[0],
                    state[1],
                    # s=(s * self.robot_radius * 2) ** 2,
                    s=(s * 0.01 * 2) ** 2,
                    c="green",
                    marker="o",
                )

        plt.xlim(0, self.size)
        plt.xticks(np.arange(self.size))
        plt.ylim(0, self.size)
        plt.yticks(np.arange(self.size))
        plt.show()

    def _random_valid_state(self):
        robot_state = np.random.uniform(self.state_space[0], self.state_space[1])
        return robot_state