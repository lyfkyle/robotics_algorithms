import random
from collections import defaultdict
from typing import Any
from typing_extensions import override
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from robotics_algorithm.env.base_env import ContinuousEnv
from robotics_algorithm.robot.differential_drive import DiffDrive


DEFAULT_OBSTACLES = [[2, 2, 0.5], [5, 5, 1], [3, 8, 0.5], [8, 3, 1]]
DEFAULT_START = [0.5, 0.5, 0]
DEFAULT_GOAL = [9, 9, math.radians(90)]


class TwoDMaze(ContinuousEnv):
    FREE_SPACE = 0
    OBSTACLE = 1
    START = 2
    GOAL = 3
    PATH = 4
    WAYPOINT = 5
    MAX_POINT_TYPE = 6

    def __init__(self, size=10):
        super().__init__()

        self.size = size
        self.maze = np.full((size, size), TwoDMaze.FREE_SPACE)

        self.colour_map = colors.ListedColormap(["white", "black", "red", "blue", "green", "yellow"])
        bounds = [
            TwoDMaze.FREE_SPACE,
            TwoDMaze.OBSTACLE,
            TwoDMaze.START,
            TwoDMaze.GOAL,
            TwoDMaze.PATH,
            TwoDMaze.WAYPOINT,
            TwoDMaze.MAX_POINT_TYPE,
        ]
        self.norm = colors.BoundaryNorm(bounds, self.colour_map.N)

        self.diff_drive_robot = DiffDrive(wheel_dist=0.2, wheel_radius=0.05)
        self.robot_radius = 0.2

        self._sample_actions = [
            (0.5, 0),
            (0.5, math.radians(30)),
            (0.5, -math.radians(30)),
            (0.25, 0),
            (0.25, math.radians(30)),
            (0.25, -math.radians(30)),
        ]

        self.path = None

    @override
    def reset(self, random_env=True):
        if random_env:
            self._random_obstacles()
            self.start_state = self._random_valid_state()
            self.goal_state = self._random_valid_state()
        else:
            self.obstacles = DEFAULT_OBSTACLES
            self.start_state = DEFAULT_START
            self.goal_state = DEFAULT_GOAL

    @override
    def state_transition_func(self, state: Any, action: Any) -> tuple[Any, float, bool, bool, dict]:
        new_state = self.diff_drive_robot.control_velocity(state, action[0], action[1], dt=1.0).tolist()

        if new_state[0] <= 0 or new_state[1] >= self.size or new_state[1] <= 0 or new_state[1] >= self.size:
            return state, 0, True, False, {"success": False}

        if self._is_in_collision(new_state):
            return state, 0, True, False, {"success": False}

        return tuple(new_state), 1, False, False, {}

    @override
    def get_available_actions(self, state: Any) -> list[Any]:
        return self._sample_actions

    def _random_obstacles(self, num_of_obstacles=5):
        self.obstacles = []
        for _ in range(num_of_obstacles):
            obstacle = np.random.uniform([0, 0, 0.1], [self.size, self.size, 1])
            self.obstacles.append(obstacle.tolist())

    def _is_in_collision(self, state):
        for obstacle in self.obstacles:
            if np.linalg.norm(np.array(state[:2]) - np.array(obstacle[:2])) <= obstacle[2] + self.robot_radius:
                return True

        return False

    def _random_valid_state(self):
        while True:
            robot_pos = np.random.uniform([0, 0, -math.pi], [self.size, self.size, math.pi])
            if not self._is_in_collision(robot_pos):
                break

        return robot_pos

    def add_path(self, path):
        interpolated_path = [self.start_state]

        state = self.start_state
        # Run simulation
        for action in path:
            for _ in range(10):
                state = self.diff_drive_robot.control_velocity(state, action[0], action[1], dt = 0.1)
                interpolated_path.append(state)

        self.path = interpolated_path

    def render(self):
        plt.figure(figsize=(10, 10), dpi=100)
        s = 1000 / self.size / 2
        for obstacle in self.obstacles:
            x, y, r = obstacle
            plt.scatter(
                x,
                y,
                s=(s * r * 2) ** 2,
                c="black",
                marker="o",
            )
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

        # plt.grid()
        plt.xlim(0, self.size)
        plt.xticks(np.arange(self.size))
        plt.ylim(0, self.size)
        plt.yticks(np.arange(self.size))
        plt.show()
