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


class TwoDMazeDiffDrive(ContinuousEnv):
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
        self.maze = np.full((size, size), TwoDMazeDiffDrive.FREE_SPACE)

        self.state_space = (np.array([0, 0, -math.pi]), np.array([self.size, self.size, math.pi]))
        self.action_space = (np.array([0, 0]), np.array([self.size, self.size]))

        self.colour_map = colors.ListedColormap(["white", "black", "red", "blue", "green", "yellow"])
        bounds = [
            TwoDMazeDiffDrive.FREE_SPACE,
            TwoDMazeDiffDrive.OBSTACLE,
            TwoDMazeDiffDrive.START,
            TwoDMazeDiffDrive.GOAL,
            TwoDMazeDiffDrive.PATH,
            TwoDMazeDiffDrive.WAYPOINT,
            TwoDMazeDiffDrive.MAX_POINT_TYPE,
        ]
        self.norm = colors.BoundaryNorm(bounds, self.colour_map.N)

        self.robot_model = DiffDrive(wheel_dist=0.2, wheel_radius=0.05)
        self._sample_actions = [
            (0.5, 0),
            (0.5, math.radians(30)),
            (0.5, -math.radians(30)),
            (0.25, 0),
            (0.25, math.radians(30)),
            (0.25, -math.radians(30)),
        ]
        self.robot_radius = 0.2

        self.path = None
        self.state_samples = None

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
        new_state = self.robot_model.control_velocity(state, action[0], action[1], dt=1.0).tolist()

        if new_state[0] <= 0 or new_state[1] >= self.size or new_state[1] <= 0 or new_state[1] >= self.size:
            return state, 0, True, False, {"success": False}

        if not self.is_state_valid(new_state):
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

    @override
    def is_state_valid(self, state):
        for obstacle in self.obstacles:
            if np.linalg.norm(np.array(state[:2]) - np.array(obstacle[:2])) <= obstacle[2] + self.robot_radius:
                return False

        return True

    def _random_valid_state(self):
        while True:
            robot_pos = np.random.uniform(self.state_space[0], self.state_space[1])
            if self.is_state_valid(robot_pos):
                break

        return robot_pos

    def add_path(self, path):
        interpolated_path = [self.start_state]

        state = self.start_state
        # Run simulation
        for action in path:
            for _ in range(10):
                state = self.robot_model.control_velocity(state, action[0], action[1], dt=0.1)
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

        # plt.grid()
        plt.xlim(0, self.size)
        plt.xticks(np.arange(self.size))
        plt.ylim(0, self.size)
        plt.yticks(np.arange(self.size))
        plt.show()


class TwoDMazeOmni(TwoDMazeDiffDrive):
    def __init__(self, size=10):
        super().__init__(size)

        self.state_space = (np.array([0, 0]), np.array([self.size, self.size]))
        self.action_space = (np.array([0, 0]), np.array([self.size, self.size]))
        self.robot_model = None

    @override
    def reset(self, random_env=True):
        if random_env:
            self._random_obstacles()
            self.start_state = self._random_valid_state()
            self.goal_state = self._random_valid_state()
        else:
            self.obstacles = DEFAULT_OBSTACLES
            self.start_state = DEFAULT_START[:2]
            self.goal_state = DEFAULT_GOAL[:2]

    @override
    def get_available_actions(self, state: Any) -> list[Any]:
        raise NotImplementedError(
            "TwoDMazeOmni does not support get_available_actions, " "use sample_available_actions instead!!"
        )

    def sample_available_actions(self, state: Any, num_to_sample=1) -> list[Any]:
        return np.random.sample(self.state_space[0], self.state_space[1], num_to_sample)

    @override
    def state_transition_func(self, state: Any, action: Any) -> tuple[Any, float, bool, bool, dict]:
        # new_state = self.robot_model.control_velocity(state, action[0], action[1], dt=1.0).tolist()
        new_state = action  # action is the new state

        if new_state[0] <= 0 or new_state[1] >= self.size or new_state[1] <= 0 or new_state[1] >= self.size:
            return state, 0, True, False, {"success": False}

        if not self.is_state_valid(new_state):
            return state, 0, True, False, {"success": False}

        return tuple(new_state), 1, False, False, {}

    def extend(self, state1, state2, step_size=0.1):
        s1 = np.array(state1)
        s2 = np.array(state2)

        if np.allclose(s1, s2):
            return [s1]

        num_of_steps = max(int((s2 - s1).max() / step_size) + 1, 2)
        step_size = (s2 - s1) / (num_of_steps - 1)

        path = [s1.tolist()]
        for i in range(num_of_steps):
            s = s1 + i * step_size
            if not self.is_state_valid(s):
                break
            else:
                path.append(s.tolist())

        return path

    def is_edge_valid(self, state1, state2, step_size=0.1):
        path = self.extend(state1, state2, step_size)
        return self.is_equal(path[-1], state2), np.linalg.norm(np.array(state2) - np.array(state1))

    def is_equal(self, state1, state2):
        return np.allclose(np.array(state1), np.array(state2))

    @override
    def add_path(self, path):
        interpolated_path = [self.start_state]

        state = self.start_state
        # Run simulation
        print(path)
        for action in path:
            interpolated_path += self.extend(state, action)
            state = action

        self.path = interpolated_path
        print(self.path)

    def add_state_samples(self, state):
        if self.state_samples is None:
            self.state_samples = []
        self.state_samples.append(state)
