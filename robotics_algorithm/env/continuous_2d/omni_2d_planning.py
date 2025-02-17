import matplotlib.pyplot as plt
import numpy as np
from typing_extensions import override

from robotics_algorithm.env.base_env import (
    ContinuousSpace,
    DeterministicEnv,
    FullyObservableEnv,
)

DEFAULT_OBSTACLES = [[2, 2, 0.5], [5, 5, 1], [3, 8, 0.5], [8, 3, 1]]
DEFAULT_START = [0.5, 0.5]
DEFAULT_GOAL = [9.0, 9.0]

class OmniDrive2DPlanning(DeterministicEnv, FullyObservableEnv):
    def __init__(self, size=10, robot_radius=0.2):
        super().__init__()

        self.size = size
        self.robot_radius = robot_radius

        self.state_space = ContinuousSpace(low=np.array([0, 0]), high=np.array([self.size, self.size]))
        self.action_space = ContinuousSpace(low=np.array([0, 0]), high=np.array([self.size, self.size]))

        self.robot_model = None
        self.state_samples = []
        self.path = []
        self.path_dict = {}

    @override
    def reset(self, random_env=True):
        self.state_samples = []
        self.path = []
        self.path_dict = {}

        if random_env:
            self._random_obstacles()
            self.start_state = self._random_valid_state()
            self.goal_state = self._random_valid_state()
        else:
            self.obstacles = DEFAULT_OBSTACLES
            self.start_state = np.array(DEFAULT_START)
            self.goal_state = np.array(DEFAULT_GOAL)

        self.cur_state = self.start_state.copy()
        return self.cur_state, {}

    @override
    def state_transition_func(self, state: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        new_state = action  # action is the new state
        return new_state

    def extend(self, s1: np.ndarray, s2: np.ndarray, step_size=0.1):
        if np.allclose(s1, s2):
            return [s1]

        num_of_steps = max(int(np.abs(s2 - s1).max() / step_size) + 1, 2)
        step_size = (s2 - s1) / (num_of_steps - 1)

        path = []
        for i in range(num_of_steps):
            s = s1 + i * step_size
            if not self.is_state_valid(s):
                break
            else:
                path.append(s)

        return path

    def is_state_transition_valid(self, state1: np.ndarray, action: np.ndarray, state2: np.ndarray, step_size=0.1):
        path = self.extend(state1, state2, step_size)
        res = np.allclose(path[-1], state2)
        return res, np.linalg.norm(np.array(state2) - np.array(state1))

    @override
    def add_state_path(self, path, id=None, interpolate=False):
        if interpolate:
            interpolated_path = []

            state = self.start_state
            # Run simulation
            for new_state in path[1:]:
                interpolated_path += self.extend(np.array(state), np.array(new_state))
                assert np.allclose(interpolated_path[-1], new_state)
                state = new_state

            path = interpolated_path

        if id is None:
            self.path = path
        else:
            self.path_dict[id] = path

    def add_state_samples(self, state):
        if self.state_samples is None:
            self.state_samples = []
        self.state_samples.append(state)

    def _random_obstacles(self, num_of_obstacles: int = 5):
        self.obstacles = []
        for _ in range(num_of_obstacles):
            obstacle = np.random.uniform([0, 0, 0.1], [self.size, self.size, 1])
            self.obstacles.append(obstacle.tolist())

    def _random_valid_state(self):
        while True:
            robot_pos = np.random.uniform(self.state_space.space[0], self.state_space.space[1])
            if self.is_state_valid(robot_pos):
                break

        return robot_pos

    @override
    def is_state_valid(self, state: np.ndarray) -> bool:
        for obstacle in self.obstacles:
            if np.linalg.norm(state[:2] - np.array(obstacle[:2])) <= obstacle[2] + self.robot_radius:
                return False

        return True

    @override
    def render(self, draw_start=True, draw_goal=True):
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
        if self.state_samples:
            for state in self.state_samples:
                plt.scatter(
                    state[0],
                    state[1],
                    # s=(s * self.robot_radius * 2) ** 2,
                    s=(s * 0.01 * 2) ** 2,
                    c='blue',
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

        plt.show()

