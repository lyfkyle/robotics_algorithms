from typing import Any
from typing_extensions import override
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats import multivariate_normal

from robotics_algorithm.env.base_env import (
    DeterministicEnv,
    FullyObservableEnv,
    ContinuousSpace,

)
from robotics_algorithm.env.continuous_2d.diff_drive_2d_planning import DiffDrive2DPlanning

DEFAULT_OBSTACLES = [[2, 2, 0.5], [5, 5, 1], [3, 8, 0.5], [8, 3, 1]]
DEFAULT_START = [0.5, 0.5]
DEFAULT_GOAL = [9.0, 9.0]

class OmniDrive2DPlanning(DiffDrive2DPlanning, DeterministicEnv, FullyObservableEnv):
    def __init__(self, size=10):
        super().__init__(size)

        self.state_space = ContinuousSpace(low=np.array([0, 0]), high=np.array([self.size, self.size]))
        self.action_space = ContinuousSpace(low=np.array([0, 0]), high=np.array([self.size, self.size]))
        self.robot_model = None

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

        self.cur_state = self.start_state.copy()
        return self.cur_state, {}

    @override
    def state_transition_func(self, state: Any, action: Any) -> tuple[Any, float, bool, bool, dict]:
        new_state = action  # action is the new state
        return new_state

    def extend(self, state1, state2, step_size=0.1):
        s1 = np.array(state1)
        s2 = np.array(state2)

        if np.allclose(s1, s2):
            return [s1.tolist()]

        num_of_steps = max(int(np.abs(s2 - s1).max() / step_size) + 1, 2)
        step_size = (s2 - s1) / (num_of_steps - 1)

        path = []
        for i in range(num_of_steps):
            s = s1 + i * step_size
            if not self.is_state_valid(s):
                break
            else:
                path.append(s.tolist())

        return path

    def is_state_transition_valid(self, state1, state2, step_size=0.1):
        path = self.extend(state1, state2, step_size)
        res = self.is_equal(path[-1], state2)
        return res, np.linalg.norm(np.array(state2) - np.array(state1))

    def is_equal(self, state1, state2):
        return np.allclose(np.array(state1), np.array(state2))

    @override
    def add_action_path(self, path):
        interpolated_path = []

        state = self.start_state
        # Run simulation
        for action in path[1:]:
            interpolated_path += self.extend(state, action)
            assert tuple(interpolated_path[-1]) == action
            state = action

        self.path = interpolated_path
        # print("[TwoDMaze]: Before interpolation", path)
        # print("[TwoDMaze]: After interpolation", interpolated_path)

    def add_state_samples(self, state):
        if self.state_samples is None:
            self.state_samples = []
        self.state_samples.append(state)
