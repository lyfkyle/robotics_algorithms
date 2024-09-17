import random
from collections import defaultdict
from typing import Any
from typing_extensions import override
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from robotics_algorithm.env.base_env import DeterministicEnv, ContinuousEnv, FullyObservableEnv
from robotics_algorithm.robot.differential_drive import DiffDrive
from robotics_algorithm.utils import math_utils


DEFAULT_OBSTACLES = [[2, 2, 0.5], [5, 5, 1], [3, 8, 0.5], [8, 3, 1]]
DEFAULT_START = [0.5, 0.5, 0]
DEFAULT_GOAL = [9.0, 9.0, math.radians(90)]


class TwoDMazeDiffDrive(ContinuousEnv, DeterministicEnv, FullyObservableEnv):
    """A differential drive robot must reach goal state in a 2d maze with obstacles.

    State: [x, y, theta]
    Action: [lin_vel, ang_vel]

    Continuous state space.
    Continuous action space.
    Deterministic transition.
    Fully observable.
    """

    FREE_SPACE = 0
    OBSTACLE = 1
    START = 2
    GOAL = 3
    PATH = 4
    WAYPOINT = 5
    MAX_POINT_TYPE = 6

    def __init__(self, size=10, ref_path=None):
        super().__init__()

        self.size = size
        self.maze = np.full((size, size), TwoDMazeDiffDrive.FREE_SPACE)

        self.state_space = (np.array([0, 0, -math.pi]), np.array([self.size, self.size, math.pi]))
        self.action_space = (np.array([0, 0]), np.array([self.size, self.size]))
        self.state_space_size = self.state_space[1] - self.state_space[0]
        self.action_space_size = self.action_space[1] - self.action_space[0]

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
        self.action_dt = 1.0

        self.path = None
        self.state_samples = None

        # Path following weights
        self.ref_path = ref_path
        self.x_cost_weight = 50
        self.y_cost_weight = 50
        self.yaw_cost_weight = 1.0

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

    def set_ref_path(self, ref_path):
        self.ref_path = ref_path

    @override
    def state_transition_func(self, state: Any, action: Any) -> tuple[Any, float, bool, bool, dict]:
        new_state = self.robot_model.control_velocity(state, action[0], action[1], dt=self.action_dt)

        term = False
        info = {}
        if new_state[0] <= 0 or new_state[0] >= self.size or new_state[1] <= 0 or new_state[1] >= self.size:
            term = True
            info = {"success": False}

        if not self.is_state_valid(new_state):
            term = True
            info = {"success": False}

        # Check goal state reached for termination
        if self.is_state_similar(new_state, self.goal_state):
            term = True
            info = {"success": True}

        reward = self.reward_func(state, new_state=new_state)

        return new_state, reward, term, False, info

    @override
    def reward_func(self, state, action=None, new_state=None):
        if self.ref_path is None:
            if new_state[0] <= 0 or new_state[0] >= self.size or new_state[1] <= 0 or new_state[1] >= self.size:
                return -100

            if not self.is_state_valid(new_state):
                return -100

            if self.is_state_similar(new_state, self.goal_state):
                return 0

            return -1

        else:
            x, y, yaw = new_state
            yaw = math_utils.normalize_angle(yaw)  # normalize theta to [0, 2*pi]

            # calculate stage cost
            ref_x, ref_y, ref_yaw = self._get_nearest_waypoint(x, y)
            yaw_diff = math_utils.normalize_angle(yaw - ref_yaw)
            path_cost = (
                self.x_cost_weight * (x - ref_x) ** 2
                + self.y_cost_weight * (y - ref_y) ** 2
                + self.yaw_cost_weight * yaw_diff ** 2
            )

            return -path_cost

    @override
    def get_available_actions(self, state: Any) -> list[Any]:
        return self._sample_actions

    @override
    def is_state_valid(self, state):
        for obstacle in self.obstacles:
            if np.linalg.norm(np.array(state[:2]) - np.array(obstacle[:2])) <= obstacle[2] + self.robot_radius:
                return False

        return True

    def is_state_similar(self, state1, state2):
        return self.calc_state_key(state1) == self.calc_state_key(state2)

    def calc_state_key(self, state):
        return (round(state[0] / 0.25), round(state[1] / 0.25), round((state[2] + math.pi) / math.radians(30)))

    def _random_obstacles(self, num_of_obstacles=5):
        self.obstacles = []
        for _ in range(num_of_obstacles):
            obstacle = np.random.uniform([0, 0, 0.1], [self.size, self.size, 1])
            self.obstacles.append(obstacle.tolist())

    def _random_valid_state(self):
        while True:
            robot_pos = np.random.uniform(self.state_space[0], self.state_space[1])
            if self.is_state_valid(robot_pos):
                break

        return robot_pos.tolist()

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

    def _get_nearest_waypoint(self, x: float, y: float, update_prev_idx: bool = False):
        """search the closest waypoint to the vehicle on the reference path"""

        SEARCH_IDX_LEN = 200 # [points] forward search range
        prev_idx = self.prev_waypoints_idx
        dx = [x - ref_x for ref_x in self.ref_path[prev_idx:(prev_idx + SEARCH_IDX_LEN), 0]]
        dy = [y - ref_y for ref_y in self.ref_path[prev_idx:(prev_idx + SEARCH_IDX_LEN), 1]]
        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
        min_d = min(d)
        nearest_idx = d.index(min_d) + prev_idx

        # get reference values of the nearest waypoint
        ref_x = self.ref_path[nearest_idx,0]
        ref_y = self.ref_path[nearest_idx,1]
        ref_yaw = self.ref_path[nearest_idx,2]

        # update nearest waypoint index if necessary
        if update_prev_idx:
            self.prev_waypoints_idx = nearest_idx

        return ref_x, ref_y, ref_yaw


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
    def sample_available_actions(self, state: Any, num_samples=1) -> list[Any]:
        return np.random.sample(self.state_space[0], self.state_space[1], num_samples)

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
    def add_path(self, path):
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
