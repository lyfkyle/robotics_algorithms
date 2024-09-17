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


class TwoDWorldDiffDrive(ContinuousEnv, DeterministicEnv, FullyObservableEnv):
    """A differential drive robot must reach goal state in a 2d maze with obstacles.

    State: [x, y, theta]
    Action: [lin_vel, ang_vel]

    Continuous state space.
    Continuous action space.
    Deterministic transition.
    Fully observable.

    There are two modes.
    If user does not set a reference path, the reward only encourages robot to reach goal as fast as possible. This
    environment hence behaves like a path planning environment.
    If user sets a reference path, the reward will encourage the robot to track the path to reach the goal. Hence, the
    environment behaves like a path following (control) environment.
    """

    FREE_SPACE = 0
    OBSTACLE = 1
    START = 2
    GOAL = 3
    PATH = 4
    WAYPOINT = 5
    MAX_POINT_TYPE = 6

    def __init__(self, size=10, robot_radius=0.2, action_dt=1.0, ref_path=None):
        super().__init__()

        self.size = size
        self.maze = np.full((size, size), TwoDWorldDiffDrive.FREE_SPACE)

        self.state_space = (np.array([0, 0, -math.pi]), np.array([self.size, self.size, math.pi]))
        self.action_space = (np.array([0, -math.radians(30)]), np.array([0.5, math.radians(30)]))  # lin_vel, ang_vel
        self.state_space_size = self.state_space[1] - self.state_space[0]
        self.action_space_size = self.action_space[1] - self.action_space[0]

        self.colour_map = colors.ListedColormap(["white", "black", "red", "blue", "green", "yellow"])
        bounds = [
            TwoDWorldDiffDrive.FREE_SPACE,
            TwoDWorldDiffDrive.OBSTACLE,
            TwoDWorldDiffDrive.START,
            TwoDWorldDiffDrive.GOAL,
            TwoDWorldDiffDrive.PATH,
            TwoDWorldDiffDrive.WAYPOINT,
            TwoDWorldDiffDrive.MAX_POINT_TYPE,
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
        self.robot_radius = robot_radius
        self.action_dt = action_dt

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
        self.ref_path = None
        self.cur_ref_path_idx = 0
        self.cur_state = self.start_state

    @override
    def step(self, action):
        res = super().step(action)

        # update current reference path index if reference path is present
        if self.ref_path is not None:
            self.cur_ref_path_idx = self._get_nearest_waypoint_to_state(res[0])

        return res

    @override
    def state_transition_func(self, state: list, action: list) -> tuple[Any, float, bool, bool, dict]:
        # compute next state
        action = np.clip(np.array(action), self.action_space[0], self.action_space[1]).tolist()
        new_state = self.robot_model.control_velocity(state, action[0], action[1], dt=self.action_dt)

        # Compute reward
        reward = self.reward_func(state, new_state=new_state)

        # Compute term and info
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

        return new_state, reward, term, False, info

    @override
    def reward_func(self, state, action=None, new_state=None):
        # If reference path does not exist, use manually distance traveled reward. Useful for computing shortest path.
        if self.ref_path is None:
            if new_state[0] <= 0 or new_state[0] >= self.size or new_state[1] <= 0 or new_state[1] >= self.size:
                return -100

            if not self.is_state_valid(new_state):
                return -100

            if self.is_state_similar(new_state, self.goal_state):
                return 0

            return -1

        # If reference path is present, the robot is supposed to track this path. Use distance to reference path
        # as reward.
        else:
            x, y, yaw = new_state
            # yaw = math_utils.normalize_angle(yaw)  # normalize theta to [0, 2*pi]

            if self.is_state_similar(new_state, self.goal_state):
                return 0

            # calculate stage cost
            nearest_idx = self._get_nearest_waypoint_to_state(new_state)
            ref_x = self.ref_path[nearest_idx, 0]
            ref_y = self.ref_path[nearest_idx, 1]
            ref_yaw = self.ref_path[nearest_idx, 2]
            yaw_diff = math_utils.normalize_angle(yaw - ref_yaw)
            # print(x, y, yaw, ref_x, ref_y, ref_yaw, yaw_diff)
            path_cost = (
                self.x_cost_weight * (x - ref_x) ** 2
                + self.y_cost_weight * (y - ref_y) ** 2
                + self.yaw_cost_weight * yaw_diff**2
            ).item()

            prev_idx = self._get_nearest_waypoint_to_state(state)
            progress_cost = 0.1 / (nearest_idx - prev_idx + 1)

            return -(path_cost + progress_cost)

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

    def add_ref_path(self, ref_path: list[list]):
        """Add a reference path for robot to follow.

        Args:
            ref_path (list(list)): the reference path.
        """
        self.ref_path = np.array(ref_path)
        self.cur_ref_path_idx = 0

    def add_action_path(self, action_path):
        """Add an action path for visualization.

        Args:
            action_path (list(list)): the path consisting a list of consecutive action.
        """
        interpolated_path = [self.start_state]

        state = self.start_state
        num_sub_steps = int(self.action_dt / self.robot_model.time_res)
        # Run simulation
        for action in action_path:
            for _ in range(num_sub_steps):
                state = self.robot_model.control_velocity(state, action[0], action[1], dt=self.robot_model.time_res)
                interpolated_path.append(state)

        self.path = interpolated_path

    def add_state_path(self, path):
        """Add a path for visualization.

        Args:
            path (list(list)): the path.
        """
        self.path = path

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
        # plt.plot(
        #     self.cur_state[0],
        #     self.cur_state[1],
        #     marker=(3, 0, math.degrees(self.cur_state[2])+30),
        #     markersize=(s * self.robot_radius * 2),
        #     c="blue",
        #     linestyle="None",
        # )
        if self.state_samples:
            for state in self.state_samples:
                plt.scatter(
                    state[0],
                    state[1],
                    # s=(s * self.robot_radius * 2) ** 2,
                    s=(s * 0.01 * 2) ** 2,
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
                    c="blue",
                    marker="o",
                )
        if self.ref_path is not None:
            for state in self.ref_path:
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

    def _get_nearest_waypoint_to_state(self, state: list):
        """search the closest waypoint to the vehicle on the reference path"""
        # x, y, _ = state
        # xy_state = np.array([x, y])
        # dists = np.linalg.norm((self.ref_path[:, :2] - xy_state), axis=-1).reshape(-1)
        # idx = np.argmin(dists).item()
        idx = self.cur_ref_path_idx
        SEARCH_IDX_LEN = 200  # [points] forward search range
        # prev_idx = self.prev_waypoints_idx
        new_x, new_y, _ = state
        new_xy_state = np.array([new_x, new_y])
        dists = np.linalg.norm((self.ref_path[idx : idx + SEARCH_IDX_LEN, :2] - new_xy_state), axis=-1).reshape(-1)
        nearest_idx = np.argmin(dists).item() + idx

        # dx = [x - ref_x for ref_x in self.ref_path[idx : idx + SEARCH_IDX_LEN, 0]]
        # dy = [y - ref_y for ref_y in self.ref_path[idx : idx + SEARCH_IDX_LEN, 1]]
        # d = [idx**2 + idy**2 for (idx, idy) in zip(dx, dy)]
        # min_d = min(d)
        # nearest_idx = d.index(min_d) + prev_idx

        # get reference values of the nearest waypoint

        # update nearest waypoint index if necessary
        # if update_prev_idx:
        #     self.prev_waypoints_idx = nearest_idx

        return nearest_idx


class TwoDWorldOmni(TwoDWorldDiffDrive):
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
