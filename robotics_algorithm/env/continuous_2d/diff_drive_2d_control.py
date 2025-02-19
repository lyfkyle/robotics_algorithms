import matplotlib.pyplot as plt
import numpy as np
from typing_extensions import override

from robotics_algorithm.env.base_env import (
    DeterministicEnv,
    FullyObservableEnv,
    FunctionType,
)
from robotics_algorithm.env.continuous_2d.diff_drive_2d_env_base import DiffDrive2DEnv
from robotics_algorithm.utils import math_utils

DEFAULT_OBSTACLES = [[2, 2, 0.5], [5, 5, 1], [3, 8, 0.5], [8, 3, 1]]

class DiffDrive2DControl(DiffDrive2DEnv, DeterministicEnv, FullyObservableEnv):
    """A differential drive robot must reach goal state in a 2d maze with obstacles.

    State: [x, y, theta]
    Action: [lin_vel, ang_vel]

    Continuous state space.
    Continuous action space.
    Deterministic transition.
    Fully observable.

    In this mode, user must set a reference path, the reward will encourage the robot to track the path to reach the
    goal.
    """

    def __init__(self, size=10, robot_radius=0.2, action_dt=0.05, use_lookahead=True, lookahead_dist=5, has_kinematics_constraint=True):
        """Constructor

        Args:
            size (int, optional): size of the environment. Defaults to 10.
            robot_radius (float, optional): robot radius. Defaults to 0.2.
            action_dt (float, optional): action dt. Defaults to 0.05.
            lookahead_dist (int, optional): lookahead distance. Defaults to -1.
        """
        # NOTE: MRO ensures DiffDrive2DEnv methods are always checked first. However, during init, we manually init
        #       DiffDrive2DEnv last.
        DeterministicEnv.__init__(self)
        FullyObservableEnv.__init__(self)
        DiffDrive2DEnv.__init__(self, size, robot_radius, action_dt, discrete_action=False, has_kinematics_constraint=has_kinematics_constraint)

        self.ref_path = None
        self.cur_ref_path_idx = 0
        self.use_lookahead = use_lookahead
        self.lookahead_index = lookahead_dist

        # declare linear state transition
        # ! The dynamics is not linear strictly but at each step can be approximated by
        # ! linearization.
        # x_new = Ax + Bu (discrete time model)
        self.state_transition_func_type = FunctionType.GENERAL.value

        # declare quadratic cost
        # L = x.T @ Q @ x + u.T @ R @ u
        self.x_cost_weight = 1
        self.y_cost_weight = 1
        self.yaw_cost_weight = 1.0
        self.reward_func_type = FunctionType.QUADRATIC.value
        self.Q = np.diag([self.x_cost_weight, self.y_cost_weight, self.yaw_cost_weight])
        self.R = np.diag([1, 1])  # control cost matrix

    def set_ref_path(self, ref_path: np.ndarray):
        """Add a reference path for robot to follow.

        Args:
            ref_path (np.ndarray): the reference path.
        """
        self.ref_path = ref_path
        self.cur_ref_path_idx = 0
        self.cur_lookahead_idx = self.lookahead_index

        self.start_state = ref_path[0]
        self.cur_state = ref_path[0]
        self.goal_state = ref_path[-1]
        self.goal_action = np.zeros(2)  # robot should stop at goal

    @override
    def reset(self, ref_path, empty=True, random_env=False):
        self.set_ref_path(np.array(ref_path))

        if random_env:
            self._random_obstacles()
        else:
            self.obstacles = DEFAULT_OBSTACLES

        # no obstacles if empty
        if empty:
            self.obstacles = []

        return self.sample_observation(self.cur_state), {}

    @override
    def step(self, action):
        res = super().step(action)

        # update current reference path index if reference path is present
        if self.ref_path is not None:
            self.cur_ref_path_idx = self.get_nearest_waypoint_to_state(res[0])
            self.cur_lookahead_idx = min(self.cur_ref_path_idx + self.lookahead_index, len(self.ref_path) - 1)

        return res

    def get_cur_lookahead_state(self):
        return self.ref_path[self.cur_lookahead_idx]

    @override
    def reward_func(self, state, action=None, new_state=None):
        x, y, yaw = new_state
        # yaw = math_utils.normalize_angle(yaw)  # normalize theta to [0, 2*pi]

        if self.is_state_similar(new_state, self.goal_state):
            return 0

        # calculate state cost
        nearest_idx = self.get_nearest_waypoint_to_state(new_state)
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

        # Encourage the robot to progress along the reference path
        prev_idx = self.get_nearest_waypoint_to_state(state)
        if nearest_idx <= prev_idx:
            progress_cost = 10
        else:
            progress_cost = 0
        #     progress_cost = 0.1 / min((nearest_idx - prev_idx))

        return -(path_cost + progress_cost)

    def get_nearest_waypoint_to_state(self, state: np.ndarray):
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

    @override
    def linearize_state_transition(self, state, action):
        return self.robot_model.linearize_state_transition(state, action)

    def set_local_plan(self, local_plan: np.ndarray[np.ndarray]):
        """Set current local plan of controller

        Args:
            local_plan (np.ndarray[np.ndarray]): np.ndarray of predicted future states
        """
        self.local_plan = local_plan

    @override
    def render(self, draw_start=True, draw_goal=True, draw_current=True):
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
        if draw_current:
            plt.scatter(
                self.cur_state[0],
                self.cur_state[1],
                s=(s * self.robot_radius * 2) ** 2,
                c='blue',
                marker='o',
                zorder=2.5
            )
            plt.arrow(
                self.cur_state[0],
                self.cur_state[1],
                np.cos(self.cur_state[2]) * self.robot_radius * 1.5,
                np.sin(self.cur_state[2]) * self.robot_radius * 1.5,
                head_width=0.1,
                head_length=0.2,
                fc='r',
                ec='r',
                zorder=2.5
            )
        if self.ref_path is not None:
            plt.plot(
                [s[0] for s in self.ref_path],
                [s[1] for s in self.ref_path],
                # s=(s * self.robot_radius * 2) ** 2,
                ms=(s * 0.01 * 2),
                c='green',
                marker='o',
            )
        if self.use_lookahead:
            cur_ref_state = self.get_cur_lookahead_state()
            plt.scatter(
                cur_ref_state[0],
                cur_ref_state[1],
                # s=(s * self.robot_radius * 2) ** 2,
                s=(s * 0.1 * 2) ** 2,
                c='orange',
                marker='o',
                zorder=2.5
            )
        if self.local_plan:
            for state in self.local_plan:
                plt.scatter(
                    state[0],
                    state[1],
                    # s=(s * self.robot_radius * 2) ** 2,
                    s=(s * 0.01 * 2) ** 2,
                    c='blue',
                    marker='o',
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