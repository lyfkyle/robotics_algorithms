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

    def __init__(self, size=10, robot_radius=0.2, action_dt=0.05, discrete_action=False):
        # NOTE: MRO ensures DiffDrive2DEnv methods are always checked first. However, during init, we manually init
        #       DiffDrive2DEnv last.
        DeterministicEnv.__init__(self)
        FullyObservableEnv.__init__(self)
        DiffDrive2DEnv.__init__(self, size, robot_radius, action_dt, discrete_action)

        self.ref_path = None
        self.cur_ref_path_idx = 0

        # Path following weights
        self.x_cost_weight = 50
        self.y_cost_weight = 50
        self.yaw_cost_weight = 1.0

    def set_ref_path(self, ref_path: list[list]):
        """Add a reference path for robot to follow.

        Args:
            ref_path (list(list)): the reference path.
        """
        self.ref_path = np.array(ref_path)
        self.cur_ref_path_idx = 0

    @override
    def reset(self, empty=False, random_env=True):
        self.ref_path = None
        self.cur_ref_path_idx = 0

        return super().reset(empty, random_env)

    @override
    def step(self, action):
        res = super().step(action)

        # update current reference path index if reference path is present
        if self.ref_path is not None:
            self.cur_ref_path_idx = self._get_nearest_waypoint_to_state(res[0])

        return res

    @override
    def reward_func(self, state, action=None, new_state=None):
        x, y, yaw = new_state
        # yaw = math_utils.normalize_angle(yaw)  # normalize theta to [0, 2*pi]

        if self.is_state_similar(new_state, self.goal_state):
            return 0

        # calculate state cost
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

    def linearize_state_transition(self, state):
        return self.robot_model.linearize_dynamics(state)

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

    def set_local_plan(self, local_plan: list[list]):
        """Set current local plan of controller

        Args:
            local_plan (list[list]): list of predicted future states
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

class DiffDrive2DControlRelative(DiffDrive2DControl):
    """A differential drive robot must reach goal state in a 2d maze with obstacles.

    State: [x, y, theta]
    Action: [lin_vel, ang_vel]

    Continuous state space.
    Continuous action space.
    Deterministic transition.
    Fully observable.

    In this mode, user must set a reference path, the reward will encourage the robot to track the path to reach the
    goal. The difference between this class and DiffDrive2DControl is that state is defined with relative to the
    reference state. This way allows us to liearize the state transition
    """

    def __init__(self, size=10, robot_radius=0.2, action_dt=0.05, discrete_action=False):
        # NOTE: MRO ensures DiffDrive2DEnv methods are always checked first. However, during init, we manually init
        #       DiffDrive2DEnv last.
        DeterministicEnv.__init__(self)
        FullyObservableEnv.__init__(self)
        DiffDrive2DEnv.__init__(self, size, robot_radius, action_dt, discrete_action)

        self.ref_path = None
        self.cur_ref_path_idx = 0

        # declare linear state transition
        # ! The dynamics is not linear strictly but at each step can be approximated by
        # ! linearization.
        # x_new = Ax + Bu (discrete time model)
        self.state_transition_func_type = FunctionType.LINEAR.value

        # declare quadratic cost
        # L = x.T @ Q @ x + u.T @ R @ u
        self.x_cost_weight = 50
        self.y_cost_weight = 50
        self.yaw_cost_weight = 1.0
        self.reward_func_type = FunctionType.QUADRATIC.value
        self.Q = np.diag([self.x_cost_weight, self.y_cost_weight, self.yaw_cost_weight])
        self.R = np.diag([1, 1])  # control cost matrix

    @override
    def reset(self, ref_path):
        self.ref_path = ref_path
        state, info = super().reset()

        ref_state = self.ref_path[0]
        state_relative = np.array(state) - np.array(ref_state)
        return state_relative.tolist()

