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

        self.start_state = ref_path[0]
        self.cur_state = ref_path[0]
        self.goal_state = ref_path[-1]

    @override
    def reset(self, ref_path, empty=False, random_env=False):
        self.set_ref_path(ref_path)

        return super().reset(empty, random_env)

    @override
    def step(self, action):
        res = super().step(action)

        # update current reference path index if reference path is present
        if self.ref_path is not None:
            self.cur_ref_path_idx = self.get_nearest_waypoint_to_state(res[0])

        return res

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

        prev_idx = self.get_nearest_waypoint_to_state(state)
        progress_cost = 0.1 / (nearest_idx - prev_idx + 1)

        return -(path_cost + progress_cost)

    def get_nearest_waypoint_to_state(self, state: list):
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


class DiffDrive2DControlRelative(DeterministicEnv, FullyObservableEnv):
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

        # TODO refactor this...
        self._env_impl = DiffDrive2DControl(size, robot_radius, action_dt, discrete_action)
        self._env_impl.interactive_viz = True

        self.state_space = self._env_impl.state_space
        self.action_space = self._env_impl.action_space

        self.ref_path = None
        self.lookahead_index = 5
        self.cur_ref_path_idx = 0

        # declare linear state transition
        # ! The dynamics is not linear strictly but at each step can be approximated by
        # ! linearization.
        # x_new = Ax + Bu (discrete time model)
        self.state_transition_func_type = FunctionType.LINEAR.value

        # declare quadratic cost
        # L = x.T @ Q @ x + u.T @ R @ u
        self.x_cost_weight = 1
        self.y_cost_weight = 1
        self.yaw_cost_weight = 1.0
        self.reward_func_type = FunctionType.QUADRATIC.value
        self.Q = np.diag([self.x_cost_weight, self.y_cost_weight, self.yaw_cost_weight])
        self.R = np.diag([1, 1])  # control cost matrix

    @override
    def reset(self, ref_path, empty=False, random_env=False):
        self.ref_path = ref_path
        self.cur_ref_path_idx = min(self.lookahead_index, len(self.ref_path) - 1)

        self._state_impl, info = self._env_impl.reset(ref_path, empty, random_env)

        self._cur_ref_state = np.array(ref_path[self.cur_ref_path_idx])
        # self._cur_ref_state = [3.0, 3.0, 0.0]
        state_relative = (np.array(self._state_impl) - self._cur_ref_state).tolist()

        self.start_state = state_relative
        self.goal_state = self.ref_path[-1]
        self.cur_state = state_relative
        return state_relative, info

    @override
    def state_transition_func(self, state: list, action: list) -> list:
        # compute next state
        state_impl = self._cur_ref_state + np.array(state)
        new_state_impl = self._env_impl.state_transition_func(state_impl, action)
        new_state = np.array(new_state_impl) - self._cur_ref_state

        return new_state.tolist()

    @override
    def get_state_info(self, state):
        return self._env_impl.get_state_info(state)

    @override
    def step(self, action):
        new_state, reward, term, trunc, info = self._env_impl.step(action)

        # self._state_impl = np.array(new_state) + self._cur_ref_state
        self._state_impl = new_state

        # update current reference path index if reference path is present
        nearest_wp_index = self._env_impl.get_nearest_waypoint_to_state(self._state_impl)
        if np.linalg.norm(np.array(self._state_impl) - np.array(self._cur_ref_state)) < 0.2:
            # if nearest_wp_index > self.cur_ref_path_idx - 2:
            self.cur_ref_path_idx = min(nearest_wp_index + self.lookahead_index, len(self.ref_path) - 1)
            self._cur_ref_state = self.ref_path[self.cur_ref_path_idx]

        self.cur_state = (np.array(self._state_impl) - self._cur_ref_state).tolist()

        return self.cur_state, reward, term, trunc, info

    @override
    def linearize_state_transition(self, state):
        state_impl = np.array(state) + self._cur_ref_state
        return self._env_impl.robot_model.linearize_dynamics(state_impl)

    @override
    def render(self, draw_start=True, draw_goal=True, draw_current=True):
        if self._env_impl.interactive_viz:
            if not self._env_impl._fig_created:
                plt.ion()
                plt.figure(figsize=(10, 10), dpi=100)
                self._env_impl._fig_created = True
        else:
            plt.figure(figsize=(10, 10), dpi=100)

        plt.clf()
        s = 1000 / self._env_impl.size / 2
        for obstacle in self._env_impl.obstacles:
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
                self._env_impl.goal_state[0],
                self._env_impl.goal_state[1],
                s=(s * self._env_impl.robot_radius * 2) ** 2,
                c='red',
                marker='o',
            )
        if draw_start:
            plt.scatter(
                self._env_impl.start_state[0],
                self._env_impl.start_state[1],
                s=(s * self._env_impl.robot_radius * 2) ** 2,
                c='yellow',
                marker='o',
            )
        if self.ref_path is not None:
            plt.plot(
                [s[0] for s in self._env_impl.ref_path],
                [s[1] for s in self._env_impl.ref_path],
                # s=(s * self.robot_radius * 2) ** 2,
                ms=(s * 0.01 * 2),
                c='green',
                marker='o',
            )
        plt.scatter(
            self._cur_ref_state[0],
            self._cur_ref_state[1],
            # s=(s * self.robot_radius * 2) ** 2,
            s=(s * 0.1 * 2) ** 2,
            c='orange',
            marker='o',
            zorder=2.5
        )
        if draw_current:
            plt.scatter(
                self._env_impl.cur_state[0],
                self._env_impl.cur_state[1],
                s=(s * self._env_impl.robot_radius * 2) ** 2,
                c='blue',
                marker='o',
                zorder=2.5
            )
            plt.arrow(
                self._env_impl.cur_state[0],
                self._env_impl.cur_state[1],
                np.cos(self._env_impl.cur_state[2]) * self._env_impl.robot_radius * 1.5,
                np.sin(self._env_impl.cur_state[2]) * self._env_impl.robot_radius * 1.5,
                head_width=0.1,
                head_length=0.2,
                fc='r',
                ec='r',
                zorder=2.5
            )
        plt.xlim(0, self._env_impl.size)
        plt.xticks(np.arange(self._env_impl.size))
        plt.ylim(0, self._env_impl.size)
        plt.yticks(np.arange(self._env_impl.size))
        plt.legend()

        if self._env_impl.interactive_viz:
            plt.pause(0.01)
        else:
            plt.show()
