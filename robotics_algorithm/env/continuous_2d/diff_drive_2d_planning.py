import math

import matplotlib.pyplot as plt
import numpy as np
from typing_extensions import override

from robotics_algorithm.env.base_env import (
    DeterministicEnv,
    FullyObservableEnv,
)
from robotics_algorithm.env.continuous_2d.diff_drive_2d_env_base import DiffDrive2DEnv

DEFAULT_OBSTACLES = [[2, 2, 0.5], [5, 5, 1], [3, 8, 0.5], [8, 3, 1]]
DEFAULT_START = [0.5, 0.5, 0]
DEFAULT_GOAL = [9.0, 9.0, math.radians(90)]


class DiffDrive2DPlanning(DiffDrive2DEnv, DeterministicEnv, FullyObservableEnv):
    """A differential drive robot must reach goal state in a 2d maze with obstacles.

    State: [x, y, theta]
    Action: [lin_vel, ang_vel]

    Continuous state space.
    Continuous action space.
    Deterministic transition.
    Fully observable.

    In this mode, user provides start and goal. The reward encourages robot to reach goal via the shortest path.
    """

    def __init__(self, size=10, robot_radius=0.2, action_dt=1.0, ref_path=None, discrete_action=False):
        # NOTE: MRO ensures DiffDrive2DEnv methods are always checked first. However, during init, we manually init
        #       DiffDrive2DEnv last.
        DeterministicEnv.__init__(self)
        FullyObservableEnv.__init__(self)
        DiffDrive2DEnv.__init__(self, size, robot_radius, action_dt, ref_path, discrete_action)

        self.state_samples = []

    @override
    def reset(self, empty=False, random_env=True):
        self.state_samples = []

        return super().reset(empty, random_env)

    @override
    def reward_func(self, state, action=None, new_state=None):
        if new_state[0] <= 0 or new_state[0] >= self.size or new_state[1] <= 0 or new_state[1] >= self.size:
            return -100

        if not self.is_state_valid(new_state):
            return -100

        if self.is_state_similar(new_state, self.goal_state):
            return 0

        return -1

    def add_action_path(self, action_path):
        """Add an action path for visualization.

        Args:
            action_path (list(list)): the path consisting a list of consecutive action.
        """
        interpolated_path = [self.start_state]

        state = self.start_state
        num_sub_steps = round(self.action_dt / self.robot_model.time_res)

        # Run simulation
        for action in action_path:
            for _ in range(num_sub_steps):
                state = self.robot_model.control_velocity(state, action[0], action[1], dt=self.robot_model.time_res)
                interpolated_path.append(state)

        self.path = interpolated_path

    def add_state_path(self, path, id=None):
        """Add a path for visualization.

        Args:
            path (list(list)): the path.
        """
        if id is None:
            self.path = path
        else:
            self.path_dict[id] = path

    def render(self, draw_start=True, draw_goal=True):
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

        if self.interactive_viz:
            plt.pause(0.01)
        else:
            plt.show()


class DiffDrive2DPlanningWithCost(DiffDrive2DPlanning):
    def __init__(self, size=10, robot_radius=0.2, action_dt=1.0, ref_path=None, discrete_action=False):
        # NOTE: MRO ensures DiffDrive2DEnv methods are always checked first. However, during init, we manually init
        #       DiffDrive2DEnv last.
        super().__init__(size, robot_radius, action_dt, ref_path, discrete_action)

        self._cost_map = {}
        self._costmap_res = 0.1
        self._exp_decay = 0.5
        self.cost_penalty = 1.0
        self.max_cost = 255

    @override
    def reset(self):
        self.obstacles = [[4, 6, 1], [6, 4, 1]]
        self.start_state = DEFAULT_START
        self.goal_state = DEFAULT_GOAL

        self.cur_state = self.start_state.copy()

        self._cost_map = {}
        self.precompute_cost()

        return self.sample_observation(self.cur_state), {}

    def precompute_cost(self):
        costmap_size = int(self.size / self._costmap_res)
        for i in range(costmap_size):
            for j in range(costmap_size):
                # key = (math.floor(x / self._costmap_res), math.floor(y / self._costmap_res))
                self._cost_map[(i, j)] = self._calc_cost(
                    (i * self._costmap_res + self._costmap_res / 2, j * self._costmap_res + self._costmap_res / 2)
                )

    def _calc_cost(self, state_xy):
        """
        Calculate the cost for a given state based on the distance to obstacles.

        Args:
            state_xy (tuple): The current xy state as a tuple (x, y).

        Returns:
            float: The computed cost based on the proximity to obstacles.
        """
        # Convert obstacles to numpy array for vectorized operations
        obstacles_np = np.array(self.obstacles)

        # Calculate Euclidean distance from the state to each obstacle
        dist = np.linalg.norm(np.array(state_xy) - obstacles_np[:, :2], axis=-1)

        # Adjust distances by subtracting the robot's and obstacles' radii
        dist = dist - self.robot_radius - obstacles_np[:, 2]

        # Find the minimum distance among all obstacles
        dist = dist.min()

        # Calculate cost using exponential decay based on distance
        cost = self.max_cost * math.exp(-self._exp_decay * dist)

        return cost

    def get_cost(self, state):
        # query the precomputed cost map
        key = (math.floor(state[0] / self._costmap_res), math.floor(state[1] / self._costmap_res))
        return self._cost_map[key]

    @override
    def reward_func(self, state, action=None, new_state=None):
        # cost-weighted distance travelled reward.
        if new_state[0] <= 0 or new_state[0] >= self.size or new_state[1] <= 0 or new_state[1] >= self.size:
            return -100

        if not self.is_state_valid(new_state):
            return -100

        if self.is_state_similar(new_state, self.goal_state):
            return 0

        return -1 * (1.0 + self.cost_penalty * self.get_cost(new_state) / self.max_cost)  # cost-weighted distance.

    def render(self, draw_start=True, draw_goal=True, title="environment"):
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
        # plt.plot(
        #     self.cur_state[0],
        #     self.cur_state[1],
        #     marker=(3, 0, math.degrees(self.cur_state[2])+30),
        #     markersize=(s * self.robot_radius * 2),
        #     c="blue",
        #     linestyle="None",
        # )
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

        # Plot cost from cost map
        if self._cost_map is not None:
            costmap_img = np.array(list(self._cost_map.values())).reshape(100, 100)
            plt.imshow(costmap_img, cmap='hot', alpha=0.5, origin='lower', extent=[0, 10, 0, 10])

        plt.xlim(0, self.size)
        plt.xticks(np.arange(self.size))
        plt.ylim(0, self.size)
        plt.yticks(np.arange(self.size))
        plt.title(title)
        plt.legend()

        if self.interactive_viz:
            plt.pause(0.01)
        else:
            plt.show()
