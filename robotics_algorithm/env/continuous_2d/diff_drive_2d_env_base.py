import math
from typing import Any

import numpy as np
from matplotlib import colors
from typing_extensions import override

from robotics_algorithm.env.base_env import (
    BaseEnv,
    ContinuousSpace,
    DiscreteSpace,
)
from robotics_algorithm.robot.differential_drive import DiffDrive

DEFAULT_OBSTACLES = [[2, 2, 0.5], [5, 5, 1], [3, 8, 0.5], [8, 3, 1]]
DEFAULT_START = [0.5, 0.5, 0]
DEFAULT_GOAL = [9.0, 9.0, math.radians(90)]


class DiffDrive2DEnv(BaseEnv):
    """A differential drive robot must reach goal state in a 2d maze with obstacles.

    State: [x, y, theta]
    Action: [lin_vel, ang_vel]

    There are two modes.
    - If user does not set a reference path, the reward only encourages robot to reach goal as fast as possible. This
    environment hence behaves like a path planning environment.
    - If user sets a reference path, the reward will encourage the robot to track the path to reach the goal. Hence, the
    environment behaves like a path following (control) environment.
    """

    FREE_SPACE = 0
    OBSTACLE = 1
    START = 2
    GOAL = 3
    PATH = 4
    WAYPOINT = 5
    MAX_POINT_TYPE = 6

    def __init__(self, size=10, robot_radius=0.2, action_dt=1.0, discrete_action=False, has_kinematics_constraint=True):
        """
        Initialize a differential drive robot environment.

        Args:
            size (int): size of the maze
            robot_radius (float): radius of the robot
            action_dt (float): time step for the robot actions.
            discrete_action (bool): whether the action space is discrete or continuous
        """
        super().__init__()

        self.size = size
        self.maze = np.full((size, size), DiffDrive2DEnv.FREE_SPACE)

        self.state_space = ContinuousSpace(low=[0, 0, -math.pi], high=[self.size, self.size, math.pi])
        if not discrete_action:
            if has_kinematics_constraint:
                self.action_space = ContinuousSpace(low=[0, -math.radians(30)], high=[0.5, math.radians(30)])
            else:
                self.action_space = ContinuousSpace(low=[-float('inf'), -float('inf')], high=[float('inf'), float('inf')])
        else:
            self.action_space = DiscreteSpace(
                [
                    (0.5, 0),
                    (0.5, math.radians(30)),
                    (0.5, -math.radians(30)),
                    (0.25, 0),
                    (0.25, math.radians(30)),
                    (0.25, -math.radians(30)),
                ]
            )

        # self.colour_map = colors.ListedColormap(['white', 'black', 'red', 'blue', 'green', 'yellow'])
        # bounds = [
        #     DiffDrive2DEnv.FREE_SPACE,
        #     DiffDrive2DEnv.OBSTACLE,
        #     DiffDrive2DEnv.START,
        #     DiffDrive2DEnv.GOAL,
        #     DiffDrive2DEnv.PATH,
        #     DiffDrive2DEnv.WAYPOINT,
        #     DiffDrive2DEnv.MAX_POINT_TYPE,
        # ]
        # self.norm = colors.BoundaryNorm(bounds, self.colour_map.N)

        self.robot_model = DiffDrive(wheel_dist=0.2, wheel_radius=0.05)
        self.robot_radius = robot_radius
        self.action_dt = action_dt

        self.path = None
        self.path_dict = {}
        self.local_plan = None

        # others
        self.interactive_viz = False  # Interactive viz
        self._fig_created = False

    @override
    def reset(self, empty=False, random_env=True):
        if random_env:
            self._random_obstacles()
            self.start_state = self._random_valid_state()
            self.goal_state = self._random_valid_state()
        else:
            self.obstacles = DEFAULT_OBSTACLES
            self.start_state = DEFAULT_START
            self.goal_state = DEFAULT_GOAL

        # no obstacles if empty
        if empty:
            self.obstacles = []

        self.cur_state = self.start_state.copy()

        return self.sample_observation(self.cur_state), {}

    @override
    def state_transition_func(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        # compute next state
        new_state = self.robot_model.control(state, action, dt=self.action_dt)

        return new_state

    @override
    def get_state_info(self, state: np.ndarray) -> tuple[bool, bool, dict]:
        # Compute term and info
        term = False
        info = {}
        if state[0] <= 0 or state[0] >= self.size or state[1] <= 0 or state[1] >= self.size:
            term = True
            info = {'success': False}

        if not self.is_state_valid(state):
            term = True
            info = {'success': False}

        # Check goal state reached for termination
        if self.is_state_similar(state, self.goal_state):
            term = True
            info = {'success': True}

        return term, False, info

    @override
    def is_state_valid(self, state: np.ndarray) -> bool:
        for obstacle in self.obstacles:
            if np.linalg.norm(np.array(state[:2]) - np.array(obstacle[:2])) <= obstacle[2] + self.robot_radius:
                return False

        return True

    def is_state_similar(self, state1: np.ndarray, state2: np.ndarray) -> bool:
        # return self.calc_state_key(state1) == self.calc_state_key(state2)
        return np.linalg.norm(state1 - state2) < 0.2

    def calc_state_key(self, state: np.ndarray) -> tuple[int, int, int]:
        return (round(state[0] / 0.1), round(state[1] / 0.1), round((state[2] + math.pi) / math.radians(30)))

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
