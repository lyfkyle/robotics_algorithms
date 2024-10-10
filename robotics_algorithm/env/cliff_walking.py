import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from typing_extensions import override

from robotics_algorithm.env.base_env import MDPEnv, DiscreteSpace, DistributionType

GRID_HEIGHT = 4
GRID_WIDTH = 9
OBSTACLES = [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0)]


class CliffWalking(MDPEnv):
    """
    A player is placed in grid world. The player should move from start to goal. If the player reaches the goal the
    episode ends. If the player moves to a cliff location the episode terminates with failure.
    During each move, the player has a chance of ending up in the left or right of the target grid.

    State: [x, y]
    Action: [up, down, left, right]

    Discrete state space.
    Discrete action space.
    Stochastic transition.
    Fully observable.
    """

    def __init__(
        self,
        start: tuple = (0, 0),
        goal: tuple = (8, 0),
        obstacles: list[tuple] = OBSTACLES,
        dense_reward: bool = False,
    ):
        """Constructor.

        Args:
            start (tuple): the start position of agent.
            goal (tuple): the goal position.
            obstacles (list[tuple]): a list of obstacle positions.
            dense_reward (bool): whether to use dense reward for this env.
        """
        super().__init__()

        # Define spaces
        self.state_space = DiscreteSpace([(i, j) for i in range(GRID_WIDTH) for j in range(GRID_HEIGHT)])
        self.action_space = DiscreteSpace([0, 1, 2, 3])  # action, 0: up, 1: right, 2: down, 3: left)
        self.state_transition_dist_type = DistributionType.CATEGORICAL.value

        self.start_state = start
        self.cur_state = start
        self.goal_state = goal
        self.obstacles = obstacles
        self.step_reward = -1
        self.obstacle_reward = -100
        self.goal_reward = 100
        self.path = []
        self.dense_reward = dense_reward

    @override
    def state_transition_func(self, state: tuple, action: int) -> tuple[list[tuple], list[float]]:
        i, j = state

        new_states = []
        if action == 0:
            new_states.append((i, min(j + 1, GRID_HEIGHT - 1)))
            new_states.append((max(0, i - 1), min(j + 1, GRID_HEIGHT - 1)))
            new_states.append((min(i + 1, GRID_WIDTH - 1), min(j + 1, GRID_HEIGHT - 1)))
        elif action == 1:
            new_states.append((min(i + 1, GRID_WIDTH - 1), j))
            new_states.append((min(i + 1, GRID_WIDTH - 1), min(j + 1, GRID_HEIGHT - 1)))
            new_states.append((min(i + 1, GRID_WIDTH - 1), max(0, j - 1)))
        elif action == 2:
            new_states.append((i, max(0, j - 1)))
            new_states.append((max(0, i - 1), max(0, j - 1)))
            new_states.append((min(i + 1, GRID_WIDTH - 1), max(0, j - 1)))
        elif action == 3:
            new_states.append((max(0, i - 1), j))
            new_states.append((max(0, i - 1), min(j + 1, GRID_HEIGHT - 1)))
            new_states.append((max(0, i - 1), max(0, j - 1)))
        probs = [0.8, 0.1, 0.1]

        # results = []
        # for next_state in next_states:
            # reward = self.reward_func(next_state)
            # term, trunc, info = self.get_state_info(next_state)
            # results.append([next_state, reward, term, trunc, info])

        return new_states, probs

    @override
    def get_state_info(self, state: tuple) -> dict:
        info = {}
        term = False
        if state[0] == self.goal_state[0] and state[1] == self.goal_state[1]:
            term = True
            info["success"] = True
        elif state in self.obstacles:
            term = True
            info["success"] = False
        else:
            term = False
            info["success"] = False

        return term, False,info

    @override
    def reward_func(self, state: list, action: list = None, new_state: list = None) -> float:
        # R(s, s')
        # Transition to goal state gives goal reward.
        # Transition to obstacle gives obstacle reward.
        # Transition to free gives step reward.
        if new_state[0] == self.goal_state[0] and new_state[1] == self.goal_state[1]:
            reward = self.goal_reward
        elif new_state in self.obstacles:
            reward = self.obstacle_reward
        else:
            if not self.dense_reward:
                reward = self.step_reward
            else:
                # Negative of dist as penalty.
                # This encourages moving to goal.
                reward = -(abs(new_state[0] - self.goal_state[0]) + abs(new_state[1] - self.goal_state[1])) * 0.1

        return reward

    @override
    def reset(self):
        self.cur_state = self.start_state
        return self.cur_state, {}

    def add_path(self, path):
        self.path = path

    @override
    def render(self):
        _, ax = plt.subplots()
        self.gridworld = np.full((GRID_HEIGHT, GRID_WIDTH), 0)

        for obstacle in self.obstacles:
            self.gridworld[GRID_HEIGHT - obstacle[1] - 1][obstacle[0]] = 1

        self.gridworld[GRID_HEIGHT - self.start_state[1] - 1][self.start_state[0]] = 2
        self.gridworld[GRID_HEIGHT - self.goal_state[1] - 1][self.goal_state[0]] = 3

        if len(self.path) > 0:
            for state in self.path:
                self.gridworld[GRID_HEIGHT - state[1] - 1][state[0]] = 4
        else:
            self.gridworld[GRID_HEIGHT - self.cur_state[1] - 1][self.cur_state[0]] = 4

        self.colour_map = colors.ListedColormap(["white", "black", "yellow", "red", "green"])
        bounds = [0, 1, 2, 3, 4, 5]
        self.norm = colors.BoundaryNorm(bounds, self.colour_map.N)

        ax.imshow(self.gridworld, cmap=self.colour_map, norm=self.norm)

        # draw gridlines
        ax.grid(which="major", axis="both", linestyle="-", color="k", linewidth=1)
        ax.set_xticks(np.arange(GRID_WIDTH) - 0.5)
        ax.set_xticklabels(np.array([str(i) for i in range(GRID_WIDTH)]))
        ax.set_yticks(np.flip(np.arange(GRID_HEIGHT) + 0.5))
        ax.set_yticklabels(np.array([str(i) for i in range(GRID_HEIGHT)]))
        plt.tick_params(axis="both", labelsize=5, length=0)

        plt.show()
