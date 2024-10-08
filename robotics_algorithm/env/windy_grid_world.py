import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from typing_extensions import override

from robotics_algorithm.env.base_env import MDPEnv, DiscreteSpace, DistributionType

GRID_HEIGHT = 7
GRID_WIDTH = 10


class WindyGridWorld(MDPEnv):
    """
    Windy Gridworld is a standard gridworld with start and goal states. The difference is that there is a crosswind
    running upward through the middle of the grid. Actions are the standard four: up, right, down, and left.
    In the middle region the resultant next states are shifted upward by the "wind" which strength varies from column
    to column. The reward is -1 until goal state is reached.

    State: [x, y]
    Action: [up, down, left, right]

    Discrete state space.
    Discrete action space.
    Deterministic transition.
    Fully observable.
    """

    def __init__(self, start: tuple = (0, 3), goal: tuple = (7, 3), wind: list = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]):
        """Constructor.

        Args:
            start (tuple): the start position of agent.
            goal (tuple): the goal position.
            wind (list): the strength of upward wind in each horizontal position.
                      The number decides how many grid cell the agent will be pushed upward.
        """
        super().__init__()

        # Define spaces
        self.state_space = DiscreteSpace([(i, j) for i in range(GRID_WIDTH) for j in range(GRID_HEIGHT)])
        self.action_space = DiscreteSpace([0, 1, 2, 3])  # action, 0: up, 1: right, 2: down, 3: left
        self.state_transition_dist_type = DistributionType.CATEGORICAL.value

        self.start_state = start
        self.cur_state = start
        self.goal_state = goal
        self.wind = wind
        self.step_reward = -1
        self.goal_reward = 10
        self.path = []

    @override
    def state_transition_func(self, state: tuple, action: int) -> tuple[list[tuple], list[float]]:
        # NOTE: This environment has deterministic transition
        i, j = state

        new_states = []
        if action == 0:
            next_state = (i, min(j + 1 + self.wind[i], GRID_HEIGHT - 1))
        elif action == 1:
            new_i = min(i + 1, GRID_WIDTH - 1)
            next_state = (new_i, min(j + self.wind[new_i], GRID_HEIGHT - 1))
        elif action == 2:
            next_state = (i, max(0, min(j - 1 + self.wind[i], GRID_HEIGHT - 1)))
        elif action == 3:
            new_i = max(i - 1, 0)
            next_state = (new_i, min(j + self.wind[new_i], GRID_HEIGHT - 1))

        new_states.append(next_state)
        probs = [1.0]

        return new_states, probs

    @override
    def get_state_info(self, state: tuple) -> dict:
        info = {}
        term = False
        if state[0] == self.goal_state[0] and state[1] == self.goal_state[1]:
            term = True
            info["success"] = True
        else:
            term = False
            info["success"] = False

        return term, False, info

    @override
    def reward_func(self, state: list, action: list = None, new_state: list = None) -> float:
        # R(s, s')
        # Transition to goal state gives goal reward.
        # Transition to free gives step reward.
        if new_state[0] == self.goal_state[0] and new_state[1] == self.goal_state[1]:
            reward = self.goal_reward
        else:
            reward = self.step_reward

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
