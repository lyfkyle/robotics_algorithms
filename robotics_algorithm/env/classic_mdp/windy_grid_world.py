import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from typing_extensions import override

from robotics_algorithm.env.base_env import DiscreteSpace, DistributionType, MDPEnv

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

    def __init__(
        self,
        start: np.ndarray = np.array([0, 3]),
        goal: np.ndarray = np.array([7, 3]),
        wind: np.ndarray = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
    ):
        """Constructor.

        Args:
            start (np.ndarray): the start position of agent.
            goal (np.ndarray): the goal position.
            wind (np.ndarray): the strength of upward wind in each horizontal position.
                      The number decides how many grid cell the agent will be pushed upward.
        """
        super().__init__()

        # Define spaces
        self.state_space = DiscreteSpace([np.array([i, j]) for i in range(GRID_WIDTH) for j in range(GRID_HEIGHT)])
        self.action_space = DiscreteSpace(
            [np.array([0]), np.array([1]), np.array([2]), np.array([3])]
        )  #  0: up, 1: right, 2: down, 3: left
        self.state_transition_dist_type = DistributionType.CATEGORICAL.value

        self.start_state = start
        self.cur_state = start
        self.goal_state = goal
        self.wind = wind
        self.step_reward = -1
        self.goal_reward = 10
        self.path = []

    @override
    def reset(self):
        self.cur_state = self.start_state
        return self.cur_state, {}

    @override
    def state_transition_func(self, state: np.ndarray, action: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
        # NOTE: This environment has deterministic transition
        i, j = state

        new_states = []
        if action == 0:
            next_state = np.array([i, min(j + 1 + self.wind[i], GRID_HEIGHT - 1)])
        elif action == 1:
            new_i = min(i + 1, GRID_WIDTH - 1)
            next_state = np.array([new_i, min(j + self.wind[new_i], GRID_HEIGHT - 1)])
        elif action == 2:
            next_state = np.array([i, max(0, min(j - 1 + self.wind[i], GRID_HEIGHT - 1))])
        elif action == 3:
            new_i = max(i - 1, 0)
            next_state = np.array([new_i, min(j + self.wind[new_i], GRID_HEIGHT - 1)])

        new_states.append(next_state)
        probs = [1.0]

        return new_states, probs

    @override
    def reward_func(self, state: np.ndarray, action: np.ndarray = None, new_state: np.ndarray = None) -> float:
        # R(s, s')
        # Transition to goal state gives goal reward.
        # Transition to free gives step reward.
        if np.allclose(new_state, self.goal_state):
            reward = 100
            reward = self.goal_reward
        else:
            reward = self.step_reward

        return reward

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

        self.colour_map = colors.ListedColormap(['white', 'black', 'yellow', 'red', 'green'])
        bounds = [0, 1, 2, 3, 4, 5]
        self.norm = colors.BoundaryNorm(bounds, self.colour_map.N)

        ax.imshow(self.gridworld, cmap=self.colour_map, norm=self.norm)

        # draw gridlines
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
        ax.set_xticks(np.arange(GRID_WIDTH) - 0.5)
        ax.set_xticklabels(np.array([str(i) for i in range(GRID_WIDTH)]))
        ax.set_yticks(np.flip(np.arange(GRID_HEIGHT) + 0.5))
        ax.set_yticklabels(np.array([str(i) for i in range(GRID_HEIGHT)]))
        plt.tick_params(axis='both', labelsize=5, length=0)

        plt.show()
