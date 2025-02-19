import matplotlib.pyplot as plt
import numpy as np
from typing_extensions import override

from robotics_algorithm.env.base_env import DiscreteSpace, DistributionType, MDPEnv


class FrozenLake(MDPEnv):
    """Frozen lake involves crossing a frozen lake from start to goal without falling into any holes by walking over
    the frozen lake. The player may not always move in the intended direction due to the slippery nature of the
    frozen lake.

    State: [x, y]
    Action: [up, down, left, right]

    Discrete state space.
    Discrete action space.
    Stochastic transition.
    Fully observable.
    """

    FREE = 0
    OBSTACLE = 1

    def __init__(self, size=5, num_of_obstacles=None, dense_reward: float = False, random_env: bool = False):
        """
        Constructor.

        Args:
            size (int): the size of the environment, the environment is a square of size x size.
            num_of_obstacles (int): the number of obstacles in the environment.
            dense_reward (bool): whether the environment uses dense reward.
            random_env (bool): whether to use a random environment.
        """
        super().__init__()

        self.size = size
        self.num_of_obstacles = size if num_of_obstacles is None else num_of_obstacles
        self.map = np.zeros((size, size), dtype=np.int8)
        self.obstacles = []
        self.cur_state = None
        self.dense_reward = dense_reward
        self.random_env = random_env

        # Get all indices inside the 2D grid
        indices = np.indices((size, size))
        all_states = np.stack(indices, axis=-1).reshape(-1, 2)

        # Define spaces
        self.state_space = DiscreteSpace([s for s in all_states])
        self.action_space = DiscreteSpace(
            [np.array([0]), np.array([1]), np.array([2]), np.array([3])]
        )  #  0: up, 1: right, 2: down, 3: left
        self.state_transition_dist_type = DistributionType.CATEGORICAL.value

    @override
    def reset(self):
        if not self.random_env:
            self.start_state = np.array([0, 0])
            self.goal_state = np.array([self.size - 1, self.size - 1])
            self.obstacles = [(1, 1), (3, 1), (1, 2), (4, 2), (4, 3), (2, 4)]
        else:
            for _ in range(self.num_of_obstacles):
                self.obstacles.append(self.random_valid_state())
            self.start_state = self.random_valid_state()
            self.goal_state = self.random_valid_state()

        self.cur_state = self.start_state
        return self.cur_state, {}

    @override
    def state_transition_func(self, state: np.ndarray, action: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
        i, j = state

        new_states = []
        if action == 0:  # up
            new_states.append(np.array([i, min(j + 1, self.size - 1)]))
            new_states.append(np.array([max(0, i - 1), min(j + 1, self.size - 1)]))
            new_states.append(np.array([min(i + 1, self.size - 1), min(j + 1, self.size - 1)]))
        elif action == 1:  # right
            new_states.append(np.array([min(i + 1, self.size - 1), j]))
            new_states.append(np.array([min(i + 1, self.size - 1), min(j + 1, self.size - 1)]))
            new_states.append(np.array([min(i + 1, self.size - 1), max(0, j - 1)]))
        elif action == 2:  # btm
            new_states.append(np.array([i, max(0, j - 1)]))
            new_states.append(np.array([max(0, i - 1), max(0, j - 1)]))
            new_states.append(np.array([min(i + 1, self.size - 1), max(0, j - 1)]))
        elif action == 3:  # left
            new_states.append(np.array([max(0, i - 1), j]))
            new_states.append(np.array([max(0, i - 1), min(j + 1, self.size - 1)]))
            new_states.append(np.array([max(0, i - 1), max(0, j - 1)]))
        probs = [0.9, 0.05, 0.05]

        return new_states, probs

    @override
    def is_state_terminal(self, state):
        term = False
        if state[0] == self.goal_state[0] and state[1] == self.goal_state[1]:
            term = True
        elif tuple(state.tolist()) in self.obstacles:
            term = True

        return term

    @override
    def reward_func(self, state: np.ndarray, action: np.ndarray = None, new_state: np.ndarray = None) -> float:
        # R(s, s')
        # Transition to goal state gives goal reward.
        # Transition to obstacle gives obstacle reward.
        # Transition to free gives step reward.
        if np.allclose(new_state, self.goal_state):
            reward = 100
        elif tuple(new_state.tolist()) in self.obstacles:
            reward = -100
        else:
            if not self.dense_reward:
                reward = -1
            else:
                # Negative of dist as penalty.
                # This encourages moving to goal.
                reward = -(abs(new_state[0] - self.goal_state[0]) + abs(new_state[1] - self.goal_state[1]))

        return reward

    def random_valid_state(self) -> tuple:
        valid = False
        while not valid:
            state = np.random.randint(0, self.size, (2))
            if self.map[state[0], state[1]] == FrozenLake.FREE:
                return state

    @override
    def render(self):
        plt.figure(figsize=(10, 10), dpi=100)
        s = 1000 / self.size / 2

        plt.scatter(
            [x[0] + 0.5 for x in self.obstacles],
            [x[1] + 0.5 for x in self.obstacles],
            s=s**2,
            c='black',
            marker='s',
        )
        plt.scatter(
            [self.start_state[0] + 0.5],
            [self.start_state[1] + 0.5],
            s=s**2,
            c='yellow',
            marker='s',
        )
        plt.scatter(
            [self.goal_state[0] + 0.5],
            [self.goal_state[1] + 0.5],
            s=s**2,
            c='red',
            marker='s',
        )
        plt.scatter(self.cur_state[0] + 0.5, self.cur_state[1] + 0.5, s=2500, c='blue', marker='s')

        plt.grid()
        plt.xlim(0, self.size)
        plt.xticks(np.arange(self.size))
        plt.ylim(0, self.size)
        plt.yticks(np.arange(self.size))
        plt.show()
