from typing_extensions import override

import numpy as np
import matplotlib.pyplot as plt

from .base_env import MDPEnv


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

    def __init__(self, size=5, num_of_obstacles=None, dense_reward: float = False):
        super().__init__()

        self.size = size
        self.num_of_obstacles = size if num_of_obstacles is None else num_of_obstacles
        self.map = np.zeros((size, size), dtype=np.int8)
        self.obstacles = []
        self.cur_state = None
        self.dense_reward = dense_reward

        # Get all indices inside the 2D grid
        indices = np.indices((size, size))
        all_states = np.stack(indices, axis=-1).reshape(-1, 2).tolist()

        # Define spaces
        self.state_space = [tuple(s) for s in all_states]
        self.action_space = [0, 1, 2, 3]  #  0: up, 1: right, 2: down, 3: left

    @override
    def reset(self, random_env=True) -> tuple:
        if not random_env:
            self.start_state = (0, 0)
            self.goal_state = (self.size - 1, self.size - 1)
            self.obstacles = [(1, 1), (3, 1), (1, 2), (4, 2), (4, 3), (2, 4)]
        else:
            for _ in range(self.num_of_obstacles):
                self.obstacles.append(self.random_valid_state())
            self.start_state = self.random_valid_state()
            self.goal_state = self.random_valid_state()

        self.cur_state = self.start_state
        return self.cur_state, {}

    @override
    def state_transition_func(self, state: tuple, action: tuple) -> tuple[tuple, float]:
        # sanity check
        info = self._get_state_info(state)
        assert not info["term"]

        i, j = state

        next_states = []
        if action == 0:
            next_states.append((i, min(j + 1, self.size - 1)))
            next_states.append((max(0, i - 1), min(j + 1, self.size - 1)))
            next_states.append((min(i + 1, self.size - 1), min(j + 1, self.size - 1)))
        elif action == 1:
            next_states.append((min(i + 1, self.size - 1), j))
            next_states.append((min(i + 1, self.size - 1), min(j + 1, self.size - 1)))
            next_states.append((min(i + 1, self.size - 1), max(0, j - 1)))
        elif action == 2:
            next_states.append((i, max(0, j - 1)))
            next_states.append((max(0, i - 1), max(0, j - 1)))
            next_states.append((min(i + 1, self.size - 1), max(0, j - 1)))
        elif action == 3:
            next_states.append((max(0, i - 1), j))
            next_states.append((max(0, i - 1), min(j + 1, self.size - 1)))
            next_states.append((max(0, i - 1), max(0, j - 1)))
        probs = [0.8, 0.1, 0.1]

        results = []
        for next_state in next_states:
            reward = self.reward_func(next_state)
            info = self._get_state_info(next_state)
            results.append([next_state, reward, info["term"], False, info])

        return results, probs

    @override
    def get_available_actions(self, state: tuple) -> list[tuple]:
        return self.action_space

    @override
    def _get_state_info(self, state: tuple) -> dict:
        info = {}
        if state[0] == self.goal_state[0] and state[1] == self.goal_state[1]:
            info["term"] = True
            info["success"] = True
        elif state in self.obstacles:
            info["term"] = True
            info["success"] = False
        else:
            info["term"] = False
            info["success"] = False

        return info

    @override
    def reward_func(self, state: tuple, new_state: tuple | None = None) -> float:
        # R(s, s')
        # Transition to goal state gives goal reward.
        # Transition to obstacle gives obstacle reward.
        # Transition to free gives step reward.
        if state == self.goal_state:
            reward = 100
        elif state in self.obstacles:
            reward = -100
        else:
            if not self.dense_reward:
                reward = -1
            else:
                # Negative of dist as penalty.
                # This encourages moving to goal.
                reward = -(abs(state[0] - self.goal_state[0]) + abs(state[1] - self.goal_state[1]))

        return reward

    def random_valid_state(self) -> tuple:
        valid = False
        while not valid:
            state = np.random.randint(0, self.size, (2))
            if self.map[state[0], state[1]] == FrozenLake.FREE:
                return (state[0], state[1])

    @override
    def render(self):
        plt.figure(figsize=(10, 10), dpi=100)
        s = 1000 / self.size / 2

        plt.scatter(
            [x[0] + 0.5 for x in self.obstacles],
            [x[1] + 0.5 for x in self.obstacles],
            s=s**2,
            c="black",
            marker="s",
        )
        plt.scatter(
            [self.start_state[0] + 0.5],
            [self.start_state[1] + 0.5],
            s=s**2,
            c="yellow",
            marker="s",
        )
        plt.scatter(
            [self.goal_state[0] + 0.5],
            [self.goal_state[1] + 0.5],
            s=s**2,
            c="red",
            marker="s",
        )
        plt.scatter(self.cur_state[0] + 0.5, self.cur_state[1] + 0.5, s=2500, c="blue", marker="s")

        plt.grid()
        plt.xlim(0, self.size)
        plt.xticks(np.arange(self.size))
        plt.ylim(0, self.size)
        plt.yticks(np.arange(self.size))
        plt.show()
