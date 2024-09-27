from typing_extensions import override

import numpy as np
import matplotlib.pyplot as plt

from robotics_algorithm.env.base_env import StochasticEnv, PartiallyObservableEnv, DiscreteSpace

# center value being the probability of perfect control
TRANSITION_PROB = [0.05, 0.1, 0.7, 0.1, 0.05]
# center value being the probability of perfect measurement
OBS_PROB = [0.05, 0.1, 0.7, 0.1, 0.05]


class DiscreteWorld1D(StochasticEnv, PartiallyObservableEnv):
    """A robot moves on a straight line and can only end up in integer value positions.

    State: [pos]
    Action: steps to move

    Discrete state space.
    Discrete action space.
    Stochastic transition.
    Partially observable.
    """
    def __init__(self, size=100, action_size=5, state_transition_prob=TRANSITION_PROB, obs_prob=OBS_PROB):
        super().__init__()

        self.size = size

        self.state_space = DiscreteSpace(values=[[x] for x in range(size)])
        self.action_space = DiscreteSpace(values=[[x] for x in range(-action_size, action_size + 1)])
        self.observation_space = DiscreteSpace(values=[[x] for x in range(size)])

        self.state_transition_prob = state_transition_prob
        self.obs_prob = obs_prob

    @override
    def reset(self):
        self.cur_state = self.state_space.sample()
        self.goal_state = [self.size // 2]

        return self.sample_observation(self.cur_state), {}

    @override
    def render(self):
        plt.figure(dpi=100)
        plt.plot(self.cur_state[0], 0, "o")
        plt.ylim(-1, 1)
        plt.xlim(0, self.size)
        plt.show()

    @override
    def state_transition_func(self, state: list, action: list) -> tuple[list[tuple], list[float]]:
        state = state[0]
        action = action[0]

        # move in the specified direction with some small chance of error
        tmp = int(len(self.state_transition_prob) // 2)
        new_state = state + action
        new_states = [[x] for x in range(new_state - tmp, new_state + tmp + 1)]
        probs = self.state_transition_prob

        results = []
        for new_state in new_states:
            reward = self.reward_func(new_state)
            term, trunc, info = self._get_state_info(new_state)

            # clamp state
            if new_state[0] < 0:
                new_state = [0]
            if new_state[0] >= self.size:
                new_state = [self.size - 1]

            results.append([new_state, reward, term, trunc, info])

        return results, probs

    @override
    def reward_func(self, state: list) -> float:
        # Check bounds
        if state[0] < 0 or state[0] >= self.size:
            return -100

        return -abs(state[0])

    @override
    def observation_func(self, state: list) -> tuple[list[list], list[float]]:
        state = state[0]
        tmp = int(len(self.obs_prob) // 2)
        obss = [[x] for x in range(state - tmp, state + tmp + 1)]
        probs = self.obs_prob

        return obss, probs

    def _get_state_info(self, state: list) -> tuple[bool, bool, dict]:
        term = False
        info = {}
        info["success"] = False
        if state[0] < 0 or state[0] >= self.size:
            term = True

        if state[0] == self.goal_state[0]:
            term = True
            info["success"] = True

        return term, False, info
