from typing_extensions import override

import numpy as np
import matplotlib.pyplot as plt

from robotics_algorithm.env.base_env import StochasticEnv, PartiallyObservableEnv, DiscreteSpace, DistributionType

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
        self.state_transition_dist_type = DistributionType.CATEGORICAL.value
        self.observation_dist_type = DistributionType.CATEGORICAL.value

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
    def state_transition_func(self, state: np.ndarray, action: np.ndarray) -> tuple[np.ndarray[tuple], np.ndarray[float]]:
        state = state[0]
        action = action[0]

        # move in the specified direction with some small chance of error
        tmp = int(len(self.state_transition_prob) // 2)
        new_state = state + action
        new_states = [[x] for x in range(new_state - tmp, new_state + tmp + 1)]
        probs = self.state_transition_prob

        for i in range(len(new_states)):
            # clamp state
            if new_states[i][0] < 0:
                new_states[i] = [0]
            if new_states[i][0] >= self.size:
                new_states[i] = [self.size - 1]

        return new_states, probs

    @override
    def reward_func(self, state: np.ndarray, action: np.ndarray = None, new_state: np.ndarray = None) -> float:
        # Check bounds
        if new_state[0] < 0 or new_state[0] >= self.size:
            return -100

        return -abs(new_state[0])

    @override
    def observation_func(self, state: np.ndarray) -> tuple[np.ndarray[np.ndarray], np.ndarray[float]]:
        state = state[0]
        tmp = int(len(self.obs_prob) // 2)
        obss = [[x] for x in range(state - tmp, state + tmp + 1)]
        probs = self.obs_prob

        return obss, probs

    @override
    def is_state_terminal(self, state):
        term = False
        if state[0] < 0 or state[0] >= self.size:
            term = True

        if state[0] == self.goal_state[0]:
            term = True

        return term
