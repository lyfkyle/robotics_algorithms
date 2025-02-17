import numpy as np

from robotics_algorithm.env.base_env import BaseEnv, SpaceType, EnvType


class DiscreteBayesFilter:
    """Bayes filter for environment with discrete state and observations.

    Analytically computes bayes update rule by iterating over all possible states and observations.
    """
    def __init__(self, env: BaseEnv):
        assert env.state_space.type == SpaceType.DISCRETE.value
        assert env.observation_space.type == SpaceType.DISCRETE.value
        assert env.state_transition_type == EnvType.STOCHASTIC.value
        assert env.observability == EnvType.PARTIALLY_OBSERVABLE.value

        self.env = env
        self._state_space_size = env.state_space.size

        # Uniformly distributed
        self.state_belief = {}
        for state in self.env.state_space.get_all():
            self.state_belief[tuple(state)] = 1 / self._state_space_size

    def set_initial_state(self, state: np.ndarray):
        """
        Set the initial state of filter.

        @param state, initial state
        """
        self.state_belief[tuple(state)] += 0.05  # slightly increase the initial state belief
        self._normalize()

    def get_state(self) -> np.ndarray:
        # return the state with highest probability
        return max(self.state_belief, key=self.state_belief.get)

    def run(self, action: np.ndarray, obs: np.ndarray):
        """
        Run one iteration of filter.

        @param action, control
        @param obs, observation
        """
        self.predict(action)
        self.update(obs)

    def predict(self, action: np.ndarray):
        """
        @param action, action
        """
        new_state_belief = self.state_belief.copy()

        # In discrete case, just add them up.
        for state in self.env.state_space.get_all():
            new_states, probs = self.env.state_transition_func(state, action)
            for new_state, prob in zip(new_states, probs):
                new_state_belief[tuple(new_state)] += prob * self.state_belief[tuple(state)]

        self.state_belief = new_state_belief

        self._normalize()

    def update(self, obs: np.ndarray):
        """
        @param obs, current observation.
        """
        for state in self.env.state_space.get_all():
            obss, obs_probs = self.env.observation_func(state)

            obs_prob = 0
            if obs in obss:
                obs_prob = obs_probs[obss.index(obs)]

            self.state_belief[tuple(state)] *= obs_prob

        self._normalize()

    def _normalize(self):
        # normalize
        total_sum = sum(self.state_belief.values())
        for state in self.state_belief.keys():
            self.state_belief[state] /= total_sum
