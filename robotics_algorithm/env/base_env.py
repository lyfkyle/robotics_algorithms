from typing import Any
from typing_extensions import override

import numpy as np
from numpy.random import choice


class DeterministicEnv:
    """Base environment for planning.

    Continuous/discrete state space.
    Continuous/discrete action space.
    Deterministic transition.
    Fully observable.
    """

    def __init__(self):
        # For continuous env, this is space.
        # For discrete env, this is a list of all states
        self.state_space = None
        self.action_space = None

        # Not applicable for continuous state space.
        self.state_space_size = None
        self.action_space_size = None

    def reset(self):
        """Reset env."""
        self.cur_state = None

    def step(self, action: Any) -> tuple[Any, float]:
        """Apply action to current environment. H

        Args:
            action (Any): action to apply

        Returns:
            new_state (any): new state.
            reward (float): the reward(cost) for the transition.
        """
        return self.state_transition_func(self.cur_state, action)

    def state_transition_func(self, state: Any, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """State transition function.

        Models both stochastic and deterministic transition.

        Args:
            state (Any): state to transit from
            action (Any): action to apply

        Returns:
            new_state (Any): the new state
            reward (float): the reward(cost) for the transition.
            term (bool): whether the transition terminates the episode
            trunc (bool): whether the transition truncates the episode
            info (dict): additional info
        """
        raise NotImplementedError()

    def render(self) -> None:
        """Visualize env."""
        pass

    def _get_state_info(self, state: Any) -> dict:
        """retrieve information about the state.

        Args:
            state (Any): state

        Returns:
            dict: a dictionary containing state information. Must contain term, trunc, reward to conform with
                gymanasium.
        """
        return {
            "term": False,
            "trunc": False,
        }

    def get_available_actions(self, state: Any) -> list[Any]:
        """Get available actions in the current state.

        Args:
            state (Any): the state

        Returns:
            list[Any]: a list of available actions for the given state.
        """
        raise NotImplementedError()

    def is_state_valid(self, state: Any) -> bool:
        """Check whether a state is valid

        Args:
            state (Any): state

        Returns:
            bool: return True if valid
        """
        raise NotImplementedError()


class StochasticEnv(DeterministicEnv):
    @override
    def step(self, action: Any) -> Any:
        """Apply action to current environment. H

        Args:
            action (Any): action to apply

        Returns:
            new_state (any): new state
        """
        results, probs = self.state_transition_func(self.cur_state, action)
        idx = choice(np.arange(len(results)), 1, p=probs)[0]  # choose new state according to the transition probability.

        self.cur_state = results[idx][0]

        # Conform to gymnasium env
        return results[idx]

    @override
    def state_transition_func(self, state: Any, action: Any) -> tuple[list[tuple], list[float]]:
        """State transition function.

        Models both stochastic and deterministic transition.

        Args:
            state (Any): state to transit from
            action (Any): action to apply

        Returns:
            new_states (list[tuple]): a list of state transition result, consisting of new_state, reward, term,
                trunc and info.
            probs (list[float]): the probabilities of transitioning to new states.
        """
        pass


class MDPEnv(StochasticEnv):
    """Markov Decision Process environments.

    Compared to DiscretePlanningEnv, it now has a reward function.
    """

    def __init__(self) -> None:
        pass

    def reward_func(self, state: Any, new_state: Any = None) -> float:
        """Calculate the reward of apply action at a state.

        NOTE: Following the gym convention, reward is defined as R(s, s') instead of R(s, a)

        Args:
            state (Any): the state
            action (Any, optional): the action to apply. Defaults to None.

        Returns:
            float: reward
        """
        pass


class POMDPEnv(MDPEnv):
    """Partially Observable Markov Decision Process environments.

    Compared to MDP, it in addition has observation function.
    """

    def __init__(self):
        super().__init__()
        self._observation_space = None

    def observation_func(self, state: Any) -> tuple[list[Any], list[float]]:
        """Calculate the possible observations for a given state.

        Args:
            state (Any): the state

        Returns:
            observations ([list[Any]): a list of possible observations.
            obs_probs (list[float]]): probabilities of each observation.
        """
        pass

    @override
    def step(self):
        new_state, reward, term, trunc, info = super().step()

        observations, obs_prob = self.observation_func(new_state)
        new_obs = choice(observations, 1, obs_prob)  # choose new observation according to the transition probability.

        return new_obs, reward, term, trunc, info
