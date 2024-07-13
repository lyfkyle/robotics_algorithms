from typing import Any
from typing_extensions import override

import numpy as np
from numpy.random import choice


class PlanningEnv:
    """Base environment for planning.

    Continuous state space.
    Continuous action space.
    Deterministic transition.
    Fully observable.
    """

    def __init__(self):
        self._state_space = None
        self._action_space = None

    def reset(self):
        """Reset env."""
        self._cur_state = None

    def step(self, action: Any) -> tuple[Any, float]:
        """Apply action to current environment. H

        Args:
            action (Any): action to apply

        Returns:
            new_state (any): new state.
            reward (float): the reward(cost) for the transition.
        """
        new_state, reward = self.state_transition_func(self._cur_state, action)
        return new_state, reward

    def state_transition_func(self, state: Any, action: Any) -> tuple[Any, float]:
        """State transition function.

        Models both stochastic and deterministic transition.

        Args:
            state (Any): state to transit from
            action (Any): action to apply

        Returns:
            new_state (Any): the new state
            reward (float): the reward(cost) for the transition.
        """
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


class StochasticPlanningEnv(PlanningEnv):
    @override
    def step(self, action: Any) -> Any:
        """Apply action to current environment. H

        Args:
            action (Any): action to apply

        Returns:
            new_state (any): new state
        """
        new_states, rewards, probs = self.state_transition_func(self._cur_state, action)
        idx = choice(np.arange(len(new_states)), 1, probs)  # choose new state according to the transition probability.

        new_state = new_states[idx]
        reward = rewards[idx]
        info = self._get_state_info(new_state)

        # Conform to gymnasium env
        return new_state, reward, info["term"], info["trunc"], info

    @override
    def state_transition_func(self, state: Any, action: Any) -> tuple[list[Any], list[float], list[float]]:
        """State transition function.

        Models both stochastic and deterministic transition.

        Args:
            state (Any): state to transit from
            action (Any): action to apply

        Returns:
            new_states (list[Any]): a list of possible new states.
            rewards (float): a list of reward(cost) for the transition.
            probs (list[float]): the probabilities of transitioning to new states.
        """
        pass


class DiscretePlanningEnv(PlanningEnv):
    """A planning environment with discrete state and discrete action space."""

    def __init__(self):
        self.all_states = None
        self.all_actions = None

    def get_available_actions(self, state: Any) -> list[Any]:
        """Get available actions in the current state.

        Args:
            state (Any): the state

        Returns:
            list[Any]: a list of available actions for the given state.
        """
        pass


class MDPEnv(DiscretePlanningEnv, StochasticPlanningEnv):
    """Markov Decision Process environments.

    Compared to DiscretePlanningEnv, it now has a reward function.
    """

    def __init__(self) -> None:
        pass

    def reward_func(self, state: Any, action: Any = None) -> float:
        """Calculate the reward of apply action at a state.

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
