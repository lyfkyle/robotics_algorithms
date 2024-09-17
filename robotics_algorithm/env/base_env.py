from typing import Any
from typing_extensions import override
from enum import Enum

import numpy as np
from numpy.random import choice


class EnvType(Enum):
    DISCRETE = 0
    CONTINUOUS = 1
    DETERMINISTIC = 2
    STOCHASTIC = 3
    FULLY_OBSERVABLE = 4
    PARTIALLY_OBSERVABLE = 5


class FunctionType(Enum):
    GENERAL = 0
    LINEAR = 1
    QUADRATIC = 2


class NoiseType(Enum):
    GENERAL = 0
    GAUSSIAN = 3


class BaseEnv(object):
    """
    Base class for environment that is used for planning, control and learning.

    Continuous/discrete state and action space.
    Deterministic/stochastic transition.
    Fully/partially observable.
    """

    def __init__(self):
        # For continuous env, this is the space bounds in the form of [low, high].
        # For discrete env, this is a list of all states
        self.state_space = None
        self.action_space = None

        # Env characteristic
        self.space_type = None
        self.state_transition_type = None
        self.observability = None

        # Function type
        self.state_transition_type = FunctionType.GENERAL.value
        self.reward_func_type = FunctionType.GENERAL.value

        # Noise type
        self.state_transition_noise_type = NoiseType.GAUSSIAN.value
        self.observation_noise_type = NoiseType.GAUSSIAN.value

    def reset(self):
        """Reset env."""
        self.cur_state = None

    def render(self) -> None:
        """Visualize env."""
        pass

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Apply action to current environment. H

        Args:
            action (Any): action to apply

        Returns:
            new_state (any): new state
        """
        new_state, reward, term, trunc, info = self.sample_state_transition(self.cur_state, action)
        self.cur_state = new_state

        # replace state with its observation
        new_obs = self.sample_observation(new_state)

        # Conform to gymnasium env
        return new_obs, reward, term, trunc, info

    def sample_state(self) -> Any:
        """Sample a state"""
        raise NotImplementedError()

    def sample_action(self) -> Any:
        """Sample an action"""
        raise NotImplementedError()

    def sample_state_transition(self, state, action) -> tuple[Any, float, bool, bool, dict]:
        """Sample a transition."""
        raise NotImplementedError()

    def sample_observation(self, state) -> Any:
        """Sample an observation for the state."""
        raise NotImplementedError()

    def state_transition_func(self, state: Any, action: Any) -> Any:
        """Calculate the next state given current state and action"""
        raise NotImplementedError()

    def observation_func(self, state: Any) -> Any:
        """Calculate the possible observations for a given state."""
        raise NotImplementedError()

    def reward_func(self, state: Any, action: Any = None, new_state: Any = None) -> float:
        """Calculate the reward of apply action at a state.

        Reward function is normally defined as R(s, a). However, in the gym convention, reward is defined as R(s, s').
        This base function supports both.

        Args:
            state (Any): the state
            action (Any, optional): the action to apply. Defaults to None.
            new_state (Any, optional): the new_state after transition. Defaults to None.

        Returns:
            float: reward
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


class DiscreteEnv(BaseEnv):
    def __init__(self):
        super().__init__()
        self.space_type == EnvType.DISCRETE.value

    @property
    def state_space_size(self):
        return len(self.state_space)

    @property
    def action_space_size(self):
        return len(self.action_space)

    def get_available_actions(self, state: Any) -> list[Any]:
        """Get available actions in the current state.

        Args:
            state (Any): the state

        Returns:
            list[Any]: a list of available actions for the given state.
        """
        raise NotImplementedError()

    @override
    def sample_state(self):
        return np.random.choice(np.array(self.state_space))

    @override
    def sample_action(self):
        return np.random.choice(np.array(self.action_space))


class ContinuousEnv(BaseEnv):
    def __init__(self):
        super().__init__()
        self.space_type == EnvType.CONTINUOUS.value

    def sample_available_actions(self, state: Any, num_samples=1) -> list[Any]:
        """Sample an available action in the current state.

        Args:
            state (Any): the state
            num_samples (int): the number of samples

        Returns:
            list[Any]: a list of available actions for the given state.
        """
        raise NotImplementedError()

    @override
    def sample_state(self):
        return np.random.uniform(self.state_space[0], self.state_space[1])

    @override
    def sample_action(self):
        return np.random.uniform(self.action_space[0], self.action_space[1])


class DeterministicEnv(BaseEnv):
    """Base environment for planning.

    Continuous/discrete state space.
    Deterministic transition.
    Fully observable.
    """

    def __init__(self):
        super().__init__()
        self.state_transition_type = EnvType.DETERMINISTIC.value

    @override
    def sample_state_transition(self, state, action) -> tuple[Any, float, bool, bool, dict]:
        return self.state_transition_func(state, action)

    @override
    def state_transition_func(self, state: Any, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """State transition function.

        Models deterministic transition and therefore returns a single new_state along with its info

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


class StochasticEnv(DeterministicEnv):
    def __init__(self):
        super().__init__()
        self.state_transition_type = EnvType.STOCHASTIC.value

    def sample_transition_discrete(self, state, action) -> tuple[Any, float, bool, bool, dict]:
        assert self.space_type == EnvType.DISCRETE.value
        results, probs = self.state_transition_func(state, action)
        idx = choice(np.arange(len(results)), 1, p=probs)[
            0
        ]  # choose new state according to the transition probability.

        return results[idx]

    @override
    def state_transition_func(self, state: Any, action: Any) -> tuple[list[tuple], list[float]]:
        """State transition function.

        Models stochastic transition. Hence, we return each possible transition results with its corresponding
        probability.

        Args:
            state (Any): state to transit from
            action (Any): action to apply

        Returns:
            new_states (list[tuple]): a list of state transition result, consisting of new_state, reward, term,
                trunc and info.
            probs (list[float]): the probabilities of transitioning to new states.
        """
        pass


class FullyObservableEnv(BaseEnv):
    def __init__(self):
        super().__init__()
        self.observability = EnvType.FULLY_OBSERVABLE.value

    @property
    def observation_space(self):
        return len(self.state_space)

    @property
    def observation_space_size(self):
        return len(self.state_space)

    @override
    def observation_func(self, state: Any) -> Any:
        """Return the observation for a given state.

        Since the environment is fully-observable, observation equals to the state.

        Args:
            state (Any): the state

        Returns:
            observations (Any):  observations.
        """
        return state

    @override
    def sample_observation(self, state: Any) -> Any:
        return self.observation_func(self.cur_state)


class PartiallyObservableEnv(BaseEnv):
    def __init__(self):
        super().__init__()
        self.observability = EnvType.PARTIALLY_OBSERVABLE.value

        # Define an additional observation space
        self.observation_space = None

    def sample_observation_discrete(self, state) -> Any:
        assert self.space_type == EnvType.DISCRETE.value

        obss, obs_prob = self.observation_func(state)
        obs = choice(obss, 1, obs_prob)  # choose new observation according to the transition probability.
        return obs

    @override
    def observation_func(self, state: Any) -> tuple[list[Any], list[float]]:
        """Return the observation for a given state.

        Note the environment is partially-observable.
        Therefore, for discrete env, this function return each possible observations with its probability.
        For continuous env, we should return a parameterized distributions. (Unsupported for now).

        Args:
            state (Any): the state

        Returns:
            observations ([list[Any]): a list of possible observations.
            obs_probs (list[float]]): probabilities of each observation.
        """
        raise NotImplementedError()


class MDPEnv(DiscreteEnv, StochasticEnv, FullyObservableEnv):
    """Markov Decision Process environments."""

    def __init__(self):
        super().__init__()


class POMDPEnv(DiscreteEnv, StochasticEnv, PartiallyObservableEnv):
    """Partially Observable Markov Decision Process environments."""

    def __init__(self):
        super().__init__()
