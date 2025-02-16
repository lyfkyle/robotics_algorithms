from typing_extensions import override
from enum import Enum

import numpy as np
from numpy.random import choice


class EnvType(Enum):
    DETERMINISTIC = 0
    STOCHASTIC = 1
    FULLY_OBSERVABLE = 2
    PARTIALLY_OBSERVABLE = 3


class SpaceType(Enum):
    DISCRETE = 0
    CONTINUOUS = 1


class FunctionType(Enum):
    GENERAL = 0
    LINEAR = 1
    QUADRATIC = 2


class DistributionType(Enum):
    NOT_APPLICABLE = -1
    GENERAL = 0
    CATEGORICAL = 1
    GAUSSIAN = 2


class BaseEnv:
    """
    Base class for environment that is used for planning, control and learning.

    Continuous/discrete state and action space.
    Deterministic/stochastic transition.
    Fully/partially observable.
    """

    def __init__(self):
        # For continuous env, this is the space bounds in the form of [low, high].
        # For discrete env, this is a np.ndarray of all states
        self.state_space = None
        self.action_space = None
        self.observation_space = None

        # Env characteristic
        self.state_transition_type = None
        self.observability = None

        # Function type
        self.state_transition_func_type = FunctionType.GENERAL.value
        self.reward_func_type = FunctionType.GENERAL.value
        self.observation_func_type = FunctionType.GENERAL.value

        # Noise type
        self.state_transition_dist_type = DistributionType.NOT_APPLICABLE.value
        self.observation_dist_type = DistributionType.NOT_APPLICABLE.value

    def reset(self) -> tuple[np.ndarray, dict]:
        """Reset env."""
        self.cur_state = None

        return self.sample_observation(self.cur_state), {}

    def render(self) -> None:
        """Visualize env."""
        pass

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply action to current environment.

        This corresponds to gymnasium convention.

        Args:
            action (np.ndarray): action to apply

        Returns:
            new_state (np.ndarray): the new state
            reward (double): the reward obtained
            term (bool): whether the state is terminal
            trunc (bool): whether the episode is truncated
            info (dict): additional info
        """
        new_state, reward, term, trunc, info = self.sample_state_transition(self.cur_state, action)
        self.cur_state = new_state

        # replace state with its observation
        new_obs = self.sample_observation(new_state)

        # Conform to gymnasium env
        return new_obs, reward, term, trunc, info

    def random_state(self) -> np.ndarray:
        """Sample a state"""
        return self.state_space.sample()

    def random_action(self) -> np.ndarray:
        """Sample an action"""
        return self.action_space.sample()

    def random_observation(self) -> np.ndarray:
        """Sample an random observation"""
        return self.observation_space.sample()

    def sample_state_transition(
        self, state: np.ndarray, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Sample a state transition.

        Args:
            state (np.ndarray): state
            action (np.ndarray): action

        Returns:
            new_state (np.ndarray): the new state
            reward (double): the reward obtained
            term (bool): whether the state is terminal
            trunc (bool): whether the episode is truncated
            info (dict): additional info
        """
        raise NotImplementedError()

    def sample_observation(self, state: np.ndarray) -> np.ndarray:
        """Sample an observation for the state."""
        raise NotImplementedError()

    def state_transition_func(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Calculate the next state given current state and action.

        Args:
            state (np.ndarray): state
            action (np.ndarray): action

        Returns:
            new_state (np.ndarray): the new state
        """
        raise NotImplementedError()

    def linearize_state_transition(self, state: np.ndarray, action: np.ndarray):
        """Linearize the state transition function around current state

        Args:
            state (np.ndarray): state

        Returns:
            A, B such that new_state = A @ state + B @ action
        """
        raise NotImplementedError()

    def observation_func(self, state: np.ndarray) -> np.ndarray:
        """Calculate the possible observations for a given state.

        Args:
            state (np.ndarray): state

        Returns:
            obs (np.ndarray): observation
        """
        raise NotImplementedError()

    def reward_func(self, state: np.ndarray, action: np.ndarray = None, new_state: np.ndarray = None) -> float:
        """Calculate the reward of apply action at a state.

        Reward function is normally defined as R(s, a). However, in the gym convention, reward is defined as R(s, s').
        This base function supports both.

        Args:
            state (np.ndarray): the state
            action (np.ndarray, optional): the action to apply. Defaults to None.
            new_state (np.ndarray, optional): the new_state after transition. Defaults to None.

        Returns:
            reward (float)
        """
        raise NotImplementedError()

    def is_state_valid(self, state: np.ndarray) -> bool:
        """Check whether a state is valid

        Args:
            state (np.ndarray): state

        Returns:
            bool: return True if valid
        """
        raise NotImplementedError()

    def _is_action_valid(self, action) -> bool:
        res = False
        if self.action_space.type == SpaceType.DISCRETE.value:
            res = action in self.action_space.space
        elif self.action_space.type == SpaceType.CONTINUOUS.value:
            res = (np.array(action) >= np.array(self.action_space.space[0])).all() and (
                np.array(action) <= np.array(self.action_space.space[1])
            ).all()

        return res

    def get_state_info(self, state: np.ndarray) -> tuple[bool, bool, dict]:
        """Returns additional flag and info associated with state

        Args:
            state (np.ndarray): state

        Returns:
            tuple[bool, bool, dict]: term, trunc and info. In accordance to gymanasium convention.
        """
        raise NotImplementedError()


class DiscreteSpace:
    def __init__(self, values):
        self.type = SpaceType.DISCRETE.value
        self.space = values

    @property
    def size(self):
        return len(self.space)

    @property
    def state_size(self):
        return len(self.space[0])

    def get_all(self):
        return self.space

    def sample(self):
        idx = np.random.randint(len(self.space))
        return self.space[idx]


class ContinuousSpace:
    def __init__(self, low, high):
        self.type = SpaceType.CONTINUOUS.value
        self.space = [low, high]

    @property
    def state_size(self):
        return len(self.space[0])

    def sample(self):
        return np.random.uniform(np.nan_to_num(self.space[0]), np.nan_to_num(self.space[1]))


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
    def sample_state_transition(self, state, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Skip invalid action
        if not self._is_action_valid(action):
            raise Exception(f'action {action} is not within valid range!')

        new_state = self.state_transition_func(state, action)

        reward = self.reward_func(state, action, new_state)
        term, trunc, info = self.get_state_info(new_state)

        return new_state, reward, term, trunc, info

    @override
    def state_transition_func(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """State transition function.

        Models deterministic transition and therefore returns a single new_state.

        Args:
            state (np.ndarray): state to transit from
            action (np.ndarray): action to apply

        Returns:
            new_state (np.ndarray): the new state
        """
        raise NotImplementedError()


class StochasticEnv(BaseEnv):
    def __init__(self):
        super().__init__()
        self.state_transition_type = EnvType.STOCHASTIC.value

    @override
    def sample_state_transition(self, state, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Skip invalid action
        if not self._is_action_valid(action):
            raise Exception(f'action {action} is not within valid range!')

        if self.state_transition_dist_type == DistributionType.CATEGORICAL.value:
            assert self.state_space.type == SpaceType.DISCRETE.value

            results, probs = self.state_transition_func(state, action)
            idx = choice(np.arange(len(results)), 1, p=probs)[
                0
            ]  # choose new state according to the transition probability.

            new_state = results[idx]

        elif self.state_transition_dist_type == DistributionType.GAUSSIAN.value:
            assert self.state_space.type == SpaceType.CONTINUOUS.value

            mean, var = self.state_transition_func(state, action)
            new_state = np.random.multivariate_normal(mean=np.array(mean), cov=np.diag(var)).reshape(-1)

        else:
            raise NotImplementedError()

        # compute reward
        reward = self.reward_func(state, action, new_state)
        term, trunc, info = self.get_state_info(new_state)

        return new_state, reward, term, trunc, info

    @override
    def state_transition_func(
        self, state: np.ndarray, action: np.ndarray
    ) -> tuple[np.ndarray[np.ndarray], np.ndarray[float]] | tuple[np.ndarray, np.ndarray]:
        """State transition function.

        Models stochastic transition. Hence, we return each possible transition results with its corresponding
        probability.

        Args:
            state (np.ndarray): state to transit from
            action (np.ndarray): action to apply

        Returns:
            if state transition distribution type is categorical, returns possible state transition results with its
            associated probability.
                new_states (np.ndarray[np.ndarray]): a np.ndarray of new states
                probs (np.ndarray[float]): the probabilities of transitioning to new states.

            else if state transition distribution type is gaussian, returns the mean and variance of new state
            state transition result.
        """
        raise NotImplementedError()

    def get_state_transition_prob(self, state: np.ndarray, action: np.ndarray, new_state: np.ndarray) -> float:
        """Get the probability of transitioning to new_state from state with action

        Args:
            state (np.ndarray): state
            action (np.ndarray): action
            new_state (np.ndarray): new_state

        Returns:
            float: probability of the transition
        """
        raise NotImplementedError()


class FullyObservableEnv(BaseEnv):
    def __init__(self):
        super().__init__()
        self.observability = EnvType.FULLY_OBSERVABLE.value

    @property
    def observation_space_size(self):
        return len(self.state_space)

    @override
    def observation_func(self, state: np.ndarray) -> np.ndarray:
        """Return the observation for a given state.

        Since the environment is fully-observable, observation equals to the state.

        Args:
            state (np.ndarray): the state

        Returns:
            observations (np.ndarray):  observations.
        """
        return state

    @override
    def sample_observation(self, state: np.ndarray) -> np.ndarray:
        return self.observation_func(self.cur_state)


class PartiallyObservableEnv(BaseEnv):
    def __init__(self):
        super().__init__()
        self.observability = EnvType.PARTIALLY_OBSERVABLE.value

        # Define an additional observation space
        self.observation_space = None

    def sample_observation(self, state) -> np.ndarray:
        if self.observation_dist_type == DistributionType.CATEGORICAL.value:
            obss, obs_prob = self.observation_func(state)
            idx = choice(np.arange(len(obss)), 1, p=obs_prob)[
                0
            ]  # choose observation according to the observation probability.
            obs = obss[idx]

        elif self.observation_dist_type == DistributionType.GAUSSIAN.value:
            assert self.state_space.type == SpaceType.CONTINUOUS.value

            mean, var = self.observation_func(state)
            obs = np.random.multivariate_normal(mean=np.array(mean), cov=np.diag(var)).reshape(-1)

        else:
            raise NotImplementedError()

        return obs

    @override
    def observation_func(self, state: np.ndarray) -> tuple[np.ndarray[np.ndarray], np.ndarray[float]]:
        """Return the observation for a given state.

        Note the environment is partially-observable.
        Therefore, for discrete env, this function return each possible observations with its probability.
        For continuous env, we should return a parameterized distributions. (Unsupported for now).

        Args:
            state (np.ndarray): the state

        Returns:
            observations ([np.ndarray[np.ndarray]): a np.ndarray of possible observations.
            obs_probs (np.ndarray[float]]): probabilities of each observation.
        """
        raise NotImplementedError()

    def get_observation_prob(self, state: np.ndarray, observation: np.ndarray) -> float:
        """Get the probability of getting the observation from state.

        Args:
            state (np.ndarray): state
            observation (np.ndarray): observation

        Returns:
            float: probability of the observation given the state.
        """
        raise NotImplementedError()


class MDPEnv(StochasticEnv, FullyObservableEnv):
    """Markov Decision Process environments."""

    def __init__(self):
        super().__init__()


class POMDPEnv(StochasticEnv, PartiallyObservableEnv):
    """Partially Observable Markov Decision Process environments."""

    def __init__(self):
        super().__init__()
