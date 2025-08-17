from typing import Callable, Any
import heapq
import numpy as np

from robotics_algorithm.env.base_env import BaseEnv, SpaceType, EnvType


class TrajOptiPathPlanning:
    def __init__(self, env: BaseEnv):
        """Constructor.

        Args:
            env (BaseEnv): A planning env.
            heuristic_func (Callable): a function to return estimated cost-to-go from a state to goal
            state_key_func (Callable): a function to hash the state.
        """
        assert env.action_space.type == SpaceType.DISCRETE.value
        assert env.state_space.type == SpaceType.CONTINUOUS.value
        assert env.state_transition_type == EnvType.DETERMINISTIC.value
        assert env.observability == EnvType.FULLY_OBSERVABLE.value

        self.env = env
        self._heuristic_func = heuristic_func
        self._state_key_func = state_key_func

    def run(self, start: np.ndarray, goal: np.ndarray) -> tuple[bool, np.ndarray, float]:
        """Run algorithm.

        Args:
            start (np.ndarray): the start state
            goal (np.ndarray): the goal state

        Returns:
            res (bool): return true if a path is found, return false otherwise.
            shortest_path (np.ndarray): a np.ndarray of state if shortest path is found.
            shortest_path_len (float): the length of shortest path if found.
        """


