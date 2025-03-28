import heapq
from typing import Any, Callable

import numpy as np

from robotics_algorithm.env.base_env import BaseEnv, EnvType, SpaceType


class Dijkstra:
    def __init__(self, env: BaseEnv, state_key_func: Callable):
        """Constructor.

        Args:
            env (BaseEnv): A planning env.
            state_key_func (Callable): a function to hash the state.
        """
        assert env.action_space.type == SpaceType.DISCRETE.value
        assert env.state_space.type == SpaceType.DISCRETE.value
        assert env.state_transition_type == EnvType.DETERMINISTIC.value
        assert env.observability == EnvType.FULLY_OBSERVABLE.value

        self.env = env
        self._state_key_func = state_key_func

    def run(self, start: Any, goal: Any) -> tuple[bool, np.ndarray, float]:
        """Run algorithm.

        Args:
            start (Any): the start state
            goal (Any): the goal state

        Returns:
            res (bool): return true if a path is found, return false otherwise.
            shortest_path (np.ndarray): a np.ndarray of state if shortest path is found.
            shortest_path_len (float): the length of shortest path if found.
        """
        # for every state, f[v] = g(s, v)

        # initialize
        unvisited_states = set()  # OPEN set. Nodes not in this set is in CLOSE set
        priority_q = []
        shortest_path = []
        shortest_path_len = 0
        g = {}  # cost-to-come from start to a state
        prev_state_dict = {}  # used to extract shortest path
        state_dict = {}  # key is some  state key, value is the actual state.

        start_key = self._state_key_func(start)
        goal_key = self._state_key_func(goal)
        state_dict[start_key] = start
        state_dict[goal_key] = goal

        for state in self.env.state_space.get_all():
            state_key = self._state_key_func(state)
            g[state_key] = float('inf')
            state_dict[state_key] = state
            unvisited_states.add(state_key)

        g[start_key] = 0  # distance to source is 0
        heapq.heappush(priority_q, (g[start_key], start))

        # run algorithm
        path_exist = False
        while len(priority_q) > 0:
            # pop the best state found so far.
            _, best_state = heapq.heappop(priority_q)
            best_state = np.array(best_state)
            best_state_key = self._state_key_func(best_state)

            # filter out state visited before. This may happen if multiple path leads to the same state with
            # different values.
            if best_state_key not in unvisited_states:
                continue
            else:
                unvisited_states.remove(best_state_key)

            # If goal state has been added.
            if best_state_key == goal_key:
                path_exist = True
                break

            print(best_state, best_state_key, g[best_state_key])
            # Find possible transitions from best_state, and add them to queue
            actions = self.env.action_space.get_all()
            for action in actions:
                new_state, reward, term, _, info = self.env.sample_state_transition(best_state, action)
                cost = -reward

                # skip actions that result in failures
                if term and not info['success']:
                    continue

                # if new_state has not been visited,
                new_state_key = self._state_key_func(new_state)
                if new_state_key in unvisited_states:
                    g_new_state = g[best_state_key] + cost  # cost-to-come
                    if g_new_state < g[new_state_key]:
                        g[new_state_key] = g_new_state
                        # NOTE: here we do not need to remove the the previous stored new_state with old best-to-come
                        #       value because of new_state will be marked as visited when first poped with the best
                        #       cost-to-come value.
                        heapq.heappush(priority_q, (g[new_state_key], new_state.tolist()))
                        prev_state_dict[new_state_key] = best_state_key

        if path_exist:
            # extract shortest path:
            shortest_path = []
            state_key = goal_key
            while state_key in prev_state_dict:
                prev_state_key = prev_state_dict[state_key]
                shortest_path.append(state_dict[prev_state_key])
                state_key = prev_state_key

            shortest_path = list(reversed(shortest_path))
            shortest_path_len = g[goal_key]
            return (True, shortest_path, shortest_path_len)
        else:
            return (False, None, None)
