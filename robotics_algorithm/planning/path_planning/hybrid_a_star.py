from typing import Callable, Any
import heapq
import numpy as np

from robotics_algorithm.env.base_env import BaseEnv, SpaceType, EnvType


class HybridAStar:
    def __init__(self, env: BaseEnv, heuristic_func: Callable, state_key_func: Callable):
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

        # initialize
        # for every state, f[v] = g(s, v) + h(v, g)
        priority_q = []
        shortest_path = []
        shortest_path_len = 0
        g = {}  # cost-to-come from start to a state
        f = {}  # cost-to-come + heuristic cost-to-go
        prev_state_dict = {}  # used to extract shortest path
        state_dict = {}  # key is some discrete state key, value is the actual continuous state.

        start_key = self._state_key_func(start)
        goal_key = self._state_key_func(goal)
        state_dict[start_key] = start
        state_dict[goal_key] = goal

        g[start_key] = 0
        f[start_key] = self._heuristic_func(start, goal)  # cost from source to goal = g(s, s) + h(s, g) = 0 + h(s, g)
        heapq.heappush(priority_q, (f[start_key], start_key))
        open_set = set()
        close_set = set()
        open_set.add(start_key)

        # run algorithm
        path_exist = False
        while True:
            if not open_set:
                print("Error: Cannot find path, No open set")
                break

            # pop the best state found so far.
            _, best_state_key = heapq.heappop(priority_q)
            if best_state_key in open_set:
                open_set.remove(best_state_key)
                close_set.add(best_state_key)
                best_state = state_dict[best_state_key]
            else:
                continue  # If best_state in close_set, just continue

            # If goal state has been added.
            if best_state_key == goal_key:
                path_exist = True
                break

            # print(f"close set num: {len(close_set)}/81000")
            print(best_state, best_state_key, g[best_state_key])
            # Find possible transitions from best_state, and add them to queue ranked by heuristics.
            actions = self.env.action_space.get_all()
            for action in actions:
                new_state, reward, term, _, info = self.env.sample_state_transition(best_state, action)
                cost = -reward

                # skip actions that result in failures
                if term and not info["success"]:
                    continue

                # print(new_state, best_state, action)

                # if new_state has not been visited, or a shorter path to new_state has been found.
                new_state_key = self._state_key_func(new_state)
                if new_state_key not in close_set:
                    g_new_state = g[best_state_key] + cost  # cost-to-come

                    if new_state_key not in open_set or g_new_state < g[new_state_key]:
                        h_new_state = self._heuristic_func(new_state, goal)  # cost-to-go
                        # print(new_state_key, h_new_state)
                        f[new_state_key] = g_new_state + h_new_state
                        g[new_state_key] = g_new_state
                        heapq.heappush(priority_q, (f[new_state_key], new_state_key))
                        state_dict[new_state_key] = new_state
                        prev_state_dict[new_state_key] = (best_state_key, action)
                        open_set.add(new_state_key)

                        # print("adding ", best_state, action, new_state, new_state_key, h_new_state)
                    # debug.append([new_action, new_state_tup])

            # print(debug)

        if path_exist:
            # extract shortest path:
            shortest_path = []
            state_key = goal_key
            while state_key in prev_state_dict:
                prev_state_key, prev_action = prev_state_dict[state_key]
                shortest_path.append(prev_action)
                state_key = prev_state_key

            shortest_path = list(reversed(shortest_path))
            shortest_path_len = f[goal_key]
            return (True, shortest_path, shortest_path_len)
        else:
            return (False, None, None)
