from typing import Callable, Any
import heapq

from robotics_algorithm.env.base_env import DeterministicEnv


class HybridAStar(object):
    def __init__(self, env: DeterministicEnv, heuristic_func: Callable, state_key_func: Callable):
        """_summary_

        Args:
            env (ContinuousEnv): A planning env.
            heuristic_func (Callable): a function to return estimated cost-to-go from a state to goal
            transform_func (Callable): _description_
        """
        self.env = env
        self._heuristic_func = heuristic_func
        self._state_key_func = state_key_func

    def run(self, start: tuple, goal: tuple) -> tuple[bool, list[tuple], float]:
        """Run Hybrid A star.

        Args:
            start (tuple): the start state
            goal (tuple): the goal state

        Returns:
            res (bool): return true if a path is found, return false otherwise.
            shortest_path (list[tuple]): a list of state if shortest path is found.
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
        state_dict = {}  # key is some discreate state key, value is the actual continous state.

        start_key = self._state_key_func(start)
        goal_key = self._state_key_func(goal)

        g[start_key] = 0
        f[start_key] = self._heuristic_func(start, goal)  # cost from source to goal = g(s, s) + h(s, g) = 0 + h(s, g)
        heapq.heappush(priority_q, (f[start_key], start))
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
            _, best_state = heapq.heappop(priority_q)
            best_state_key = self._state_key_func(best_state)
            if best_state_key in open_set:
                open_set.remove(best_state_key)
                close_set.add(best_state_key)
            else:
                continue  # If best_state in close_set, just continue

            if best_state_key == goal_key:
                path_exist = True
                break

            # Find possible transitions from best_state, and add them to queue ranked by heuristics.
            available_actions = self.env.get_available_actions(best_state)
            for action in available_actions:
                new_state, cost, term, _, info = self.env.state_transition_func(best_state, action)
                # print(new_state, best_state, action)
                g_new_state = g[best_state_key] + cost  # cost-to-come
                new_state_key = self._state_key_func(new_state)

                # skip actions that result in failures
                if term and not info["success"]:
                    continue

                # if new_state has not been visited, or a shorter path to new_state has been found.
                if new_state_key not in open_set or g_new_state < g[new_state_key]:
                    h_new_state = self._heuristic_func(new_state, goal)  # cost-to-go
                    print(new_state_key, h_new_state)
                    f[new_state_key] = g_new_state + h_new_state
                    g[new_state_key] = g_new_state
                    heapq.heappush(priority_q, (f[new_state_key], new_state))
                    state_dict[new_state_key] = new_state
                    prev_state_dict[tuple(new_state)] = tuple(best_state), action
                    open_set.add(new_state_key)

                    # print("adding ", new_state_tup, new_action, self.heuristic(new_state))
                    # debug.append([new_action, new_state_tup])

            # print(debug)

        if path_exist:
            # extract shortest path:
            shortest_path = []
            state = tuple(state_dict[goal_key])
            while state in prev_state_dict:
                prev_state, prev_action = prev_state_dict[state]
                shortest_path.append(prev_action)
                state = prev_state

            shortest_path = reversed(shortest_path)
            shortest_path_len = f[goal_key]
            return (True, shortest_path, shortest_path_len)
        else:
            return (False, None, None)
