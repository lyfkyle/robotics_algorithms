from typing import Callable, Any
import heapq

from robotics_algorithm.env.base_env import DiscreteEnv


class AStar(object):
    def __init__(self, env: DiscreteEnv, heuristic_func: Callable):
        """_summary_

        Args:
            env (DiscreteEnv): A planning env.
            heuristic_func (Callable): a function to return estimated cost-to-go from a state to goal
        """
        self._env = env
        self._heuristic_func = heuristic_func

    def run(self, start: Any, goal: Any):
        """Run Astar.

        Args:
            start (Any): the start state
            goal (Any): the goal state

        Returns:
            res (bool): return true if a path is found, return false otherwise.
            shortest_path (list[Any]): a list of state if shortest path is found.
            shortest_path_len (float): the length of shortest path if found.
        """

        # initialize
        # for every state, f[v] = g(s, v) + h(v, g)
        unvisited_states = set()  # OPEN set. Nodes not in this set is in CLOSE set
        priority_q = []
        shortest_path = []
        shortest_path_len = 0
        g = {}  # cost-to-come from start to a state
        f = {}  # cost-to-come + heuristic cost-to-go
        prev_state_dict = {}  # used to extract shortest path

        for state in self._env.all_states:
            g[state] = float("inf")
            f[state] = float("inf")
            unvisited_states.add(state)  # All states are unvisited

        g[start] = 0
        f[start] = self._heuristic_func(start, goal)  # cost from source to goal = g(s, s) + h(s, g) = 0 + h(s, g)
        heapq.heappush(priority_q, (f[start], start))

        print(start, goal)
        # run algorithm
        path_exist = False
        while len(priority_q) > 0:
            # pop the best state found so far.
            _, best_state = heapq.heappop(priority_q)
            if best_state not in unvisited_states:
                print("Here!!")
                continue

            # path to goal is found
            if best_state == goal:
                path_exist = True
                break

            # Find possible transitions from best_state, and add them to queue ranked by heuristics.
            unvisited_states.remove(best_state)
            available_actions = self._env.get_available_actions(best_state)
            # print(best_state, available_actions)
            for action in available_actions:
                new_state, cost, _, _, _ = self._env.state_transition_func(best_state, action)
                if new_state in unvisited_states:
                    g_new_state = g[best_state] + cost  # cost-to-come
                    h_new_state = self._heuristic_func(new_state, goal)  # cost-to-go
                    print(new_state, h_new_state)
                    if g_new_state < g[new_state]:
                        g[new_state] = g_new_state
                        f[new_state] = g_new_state + h_new_state
                        heapq.heappush(priority_q, (f[new_state], new_state))
                        prev_state_dict[new_state] = best_state

        if path_exist:
            # extract shortest path:
            shortest_path.insert(0, goal)
            state = goal
            prev_v = prev_state_dict[state]
            while prev_v != -1 and prev_v != start:
                shortest_path.insert(0, prev_v)
                prev_v = prev_state_dict[prev_v]

            shortest_path.insert(0, start)
            shortest_path_len = f[goal]
            return (True, shortest_path, shortest_path_len)
        else:
            return (False, None, None)
