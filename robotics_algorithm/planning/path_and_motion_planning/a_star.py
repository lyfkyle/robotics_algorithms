from typing import Callable
import heapq

from robotics_algorithm.env.base_env import DeterministicEnv


class AStar:
    def __init__(self, env: DeterministicEnv, heuristic_func: Callable):
        """Constructor.

        Args:
            env (DeterministicEnv): A planning env.
            heuristic_func (Callable): a function to return estimated cost-to-go from a state to goal.
        """
        self.env = env
        self._heuristic_func = heuristic_func

    def run(self, start: tuple, goal: tuple) -> tuple[bool, list[tuple], float]:
        """Run algorithm.

        Args:
            start (tuple): the start state
            goal (tuple): the goal state

        Returns:
            res (bool): return true if a path is found, return false otherwise.
            shortest_path (list[tuple]): a list of state if shortest path is found.
            shortest_path_len (float): the length of shortest path if found.
        """
        # Astar gist: for every state, f[v] = g(s, v) + h(v, g)

        # initialize

        unvisited_states = set()  # OPEN set. Nodes not in this set is in CLOSE set
        priority_q = []
        shortest_path = []
        shortest_path_len = 0
        g = {}  # cost-to-come from start to a state
        f = {}  # cost-to-come + heuristic cost-to-go
        prev_state_dict = {}  # used to extract shortest path

        for state in self.env.state_space:
            g[state] = float("inf")
            f[state] = float("inf")
            unvisited_states.add(state)  # All states are unvisited

        g[start] = 0  # distance to source is 0
        f[start] = self._heuristic_func(start, goal)  # cost from source to goal = g(s, s) + h(s, g) = 0 + h(s, g)
        heapq.heappush(priority_q, (f[start], start))

        # run algorithm
        path_exist = False
        while len(priority_q) > 0:
            # pop the best state found so far.
            _, best_state = heapq.heappop(priority_q)

            # filter out state visited before. This may happen if multiple path leads to the same state with
            # different values.
            if best_state not in unvisited_states:
                continue

            # path to goal is found
            if best_state == goal:
                path_exist = True
                break

            # Find possible transitions from best_state, and add them to queue ranked by heuristics.
            unvisited_states.remove(best_state)
            available_actions = self.env.get_available_actions(best_state)
            # print(best_state, available_actions)
            for action in available_actions:
                new_state, cost, term, _, info = self.env.state_transition_func(best_state, action)

                # skip actions that result in failures
                if term and not info["success"]:
                    continue

                if new_state in unvisited_states:
                    g_new_state = g[best_state] + cost  # cost-to-come
                    h_new_state = self._heuristic_func(new_state, goal)  # cost-to-go
                    # print(new_state, h_new_state)
                    if g_new_state < g[new_state]:
                        g[new_state] = g_new_state
                        f[new_state] = g_new_state + h_new_state
                        # NOTE: here we do not need to remove the the previous stored new_state with old best-to-come
                        #       value because of new_state will be marked as visited when first poped with the best
                        #       cost-to-come value.
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
            shortest_path_len = g[goal]
            return (True, shortest_path, shortest_path_len)
        else:
            return (False, None, None)
