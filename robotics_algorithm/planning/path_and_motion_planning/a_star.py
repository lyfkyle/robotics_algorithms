from typing import Callable, Any
import heapq

from robotics_algorithm.env.base_env import DiscretePlanningEnv


class AStar(object):
    def __init__(self):
        pass

    def run(self, env: DiscretePlanningEnv, start: Any, goal: Any, heuristic_func: Callable):
        """Run Astar.

        Args:
            env (DiscretePlanningEnv): A planning env
            start (Any): the start state
            goal (Any): the goal state
            heuristic_func (Callable): a function to return estimated cost-to-go from a state to goal

        Returns:
            res, (bool), return true if a path is found, return false otherwise
            shortest_path, a list of state if shortest path is found
            shortest_path_len, the length of shortest path if found.
        """

        # initialize
        # for every state, f[v] = g(s, v) + h(v, g)
        unvisited_states = set()  # OPEN set. Nodes not in this set is in CLOSE set
        priority_q = []
        shortest_path = []
        shortest_path_len = 0
        g = {}  # cost-to-come from start to a state
        f = {}  # cost-to-come + heuristic cost-to-go
        prev = {}  # used to extract shortest path

        for state in env.all_states:
            g[state] = float("inf")
            f[state] = float("inf")
            unvisited_states.add(state)  # All states are unvisited

        g[start] = 0
        f[start] = heuristic_func(start, goal)  # cost from source to goal = g(s, s) + h(s, g) = 0 + h(s, g)
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

            # min_dist = float("inf")
            # for state in unvisited_states:
            #     if dist[state] < min_dist:
            #         min_dist = dist[state]
            #         best_state = state

            # print("min_dist: {}".format(min_dist))
            # there is no path
            # if best_dist_to_come == float("inf"):
            #     break

            # path to goal is found
            if best_state == goal:
                path_exist = True
                break

            # Find possible transitions from best_state, and add them to queue ranked by heuristics.
            unvisited_states.remove(best_state)
            available_actions = env.get_available_actions(best_state)
            # print(best_state, available_actions)
            for action in available_actions:
                new_state, cost = env.state_transition_func(best_state, action)
                if new_state in unvisited_states:
                    g_new_state = g[best_state] + cost  # cost-to-come
                    h_new_state = heuristic_func(new_state, goal)  # cost-to-go
                    print(new_state, h_new_state)
                    if g_new_state < g[new_state]:
                        g[new_state] = g_new_state
                        f[new_state] = g_new_state + h_new_state
                        heapq.heappush(priority_q, (f[new_state], new_state))
                        prev[new_state] = best_state

            # for state, edge_length in env[best_state].items():
            #     if state in unvisited_states:
            #         g_new_state = g[best_state] + edge_length
            #         h_new_state = heuristic_func(state, goal)
            #         if g_new_state < g[state]:
            #             g[state] = g_new_state
            #             dist[state] = g_new_state + h_new_state
            #             prev[state] = best_state

        if path_exist:
            # extract shortest path:
            shortest_path.insert(0, goal)
            state = goal
            prev_v = prev[state]
            while prev_v != -1 and prev_v != start:
                shortest_path.insert(0, prev_v)
                prev_v = prev[prev_v]

            shortest_path.insert(0, start)
            shortest_path_len = f[goal]
            return (True, shortest_path, shortest_path_len)
        else:
            return (False, None, None)
