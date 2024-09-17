from collections import defaultdict
import math
from typing import Callable

from sklearn.neighbors import NearestNeighbors
import numpy as np
import networkx as nx

from robotics_algorithm.env.base_env import BaseEnv, SpaceType, EnvType


class RRTStar(object):
    TRAPPED = 0
    REACHED = 1

    def __init__(
        self,
        env: BaseEnv,
        sample_func: Callable,
        vertex_expand_func: Callable,
        edge_col_check_func: Callable,
        distance_func: Callable,
        num_of_samples: int,
    ):
        """Constructor.

        Args:
            env (BaseEnv): env
            sample_func (Callable): a function to obtain a state sample from env.
            vertex_expand_func (Callable): a function to expand one state towards another.
            edge_col_check_func (Callable): a function to check whether two states are connectable
            distance_func (Callable): a function to calculate distance between two states.
            num_of_samples (int): maximum of number of samples. Defaults to 1.
        """
        assert env.state_space.type == SpaceType.CONTINUOUS.value
        assert env.action_space.type == SpaceType.CONTINUOUS.value
        assert env.state_transition_type == EnvType.DETERMINISTIC.value
        assert env.observability == EnvType.FULLY_OBSERVABLE.value

        self.env = env
        self._sample_func = sample_func
        self._vertex_expand_func = vertex_expand_func
        self._edge_col_check_func = edge_col_check_func
        self._distance_func = distance_func
        self.tree = nx.Graph()
        self.g = {}
        self.num_of_samples = num_of_samples

        self.goal_bias = 0.2

    def initialize_tree(self, root):
        self.g[root] = 0  # cost_to_come to itself is 0.
        self.tree.add_node(root)

    def run(self, start: tuple, goal: tuple) -> tuple[bool, list[tuple], float]:
        """
        Run planner.

        Args:
            start (tuple): the start state.
            goal (tuple): the goal state.

        Returns:
            success (boolean): return true if a path is found, return false otherwise.
            shortest_path (list[tuple]): a list of vertices if shortest path is found.
            shortest_path_len (float): the length of shortest path if found.
        """
        start = tuple(start)
        goal = tuple(goal)

        self.initialize_tree(start)

        for i in range(self.num_of_samples):
            if i % 100 == 0:
                print("RRTStar/run, iteration {}".format(i))

            if np.random.uniform() > self.goal_bias:
                v_target = self._sample_func(self.env)
            else:
                v_target = goal

            self.extend(v_target)

        if goal in self.tree:
            path = nx.shortest_path(self.tree, start, goal, weight="weight")
            path_len = nx.shortest_path_length(self.tree, start, goal, weight="weight")
            print(path, path_len)
            return True, path, path_len
        else:
            return False, None, None

    def extend(self, v_target: tuple):
        """
        Extend towards v_goal one step.
            1. find nearest_neighbor
            2. extend from nearest_neighbor to v_goal -> v_new
            3. find best_neighbor around v_new
            4. connect best_neighbor to v_new
            5. rewire neighbors to v_new

        Args:
            v_target (tuple): the target vertex to extend to.

        """
        # Like A-Star, the notion here is dist[v] = g(s, v) + h(v, g). (total cost = cost to come + cost to go)
        v_target = tuple(v_target)

        # RRT finds the nearest node in tree to v_target
        all_nodes = list(self.tree.nodes)
        nearest_neighbors = self.get_nearest_neighbors(all_nodes, v_target)
        v_cur = tuple(nearest_neighbors[0])

        # expand towards v_target
        v_new, path_len = self._vertex_expand_func(self.env, v_cur, v_target)
        v_new = tuple(v_new)
        # print(v_cur, v_target, v_new)

        # if expansion happens.
        if v_new != v_cur:
            raw_neighbors = self.get_nearest_neighbors(all_nodes, v_new, n_neighbors=10)

            # filter out neighbors that are not connectable
            neighbors = [n for n in raw_neighbors if self._edge_col_check_func(self.env, n, v_new)[0]]

            # shortcut. Instead of connecting v_new to v_cur, connect to neighbor that has best cost-to-come
            best_cost_to_come = self.g[v_cur] + path_len
            best_neighbor = v_cur
            best_dist = path_len
            for neighbor in neighbors:
                assert self._edge_col_check_func(self.env, v_new, neighbor)
                dist = self._distance_func(self.env, neighbor, v_new)
                if self.g[neighbor] + dist < best_cost_to_come:
                    best_cost_to_come = self.g[neighbor] + dist
                    best_neighbor = neighbor
                    best_dist = dist

            self.tree.add_edge(best_neighbor, v_new, weight=best_dist)
            self.g[v_new] = best_cost_to_come

            # rewire nearby neighbors.
            for neighbor in neighbors:
                dist_v_neighbour = self._distance_func(self.env, v_new, neighbor)
                if self.g[v_new] + dist_v_neighbour < self.g[neighbor]:
                    self.g[neighbor] = self.g[v_new] + dist_v_neighbour
                    self.tree.remove_node(neighbor)  # remove old edge by removing the node itself.
                    self.tree.add_edge(v_new, neighbor, weight=dist_v_neighbour)

        if v_new == v_target:
            return RRTStar.REACHED, v_new
        else:
            return RRTStar.TRAPPED, v_new

    def get_nearest_neighbors(self, all_vertices: list[tuple], v: tuple, n_neighbors: int = 1) -> list[tuple]:
        """
        return the closest neighbors of v in all_vertices.

        Args:
            all_vertices (list[tuple]): a list of vertices
            v (tuple): the target vertex.
            n_neighbors (int): number of nearby neighbors.

        Returns:
            (list[tuple]): a list of nearby vertices
        """
        n_neighbors = min(n_neighbors, len(all_vertices))

        all_vertices = np.array(all_vertices)
        v = np.array(v).reshape(1, -1)

        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree").fit(all_vertices)
        distances, indices = nbrs.kneighbors(v)
        # print("indices {}".format(indices))
        nbr_vertices = np.take(np.array(all_vertices), indices.ravel(), axis=0).tolist()
        nbr_vertices = [tuple(v) for v in nbr_vertices]
        return nbr_vertices

    def get_tree(self):
        return self.tree


# TODO
class RRTStarConnect(object):
    def __init__(self, num_of_samples):
        self.num_of_samples = num_of_samples
        self.start_rrt = RRTStar()
        self.goal_rrt = RRTStar()
        self.adj_list = defaultdict(dict)

    def run(self, source, goal, sample_vertex_func, expand_func, delta, distance_func, check_link):
        self.start_rrt.initialize_tree(source)
        self.goal_rrt.initialize_tree(goal)

        rrt1 = self.start_rrt
        rrt2 = self.goal_rrt
        reached = False
        for i in range(self.num_of_samples):
            if i % 100 == 0:
                print("RRTConnect/run, iteration {}".format(i))

            v_target = sample_vertex_func()
            res, v = rrt1.extend(v_target, expand_func, delta, distance_func, check_link)
            if res != RRTStar.TRAPPED:
                res = rrt2.connect(v, expand_func, delta, distance_func, check_link)
                if res == RRTStar.REACHED:
                    reached = True

            rrt1, rrt2 = rrt2, rrt1

        if reached:
            return self.get_path(source, goal)

        return False, None, None

    def get_path(self, source, goal):
        adj_list = self.get_tree()

        # call dijkstra to find the path
        path_exist, shortest_path, path_len = self._path_finder.run(adj_list, source, goal)
        if not path_exist:
            print("RRTStarConnect/get_path: No path is found, this should not happen!!!")
            return False, None, None
        else:
            return True, shortest_path, path_len

    def get_tree(self):
        # merge adj_list from two rrt
        adj_list = self.start_rrt.adj_list.copy()
        for key in self.goal_rrt.adj_list:
            for key1, val1 in self.goal_rrt.adj_list[key].items():
                adj_list[key][key1] = val1
        self.adj_list = adj_list

        return self.adj_list
