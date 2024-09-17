from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import numpy as np
import networkx as nx
import random


class RRT(object):
    TRAPPED = 0
    REACHED = 1

    def __init__(self, env, sample_func, vertex_expand_func):
        self.env = env
        self._sample_func = sample_func
        self._vertex_expand_func = vertex_expand_func
        self.tree = nx.Graph()

        self.goal_bias = 0.2

    def initialize_tree(self, start_state):
        self.tree.add_node(tuple(start_state))

    def extend(self, v_target):
        """Extend towards v_target."""

        # RRT finds the nearest node in tree to v_target
        cur_node = list(self.tree.nodes)
        nearest_neighbors = self.get_nearest_neighbour(cur_node, np.array(v_target).reshape(1, 2))
        v_cur = tuple(nearest_neighbors[0])

        # Expand towards v_target
        v_new, dist = self._vertex_expand_func(self.env, v_cur, v_target)
        print(v_cur, v_target, v_new)
        if tuple(v_new) != tuple(v_cur):
            self.tree.add_edge(tuple(v_cur), tuple(v_new), weight=dist)

        if tuple(v_new) == tuple(v_target):
            return RRT.REACHED, tuple(v_new)
        else:
            return RRT.TRAPPED, tuple(v_new)

    def get_nearest_neighbour(self, V, v):
        """
        return the closest neighbours of v in V
        @param V, a list of vertices, must be a 2D numpy array
        @param v, the target vertice, must be a 2D numpy array
        @return, list, the nearest neighbours
        """

        nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(V)
        distances, indices = nbrs.kneighbors(v)
        # print("indices {}".format(indices))
        return np.take(np.array(V), indices.ravel(), axis=0).tolist()

    def run(self, start, goal):
        start = tuple(start)
        goal = tuple(goal)

        self.initialize_tree(start)

        path_exist = False
        for i in range(self.N):
            if i % 100 == 0:
                print("RRTConnect/run, iteration {}".format(i))

            if random.uniform(0, 1) > self.goal_bias:
                v_target = self._sample_func(self.env)
            else:
                v_target = goal

            self.extend(v_target)

            if goal in self.tree:
                path_exist = True
                break

        if path_exist:
            path = nx.shortest_path(self.tree, start, goal, weight="weight")
            path_len = nx.shortest_path_length(self.tree, start, goal, weight="weight")
            print(path, path_len)
            return True, path, path_len
        else:
            return False, None, None


class RRTConnect(object):
    def __init__(self, env, sample_func, vertex_expand_func, number_of_samples):
        self.N = number_of_samples
        self.start_rrt = RRT(env, None, vertex_expand_func)
        self.goal_rrt = RRT(env, None, vertex_expand_func)
        self.tree = nx.Graph()

        self.env = env
        self._sample_func = sample_func

    def run(self, start, goal):
        # Initialize two trees, one at start, and the other at goal.
        self.start_rrt.initialize_tree(start)
        self.goal_rrt.initialize_tree(goal)

        # Iteratively expand each tree.
        rrt1 = self.start_rrt
        rrt2 = self.goal_rrt
        for i in range(self.N):
            if i % 100 == 0:
                print("RRTConnect/run, iteration {}".format(i))

            v_target = self._sample_func(self.env)
            res, v = rrt1.extend(v_target)
            if res != RRT.TRAPPED:
                res, _ = rrt2.extend(v)
                if res == RRT.REACHED:
                    return self.get_path(start, goal)

            rrt1, rrt2 = rrt2, rrt1

        return False, None, None

    def get_path(self, start, goal):
        self.combined_tree = self.get_tree()
        start = tuple(start)
        goal = tuple(goal)

        path = nx.shortest_path(self.combined_tree, start, goal, weight="weight")
        path_len = nx.shortest_path_length(self.combined_tree, start, goal, weight="weight")
        print(path, path_len)
        return True, path, path_len

    def get_tree(self):
        combined = nx.Graph()
        combined.add_edges_from(list(self.start_rrt.tree.edges(data=True)) + list(self.goal_rrt.tree.edges(data=True)))
        combined.add_nodes_from(list(self.start_rrt.tree.nodes(data=True)) + list(self.goal_rrt.tree.nodes(data=True)))

        return combined
