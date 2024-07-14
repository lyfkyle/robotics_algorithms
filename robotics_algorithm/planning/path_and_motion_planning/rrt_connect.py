from collections import defaultdict
import random

from sklearn.neighbors import NearestNeighbors
import numpy as np
import networkx as nx


class RRT(object):
    TRAPPED = 0
    REACHED = 1

    def __init__(self, env, sample_func, vertex_expand_func, num_of_samples=1):
        self.num_of_samples = num_of_samples
        self.env = env
        self._sample_func = sample_func
        self._vertex_expand_func = vertex_expand_func
        self.tree = nx.Graph()

        self.goal_bias = 0.2

    def initialize_tree(self, start_state):
        self.tree.add_node(start_state)

    def run(self, start, goal):
        start = tuple(start)
        goal = tuple(goal)

        self.initialize_tree(start)

        path_exist = False
        for i in range(self.num_of_samples):
            if i % 100 == 0:
                print("RRT/run, iteration {}".format(i))

            if np.random.uniform() > self.goal_bias:
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

    def extend(self, v_target):
        """Extend towards v_target."""

        # RRT finds the nearest node in tree to v_target
        all_nodes = list(self.tree.nodes)
        nearest_neighbors = self.get_nearest_neighbors(all_nodes, np.array(v_target).reshape(1, 2))
        v_cur = tuple(nearest_neighbors[0])

        # expand towards v_target
        v_new, path_len = self._vertex_expand_func(self.env, v_cur, v_target)
        v_new = tuple(v_new)
        # print(v_cur, v_target, v_new)

        if v_new != v_cur:
            self.tree.add_edge(v_cur, v_new, weight=path_len)

        if v_new == v_target:
            return RRT.REACHED, v_new
        else:
            return RRT.TRAPPED, v_new

    def get_nearest_neighbors(self, all_vertices: np.ndarray, v: np.ndarray, n_neighbors: int = 1) -> list[tuple]:
        """
        return the closest neighbors of v in all_vertices.

        Args:
            all_vertices: a list of vertices, must be a 2D numpy array
            v: , the target vertex, must be a 2D numpy array

        Returns:
            (list[tuple]): a list of nearby vertices
        """
        n_neighbors = min(n_neighbors, len(all_vertices))
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree").fit(all_vertices)
        distances, indices = nbrs.kneighbors(v)
        # print("indices {}".format(indices))
        nbr_vertices = np.take(np.array(all_vertices), indices.ravel(), axis=0).tolist()
        nbr_vertices = [tuple(v) for v in nbr_vertices]
        return nbr_vertices

    def get_tree(self):
        return self.tree


class RRTConnect(object):
    def __init__(self, env, sample_func, vertex_expand_func, num_of_samples):
        self.num_of_samples = num_of_samples
        self.start_rrt = RRT(env, None, vertex_expand_func)
        self.goal_rrt = RRT(env, None, vertex_expand_func)
        self.tree = nx.Graph()

        self.env = env
        self._sample_func = sample_func

    def run(self, start, goal):
        start = tuple(start)
        goal = tuple(goal)

        # Initialize two trees, one at start, and the other at goal.
        self.start_rrt.initialize_tree(start)
        self.goal_rrt.initialize_tree(goal)

        # Iteratively expand each tree.
        rrt1 = self.start_rrt
        rrt2 = self.goal_rrt
        for i in range(self.num_of_samples):
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
        nodes = set(self.start_rrt.tree.nodes())
        nodes.update(set(self.goal_rrt.tree.nodes()))
        combined.add_nodes_from(nodes)

        return combined
