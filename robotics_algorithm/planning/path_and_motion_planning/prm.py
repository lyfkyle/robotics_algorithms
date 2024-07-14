from typing import Any

from sklearn.neighbors import NearestNeighbors
import numpy as np
import networkx as nx


class ProbabilisticRoadmap(object):
    def __init__(self, env, sample_func, state_col_check_func, edge_col_check_func, num_of_samples: int, K):
        """_summary_

        Args:
            self._sample_func, the sampling function, returns a random point in space.
            state_col_check_func, a function that takes in a random point and returns whether it is in collision.
            edge_col_check_func, a function that takes in two points and returns whether there is a collision-free
                simple path between them.
            num_of_samples (int): _description_
            K (int): number of closest neighbors to attempt connection.
        """
        self.env = env
        self._sample_func = sample_func
        self._state_col_check_func = state_col_check_func
        self._edge_col_check_func = edge_col_check_func
        self.num_of_samples = num_of_samples
        self.K = K

    def compute_roadmap(self):
        """
        Offline computation of roadmap by sampling points.
        """
        self.all_samples = []
        self.roadmap = nx.Graph()
        while len(self.all_samples) < self.num_of_samples:
            collision_free = False
            while not collision_free:
                v = self._sample_func(self.env)
                collision_free = self._state_col_check_func(self.env, v)

            self.all_samples.append(v)

        print("PRM/compute_roadmap: finished adding {} of vertices".format(self.num_of_samples))
        # print(self.V)

        for v in self.all_samples:
            neighbors = self.get_K_closest_neighbors(self.all_samples, v, self.K)
            # print("neighbours {}".format(neighbours))
            for neighbor in neighbors:
                can_link, length = self._edge_col_check_func(self.env, v, neighbor)
                if can_link:
                    self.roadmap.add_edge(tuple(v), tuple(neighbor), weight=length)

    def get_K_closest_neighbors(self, V, v, K):
        """
        return K closest neighbors of v in V
        @param V, a list of vertices, must be a 2D numpy array
        @param v, the target vertices, must be a 2D numpy array
        @param K, number of neighbors to get
        @return, list, the list of nearest neighbours
        """

        nbrs = NearestNeighbors(n_neighbors=K, algorithm="ball_tree").fit(np.array(V))
        distances, indices = nbrs.kneighbors(np.array(v).reshape(1, 2))
        # print("indices {}".format(indices))
        return np.take(np.array(V), indices.ravel(), axis=0).tolist()

    def get_path(self, start: list, goal: list) -> tuple[bool, list[Any], float]:
        """
        Online computation of path.

        Args:
            start (list): the start state.
            goal (list): the goal state.

        Returns:
            success (boolean): return true if a path is found, return false otherwise
            shortest_path: a list of vertices if shortest path is found
            shortest_path_len: the length of shortest path if found.
        """
        start = tuple(start)
        goal = tuple(goal)

        source_neighbors = self.get_K_closest_neighbors(self.all_samples, start, self.K)
        for neighbor in source_neighbors:
            can_link, length = self._edge_col_check_func(self.env, start, neighbor)
            if can_link:
                self.roadmap.add_edge(start, tuple(neighbor), weight=length)
                break
        else:
            print("can't connect start to roadmap!!!")
            return False, None, None

        goal_neighbors = self.get_K_closest_neighbors(self.all_samples, goal, self.K)
        for neighbor in goal_neighbors:
            can_link, length = self._edge_col_check_func(self.env, goal, neighbor)
            if can_link:
                self.roadmap.add_edge(goal, tuple(neighbor), weight=length)
                break
        else:
            print("can't connect goal to roadmap!!!")
            return False, None, None

        # path_exist, shortest_path, path_len = self._path_finder.run(self.adj_list, start, goal)
        path = nx.shortest_path(self.roadmap, start, goal, weight="weight")
        path_len = nx.shortest_path_length(self.roadmap, start, goal, weight="weight")
        print(path, path_len)
        return True, path, path_len

    def run(self, start, goal):
        self.compute_roadmap()
        return self.get_path(start, goal)

    def get_roadmap(self):
        return self.roadmap
