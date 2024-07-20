from typing import Any, Callable

from sklearn.neighbors import NearestNeighbors
import numpy as np
import networkx as nx

from robotics_algorithm.env.base_env import DeterministicEnv


class ProbabilisticRoadmap(object):
    def __init__(
        self,
        env: DeterministicEnv,
        sample_func: Callable,
        state_col_check_func: Callable,
        edge_col_check_func: Callable,
        num_of_samples: int,
        num_neighbors: int,
    ):
        """_summary_

        Args:
            env (ContinuousEnv): the env
            sample_func (Callable): the sampling function, returns a random point in space.
            state_col_check_func (Callable): a function that takes in a random point and returns whether it is in
                collision.
            edge_col_check_func (Callable): a function that takes in two points and returns whether there is a
                collision-free simple path between them.
            num_of_samples (int): maximum of samples drawn during planning.
            num_neighbors (int): number of closest neighbors to attempt connection.
        """
        self.env = env
        self._sample_func = sample_func
        self._state_col_check_func = state_col_check_func
        self._edge_col_check_func = edge_col_check_func
        self.num_of_samples = num_of_samples
        self.num_neighbors = num_neighbors

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
            neighbors = self.get_nearest_neighbors(self.all_samples, v, self.num_neighbors)
            # print("neighbours {}".format(neighbours))
            for neighbor in neighbors:
                can_link, length = self._edge_col_check_func(self.env, v, neighbor)
                if can_link:
                    self.roadmap.add_edge(tuple(v), tuple(neighbor), weight=length)

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

    def get_path(self, start: list, goal: list) -> tuple[bool, list[tuple], float]:
        """
        Online computation of path.

        Args:
            start (tuple): the start state.
            goal (tuple): the goal state.

        Returns:
            success (boolean): return true if a path is found, return false otherwise
            shortest_path (list[tuple]): a list of vertices if shortest path is found
            shortest_path_len (float): the length of shortest path if found.
        """
        start = tuple(start)
        goal = tuple(goal)

        source_neighbors = self.get_nearest_neighbors(self.all_samples, start, self.num_neighbors)
        for neighbor in source_neighbors:
            can_link, length = self._edge_col_check_func(self.env, start, neighbor)
            if can_link:
                self.roadmap.add_edge(start, tuple(neighbor), weight=length)
                break
        else:
            print("can't connect start to roadmap!!!")
            return False, None, None

        goal_neighbors = self.get_nearest_neighbors(self.all_samples, goal, self.num_neighbors)
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

    def run(self, start: tuple, goal: tuple) -> tuple[bool, list[tuple], float]:
        """
        Run algorithm.

        Args:
            start (tuple): the start state.
            goal (tuple): the goal state.

        Returns:
            success (boolean): return true if a path is found, return false otherwise.
            shortest_path (list[tuple]): a list of vertices if shortest path is found.
            shortest_path_len (float): the length of shortest path if found.
        """
        self.compute_roadmap()
        return self.get_path(start, goal)

    def get_roadmap(self):
        return self.roadmap
