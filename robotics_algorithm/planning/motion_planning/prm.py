from typing import Any, Callable

from sklearn.neighbors import NearestNeighbors
import numpy as np
import networkx as nx

from robotics_algorithm.env.base_env import BaseEnv, SpaceType, EnvType


class ProbabilisticRoadmap:
    def __init__(
        self,
        env: BaseEnv,
        sample_func: Callable,
        state_col_check_func: Callable,
        edge_col_check_func: Callable,
        num_of_samples: int,
        num_neighbors: int,
    ):
        """_summary_

        Args:
            env (BaseEnv): the env
            sample_func (Callable): the sampling function, returns a random point in space.
            state_col_check_func (Callable): a function that takes in a random point and returns whether it is in
                collision.
            edge_col_check_func (Callable): a function that takes in two points and returns whether there is a
                collision-free simple path between them.
            num_of_samples (int): maximum of samples drawn during planning.
            num_neighbors (int): number of closest neighbors to attempt connection.
        """
        assert env.state_space.type == SpaceType.CONTINUOUS.value
        assert env.action_space.type == SpaceType.CONTINUOUS.value
        assert env.state_transition_type == EnvType.DETERMINISTIC.value
        assert env.observability == EnvType.FULLY_OBSERVABLE.value

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
                sample = self._sample_func(self.env)
                collision_free = self._state_col_check_func(self.env, sample)

            self.all_samples.append(sample)

        print("PRM/compute_roadmap: finished adding {} of vertices".format(self.num_of_samples))
        # print(self.V)

        for sample in self.all_samples:
            neighbors = self.get_nearest_neighbors(self.all_samples, sample, self.num_neighbors)
            # print("neighbours {}".format(neighbours))
            for neighbor in neighbors:
                can_link, length = self._edge_col_check_func(self.env, sample, neighbor)
                if can_link:
                    self.roadmap.add_edge(tuple(sample.tolist()), tuple(neighbor.tolist()), weight=length)

    def get_nearest_neighbors(self, all_vertices: np.ndarray[tuple], v: tuple, n_neighbors: int = 1) -> np.ndarray[tuple]:
        """
        return the closest neighbors of v in all_vertices.

        Args:
            all_vertices (np.ndarray[tuple]): a np.ndarray of vertices
            v (tuple): the target vertex.
            n_neighbors (int): number of nearby neighbors.

        Returns:
            (np.ndarray[tuple]): a np.ndarray of nearby vertices
        """
        n_neighbors = min(n_neighbors, len(all_vertices))

        all_vertices = np.array(all_vertices)
        v = np.array(v).reshape(1, -1)

        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree").fit(all_vertices)
        distances, indices = nbrs.kneighbors(v)
        # print("indices {}".format(indices))
        nbr_vertices = np.take(np.array(all_vertices), indices.ravel(), axis=0)
        return nbr_vertices

    def get_path(self, start: np.ndarray, goal: np.ndarray) -> tuple[bool, np.ndarray[tuple], float]:
        """
        Online computation of path.

        Args:
            start (tuple): the start state.
            goal (tuple): the goal state.

        Returns:
            success (boolean): return true if a path is found, return false otherwise
            shortest_path (np.ndarray[tuple]): a np.ndarray of vertices if shortest path is found
            shortest_path_len (float): the length of shortest path if found.
        """
        start_tup = tuple(start.tolist())
        goal_tup = tuple(goal.tolist())

        source_neighbors = self.get_nearest_neighbors(self.all_samples, start, self.num_neighbors)
        for neighbor in source_neighbors:
            can_link, length = self._edge_col_check_func(self.env, start, neighbor)
            if can_link:
                self.roadmap.add_edge(start_tup, tuple(neighbor.tolist()), weight=length)
                break
        else:
            print("can't connect start to roadmap!!!")
            return False, None, None

        goal_neighbors = self.get_nearest_neighbors(self.all_samples, goal, self.num_neighbors)
        for neighbor in goal_neighbors:
            can_link, length = self._edge_col_check_func(self.env, goal, neighbor)
            if can_link:
                self.roadmap.add_edge(goal_tup, tuple(neighbor.tolist()), weight=length)
                break
        else:
            print("can't connect goal to roadmap!!!")
            return False, None, None

        # path_exist, shortest_path, path_len = self._path_finder.run(self.adj_list, start, goal)
        path = nx.shortest_path(self.roadmap, start_tup, goal_tup, weight="weight")
        path_len = nx.shortest_path_length(self.roadmap, start_tup, goal_tup, weight="weight")
        print(path, path_len)
        return True, path, path_len

    def run(self, start: tuple, goal: tuple) -> tuple[bool, np.ndarray[tuple], float]:
        """
        Run algorithm.

        Args:
            start (tuple): the start state.
            goal (tuple): the goal state.

        Returns:
            success (boolean): return true if a path is found, return false otherwise.
            shortest_path (np.ndarray[tuple]): a np.ndarray of vertices if shortest path is found.
            shortest_path_len (float): the length of shortest path if found.
        """
        self.compute_roadmap()
        return self.get_path(start, goal)

    def get_roadmap(self):
        return self.roadmap
