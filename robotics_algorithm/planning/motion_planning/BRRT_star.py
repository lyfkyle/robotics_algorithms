from typing import Callable

import networkx as nx
import numpy as np

from robotics_algorithm.env.base_env import BaseEnv, EnvType, SpaceType
from robotics_algorithm.planning.motion_planning.rrt_star import RRTStar


class BRRTStar:
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
            edge_col_check_func (Callable): a function to check whether two states are connectable.
            distance_func (Callable): a function to calculate distance between two states.
            num_of_samples (int): maximum of number of samples. Defaults to 1.
        """
        assert env.state_space.type == SpaceType.CONTINUOUS.value
        assert env.action_space.type == SpaceType.CONTINUOUS.value
        assert env.state_transition_type == EnvType.DETERMINISTIC.value
        assert env.observability == EnvType.FULLY_OBSERVABLE.value

        self.env = env
        self.num_of_samples = num_of_samples
        self.start_rrt = RRTStar(env, None, vertex_expand_func, edge_col_check_func, distance_func, num_of_samples)
        self.goal_rrt = RRTStar(env, None, vertex_expand_func, edge_col_check_func, distance_func, num_of_samples)
        self.tree = nx.Graph()
        self._sample_func = sample_func
        self.cost_history = []  # best cost-to-goal found so far, indexed by iteration

    def run(self, start: np.ndarray, goal: np.ndarray) -> tuple[bool, np.ndarray[tuple], float]:
        """
        Run planner.

        Args:
            start (np.ndarray): the start state.
            goal (np.ndarray): the goal state.

        Returns:
            success (boolean): return true if a path is found, return false otherwise.
            shortest_path (np.ndarray[np.ndarray]): a np.ndarray of vertices if shortest path is found.
            shortest_path_len (float): the length of shortest path if found.
        """
        start = tuple(start.tolist())
        goal = tuple(goal.tolist())

        # Initialize two trees, one at start, and the other at goal.
        self.start_rrt.initialize_tree(start)
        self.goal_rrt.initialize_tree(goal)

        # Iteratively expand each tree.
        rrt1 = self.start_rrt
        rrt2 = self.goal_rrt
        best_cost = float('inf')
        for i in range(self.num_of_samples):
            if i % 100 == 0:
                print('B-RRT*/run, iteration {}'.format(i))

            v_target = tuple(self._sample_func(self.env).tolist())
            res, v_new = rrt1.extend(v_target)
            if res != RRTStar.TRAPPED:
                # ! No need to perform rewire stage according to original paper
                res, _ = rrt2.extend(v_new, rewire=False)
                if res == RRTStar.REACHED:
                    path_cost = rrt1.g[v_new] + rrt2.g[v_new]
                    if path_cost < best_cost:
                        best_cost = path_cost

            self.cost_history.append(best_cost)
            rrt1, rrt2 = rrt2, rrt1

        return self._get_path(start, goal)

    def _get_path(self, start: tuple, goal: tuple):
        self.combined_tree = self.get_tree()

        try:
            path = nx.shortest_path(self.combined_tree, start, goal, weight='weight')
            path_len = nx.shortest_path_length(self.combined_tree, start, goal, weight='weight')
            return True, path, path_len
        except nx.NetworkXNoPath:
            return False, None, None

    def get_tree(self) -> nx.Graph:
        """Retrieve the current planning tree.

        Returns:
            (nx.Graph): the current planning tree.
        """
        combined = nx.Graph()
        combined.add_edges_from(list(self.start_rrt.tree.edges(data=True)) + list(self.goal_rrt.tree.edges(data=True)))
        nodes = set(self.start_rrt.tree.nodes())
        nodes.update(set(self.goal_rrt.tree.nodes()))
        combined.add_nodes_from(nodes)

        return combined
