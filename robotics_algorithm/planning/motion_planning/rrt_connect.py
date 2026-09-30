from typing import Callable

import networkx as nx
import numpy as np

from robotics_algorithm.env.base_env import BaseEnv, EnvType, SpaceType
from robotics_algorithm.planning.motion_planning.rrt import RRT


class RRTConnect:
    def __init__(self, env: BaseEnv, sample_func: Callable, vertex_expand_func: Callable, num_of_samples: int):
        """Constructor.

        Args:
            env (BaseEnv): env
            sample_func (Callable): a function to obtain a state sample from env.
            vertex_expand_func (Callable): a function to expand one state towards another.
            num_of_samples (int): maximum of number of samples. Defaults to 1.
        """
        assert env.state_space.type == SpaceType.CONTINUOUS.value
        assert env.action_space.type == SpaceType.CONTINUOUS.value
        assert env.state_transition_type == EnvType.DETERMINISTIC.value
        assert env.observability == EnvType.FULLY_OBSERVABLE.value

        self.env = env
        self.num_of_samples = num_of_samples
        self.start_rrt = RRT(env, None, vertex_expand_func)
        self.goal_rrt = RRT(env, None, vertex_expand_func)
        self.tree = nx.Graph()
        self._sample_func = sample_func

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
        for i in range(self.num_of_samples):
            if i % 100 == 0:
                print('RRTConnect/run, iteration {}'.format(i))

            v_target = tuple(self._sample_func(self.env).tolist())
            res, v_new = rrt1.extend(v_target)

            # After extending rrt1 towards v_target, try to extend rrt2 towards the new vertex v_new.
            if res != RRT.TRAPPED:
                res, _ = rrt2.extend(v_new)
                if res == RRT.REACHED:
                    return self._get_path(start, goal)

            rrt1, rrt2 = rrt2, rrt1

        return False, None, None

    def _get_path(self, start: tuple, goal: tuple):
        self.combined_tree = self.get_tree()

        path = nx.shortest_path(self.combined_tree, start, goal, weight='weight')
        path_len = nx.shortest_path_length(self.combined_tree, start, goal, weight='weight')
        return True, path, path_len

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
