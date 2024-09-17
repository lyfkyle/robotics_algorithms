from dijkstra import Dirkstra
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import numpy as np


class RRT(object):
    TRAPPED = 0
    REACHED = 1
    ADVANCED = 2

    def __init__(self):
        self.V = []  # vertice list
        self.adj_list = defaultdict(dict)

    def initialize_tree(self, initial_vertice_list):
        self.V = initial_vertice_list

    def extend(self, v_goal, expand_func, delta):
        '''
        extend towards v one step
        '''
        nearest_neighbour = self.get_nearest_neighbour(np.array(self.V), np.array(v_goal).reshape(1, 2))

        v_source = tuple(nearest_neighbour[0])
        v = expand_func(v_source, v_goal, delta)
        if v is not None:
            self.V.append(list(v))
            self.adj_list[v][v_source] = delta
            self.adj_list[v_source][v] = delta

        if v is None:
            return RRT.TRAPPED, None
        elif v == v_goal:
            return RRT.REACHED, v
        else:
            return RRT.ADVANCED, v

    def connect(self, v_goal, expand_func, delta):
        '''
        extend towards v
        '''
        res = RRT.ADVANCED
        while res == RRT.ADVANCED:
            res, _ = self.extend(v_goal, expand_func, delta)

        return res

    def get_nearest_neighbour(self, V, v):
        '''
        return the closest neighbours of v in V
        @param V, a list of vertices, must be a 2D numpy array
        @param v, the target vertice, must be a 2D numpy array
        @return, list, the nearest neighbours
        '''

        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(V)
        distances, indices = nbrs.kneighbors(v)
        # print("indices {}".format(indices))
        return np.take(np.array(V), indices.ravel(), axis=0).tolist()


class RRTConnect(object):
    def __init__(self, number_of_samples):
        self.N = number_of_samples
        self._path_finder = Dirkstra()
        self.source_rrt = RRT()
        self.goal_rrt = RRT()
        self.adj_list = defaultdict(dict)

    def run(self, source, goal, sample_vertex_func, expand_func, delta):
        self.source_rrt.initialize_tree([source])
        self.goal_rrt.initialize_tree([goal])

        rrt1 = self.source_rrt
        rrt2 = self.goal_rrt
        for i in range(self.N):
            if i % 100 == 0:
                print("RRTConnect/run, iteration {}".format(i))

            v_new = sample_vertex_func()
            res, v = rrt1.extend(v_new, expand_func, delta)
            if res != RRT.TRAPPED:
                res = rrt2.connect(v, expand_func, delta)
                if res == RRT.REACHED:
                    return self.get_path(source, goal)

            rrt1, rrt2 = rrt2, rrt1

        return False, None, None

    def get_path(self, source, goal):
        adj_list = self.get_tree()

        # call dijkstra to find the path
        path_exist, shortest_path, path_len = self._path_finder.run(adj_list, source, goal)
        if not path_exist:
            print("RRTConnect/get_path: No path is found, this should not happen!!!")
            return False, None, None
        else:
            return True, shortest_path, path_len

    def get_tree(self):
        # merge adj_list from two rrt
        adj_list = self.source_rrt.adj_list.copy()
        for key in self.goal_rrt.adj_list:
            for key1, val1 in self.goal_rrt.adj_list[key].items():
                adj_list[key][key1] = val1
        self.adj_list = adj_list

        return self.adj_list

