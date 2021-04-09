from dijkstra import Dirkstra
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import numpy as np
import math

class RRTStar(object):
    TRAPPED = 0
    REACHED = 1
    ADVANCED = 2

    def __init__(self):
        self.V = {}  # vertice dict
        self.g = {}
        self.adj_list = defaultdict(dict)

    def initialize_tree(self, root):
        self.root = root
        self.g[root] = 0 # cost_to_come to itself is 0.
        self.V[root] = root

    def extend(self, v_goal, expand_func, delta, distance_func, check_link, radius = 2):
        '''
        Extend towards v_goal one step.
            1. find nearest_neighbour
            2. extend from nearest_neighbour to v_goal -> v_new
            3. find best_neighbour around v_new
            4. connect best_neighbour to v_new
            5. rewire neighbours to v_new
        @param v_goal, tuple, the goal vertex to extend to
        @param expand_func, function(v1, v2, delta) -> v_new, a function that expand tree from v1 to v2 by delta, return resulting v_new
        @param delta, the delta for expand_func
        @param distance_func, function(v1, v2) -> dist, a function to compute distance between v1 and v2
        @param check_link, function(v1, v2) -> can_link, dist, a function that checks whether v1 and v2 can be linked by a simple path.
        @param radius, the radius to search around v_new to find neighbours.
        '''
        # Like A-Star, the notion here is dist[v] = g(s, v) + h(v, g). (total cost = cost to come + cost to go)

        # check whether v_goal is already reached
        if v_goal in self.V:
            return RRTStar.REACHED, v_goal

        # find nearest neighbour
        nearest_neighbour, _ = self.get_nearest_neighbour(np.array(list(self.V)), np.array(v_goal).reshape(1, 2))
        # expand towards v_goal
        v_new = expand_func(nearest_neighbour, v_goal, delta)
        if v_new is not None:
            # !!! Instead of using nearest neighbour, find best_neighbour around v_new
            neighbours = self.find_neighbours_within_radius(v_new, radius, distance_func)
            if len(neighbours) > 0:
                # filter out those that can't connect to v_new
                neighbours = self._filter_neighbours(neighbours, v_new, check_link)
                # find best neighbour
                best_neighbour, best_g = self.find_best_neighbour(neighbours, v_new, radius, distance_func)
            else:
                best_neighbour, best_g = nearest_neighbour, distance_func(nearest_neighbour, v_new) + self.g[nearest_neighbour]

            # add v_new to tree
            v_new = tuple(v_new)
            self.V[v_new] = best_neighbour

            # add edge from best_neighbour to v_new
            self.adj_list[v_new][best_neighbour] = best_g - self.g[best_neighbour] # use best_g to compute dist
            self.adj_list[best_neighbour][v_new] = best_g - self.g[best_neighbour] # use best_g to compute dist
            self.g[v_new] = best_g

            # !!! rewire
            for neighbour in neighbours:
                dist_v_neighbour = distance_func(v_new, neighbour)
                if self.g[v_new] + dist_v_neighbour < self.g[neighbour]:
                    self.g[neighbour] = self.g[v_new] + dist_v_neighbour
                    # self.adj_list[neighbour].clear() # clear all?
                    self.adj_list[neighbour].pop(self.V[neighbour]) # delete connection from neighbour to its parent
                    self.adj_list[v_new][neighbour] = dist_v_neighbour
                    self.adj_list[neighbour][v_new] = dist_v_neighbour
                    self.V[neighbour] = v_new

        if v_new is None:
            return RRTStar.TRAPPED, None
        elif v_new == v_goal:
            return RRTStar.REACHED, v_new
        else:
            return RRTStar.ADVANCED, v_new

    def connect(self, v_goal, expand_func, delta, distance_func, check_link):
        '''
        extend towards v
        '''
        # print('RRTStar/Conect')
        res = RRTStar.ADVANCED
        while res == RRTStar.ADVANCED:
            res, _ = self.extend(v_goal, expand_func, delta, distance_func, check_link)

        return res

    def get_nearest_neighbour(self, V, v):
        '''
        return the closest neighbours of v in V
        @param V, a list of vertices, must be a 2D numpy array
        @param v, the target vertice, must be a 2D numpy array
        @return, (list, distance), the nearest neighbour index and its distance
        '''

        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(V)
        distances, indices = nbrs.kneighbors(v)
        # print("indices {}".format(indices))
        indices = np.take(np.array(V), indices.ravel(), axis=0).tolist()
        indice = tuple(indices[0])
        distance = distances[0]
        return indice, distance

    def find_neighbours_within_radius(self, v, radius, distance_func):
        '''
        find all vertices within radius to v
        @param, v, tuple, the center vertex
        @param, radius, the radius to search
        @return neighbours, list of tuple, the neighbour vertices.
        '''
        neighbours = []
        for v1 in self.V: # this is expensive, any alternatives?
            dist_v_v1 = distance_func(v, v1)
            if dist_v_v1 <= radius:
                neighbours.append(v1)

        return neighbours

    def find_best_neighbour(self, v_candidates, v_goal, radius, distance_func):
        '''
        find among v_candidates, the best v that has the minimum cost-to-come to v_goal
        @param, v_candidates, list of tuple, a candidates of vertice
        @param, v_goal, tuple, v_goal
        @param radius, float, the radius to search
        @return best_neighbour, tuple, the best_neighbour
        '''

        best_g = float('inf')
        best_neighbour = v_candidates[0]
        for v1 in v_candidates:
            dist_v1_v_goal = distance_func(v1, v_goal)
            g_v_goal = self.g[v1] + dist_v1_v_goal
            if g_v_goal < best_g:
                best_g = g_v_goal
                best_neighbour = v1

        return best_neighbour, best_g

    def _filter_neighbours(self, neighbours, v_new, check_link):
        filtered_neighbours = []
        for neighbour in neighbours:
            collision_free, _ = check_link(neighbour, v_new)
            if collision_free:
                filtered_neighbours.append(neighbour)
        return filtered_neighbours

class RRTStarConnect(object):
    def __init__(self, number_of_samples):
        self.N = number_of_samples
        self._path_finder = Dirkstra()
        self.source_rrt = RRTStar()
        self.goal_rrt = RRTStar()
        self.adj_list = defaultdict(dict)

    def run(self, source, goal, sample_vertex_func, expand_func, delta, distance_func, check_link):
        self.source_rrt.initialize_tree(source)
        self.goal_rrt.initialize_tree(goal)

        rrt1 = self.source_rrt
        rrt2 = self.goal_rrt
        reached = False
        for i in range(self.N):
            if i % 100 == 0:
                print("RRTConnect/run, iteration {}".format(i))

            v_new = sample_vertex_func()
            res, v = rrt1.extend(v_new, expand_func, delta, distance_func, check_link)
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

