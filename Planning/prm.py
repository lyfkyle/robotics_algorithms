from dijkstra import Dirkstra
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import numpy as np

class ProbabilisticRoadmap(object):
    def __init__(self, number_of_vertices, K):
        self.number_of_vertices = number_of_vertices
        self.K = K
        self._path_finder = Dirkstra()

    def compute_roadmap(self, sample_vertex, check_clear, check_link):
        '''
        Offline computation of roadmap by sampling points
        @param, sample_vertex, the sampling function, returns a random point in space
        @param, check_clear, a function that takes in a random point and returns whether it is in collision
        @param, check_link, a function that takes in two points and returns whether there is a collision-free simple path between them
        @return, None
        '''
        self.V = []
        self.adj_list = defaultdict(dict)
        while len(self.V) < self.number_of_vertices:
            collision_free = False
            while not collision_free:
                v = sample_vertex()
                collision_free = check_clear(v)

            v_x, v_y = v
            self.V.append([v_x, v_y])

        print("PRM/compute_roadmap: finished adding {} of vertices".format(self.number_of_vertices))
        # print(self.V)

        for v in self.V:
            neighbours = self.get_K_closest_neighbour(self.V, v, self.K)
            # print("neighbours {}".format(neighbours))
            for neighbour in neighbours:
                can_link, length = check_link(v, neighbour)
                if can_link:
                    self.adj_list[tuple(v)][tuple(neighbour)] = length
                    self.adj_list[tuple(neighbour)][tuple(v)] = length

    def get_K_closest_neighbour(self, V, v, K):
        '''
        return K closest neighbours of v in V
        @param V, a list of vertices, must be a 2D numpy array
        @param v, the target vertice, must be a 2D numpy array
        @param K, number of neighbours to get
        @return, 2D numpy array, the list of nearest neighbours
        '''

        nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(np.array(V))
        distances, indices = nbrs.kneighbors(np.array(v).reshape(1,2))
        # print("indices {}".format(indices))
        return np.take(np.array(V), indices.ravel(), axis=0).tolist()

    def get_path(self, source, goal, check_link):
        '''
        Online computation of path
        @param, source, the source point
        @param, goal, the goal point
        @param, check_link, a function that takes in two points and returns whether there is a collision-free simple path between them
        @return, boolean, return true if a path is found, return false otherwise
                 shortest_path, a list of vertice if shortest path is found
                 shortest_path_len, the length of shortest path if found.
        '''
        source_neighbours = self.get_K_closest_neighbour(self.V, source, self.K)
        for neighbour in source_neighbours:
            can_link, length = check_link(source, neighbour)
            if can_link:
                self.adj_list[tuple(source)][tuple(neighbour)] = length
                self.adj_list[tuple(neighbour)][tuple(source)] = length
                break
        else:
            print("can't connect source to roadmap!!!")
            return False, None, None

        goal_neighbours = self.get_K_closest_neighbour(self.V, goal, self.K)
        for neighbour in goal_neighbours:
            can_link, length = check_link(goal, neighbour)
            if can_link:
                self.adj_list[tuple(goal)][tuple(neighbour)] = length
                self.adj_list[tuple(neighbour)][tuple(goal)] = length
                break
        else:
            print("can't connect goal to roadmap!!!")
            return False, None, None

        path_exist, shortest_path, path_len = self._path_finder.run(self.adj_list, source, goal)
        if not path_exist:
            print("No path is found, this should not happen!!!")
            return False, None, None
        else:
            return True, shortest_path, path_len

    def run(self, graph, source, goal, sample_vertex, check_clear, check_link):
        self.compute_roadmap(sample_vertex, check_clear, check_link)
        return self.get_path(source, goal, check_link)

    def get_roadmap(self):
        return self.adj_list



