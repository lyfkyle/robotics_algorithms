from dijkstra import Dirkstra

class ProbabilisticRoadmap(object):
    def __init__(self, number_of_vertices, K):
        self.number_of_vertices = number_of_vertices
        self.K = K
        self._path_finder = Dirkstra()

    def compute_roadmap(self, sample_vertex, check_clear, check_link):
        self.V = []
        self.adj_list = {dict()} * self.number_of_vertices
        while len(V) < self.number_of_vertices:
            collision_free = False
            while not collision_free:
                v = sample_vertex()
                collision_free = check_clear(v)

            V.append(v)

        print("PRM/compute_roadmap: finished adding {} of vertices".format(self.number_of_vertices))

        for v in V:
            neighbours = self.get_K_closest_neighbour(V, v, self.K)
            for neighbour in neighbours:
                can_link, length = check_link(v, neigbour)
                if can_link and neighbour not in E[v]:
                    adj_list[v][neighbour] = length

    def get_K_closest_neighbour(self, V, v, K):

    def get_path(self, source, goal, check_link):
        source_neighbours = self.get_K_closest_neighbour(V, source, self.K)
        for neighbour in source_neighbours:
            can_link, length = check_link(source, neigbour)
            if can_link:
                self.E[source] = {neighbour: length}
                break

        goal_neighbours = self.get_K_closest_neighbour(V, goal, self.K)
        for neighbour in goal_neighbours:
            can_link, length = check_link(goal, neigbour)
            if can_link:
                self.E[goal] = {neighbour: length}
                break

        path_exist, shortest_path, path_len = self._path_finder(self.adj_list, source, goal)
        if not path_exist:
            print("No path is found, this should not happen!!!")
            return False, None, None
        else:
            return True, shortest_path, path_len

    def run(self, graph, source, goal, sample_vertex, check_clear, check_link):
        self.compute_roadmap(sample_vertex, check_clear, check_link)
        _, path, path_len = self.get_path(source, goal, check_link)
        return path, path_len




