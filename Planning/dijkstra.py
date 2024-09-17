class Dirkstra(object):
    def __init__(self):
        pass

    def run(self, graph, source, goal):
        '''
        @param, graph, represented by adjacency list
        @param, source, the source vertex
        @param, goal, the goal vertex
        @return, boolean, return true if a path is found, return false otherwise
                 shortest_path, a list of vertice if shortest path is found
                 shortest_path_len, the length of shortest path if found.
        '''

        # initialzie
        unvisited_vertex_set = set()
        shortest_path = []
        shortest_path_len = 0
        dist = [0] * len(graph)
        prev = [-1] * len(graph) # used to extract shortest path

        for v in range(len(graph)):
            dist[v] = float('inf')
            unvisited_vertex_set.add(v)
        dist[source] = 0 # distance to source is 0

        # run algorithm
        path_exist = True
        while len(unvisited_vertex_set) > 0:
            min_dist = float('inf')
            for v in unvisited_vertex_set:
                if dist[v] < min_dist:
                    min_dist = dist[v]
                    min_v = v

            # print("mid_dist: {}".format(min_dist))
            # there is no path
            if min_dist == float('inf'):
                path_exist = False
                break

            # path to goal is found
            if min_v == goal:
                break

            unvisited_vertex_set.remove(min_v)

            for v, edge_length in graph[min_v]:
                if v in unvisited_vertex_set:
                    if dist[min_v] + edge_length < dist[v]:
                        dist[v] = dist[min_v] + edge_length
                        prev[v] = (min_v, edge_length)

        if path_exist:
            # extract shortest path:
            v = goal
            prev_v, edge_length = prev[v]
            while prev_v != -1 and prev_v != source:
                shortest_path.insert(0, prev_v)
                shortest_path_len += edge_length
                prev_v, edge_length = prev[prev_v]

            return (True, shortest_path, shortest_path_len)
        else:
            return (False, None, None)