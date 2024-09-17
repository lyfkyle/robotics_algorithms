class AStar(object):
    def __init__(self):
        pass

    def run(self, graph, source, goal, heuristic_func):
        '''
        @param, graph, represented by adjacency list
        @param, source, the source vertex
        @param, goal, the goal vertex
        @param, heuristic_func, a function to return estimated cost to go from a vertex to goal
        @return, boolean, return true if a path is found, return false otherwise
                 shortest_path, a list of vertice if shortest path is found
                 shortest_path_len, the length of shortest path if found.
        '''

        # initialzie
        # for every vertex, dist[v] = g(s, v) + h(v, g)
        unvisited_vertices_set = set() # OPEN set. Nodes not in this set is in CLOSE set
        shortest_path = []
        shortest_path_len = 0
        g = {} # cost from source to v
        dist = {}
        prev = {} # used to extract shortest path

        for v in graph:
            g[v] = float('inf')
            dist[v] = float('inf')
            unvisited_vertices_set.add(v)
        g[source] = 0
        dist[source] = heuristic_func(source, goal) # cost from source to goal = g(s, s) + h(s, g) = 0 + h(s, g)

        # run algorithm
        path_exist = True
        while len(unvisited_vertices_set) > 0:
            min_dist = float('inf')
            for v in unvisited_vertices_set:
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

            unvisited_vertices_set.remove(min_v)

            for v, edge_length in graph[min_v].items():
                if v in unvisited_vertices_set:
                    g_v = g[min_v] + edge_length
                    h_v = heuristic_func(v, goal)
                    if g_v < g[v]:
                        g[v] = g_v
                        dist[v] = g_v + h_v
                        prev[v] = min_v

        if path_exist:
            # extract shortest path:
            shortest_path.insert(0, goal)
            v = goal
            prev_v = prev[v]
            while prev_v != -1 and prev_v != source:
                shortest_path.insert(0, prev_v)
                prev_v= prev[prev_v]

            shortest_path.insert(0, source)
            shortest_path_len = dist[goal]
            return (True, shortest_path, shortest_path_len)
        else:
            return (False, None, None)