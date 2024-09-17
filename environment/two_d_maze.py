import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import random
from collections import defaultdict

DEFAULT_MAZE_OBSTACLES= [(0, 0), (0, 4), (0, 8), (0, 15), (0, 23), (0, 30), (0, 34), (0, 37), (0, 38), (0, 40), (1, 9), (1, 12), (1, 14), (1, 16), (1, 25), (1, 30), (1, 34), (1, 35), (1, 40), (1, 45), (2, 0), (2, 4), (2, 9), (2, 14), (2, 24), (2, 33), (2, 40), (2, 42), (2, 47), (2, 49), (3, 1), (3, 2), (3, 10), (3, 17), (3, 19), (3, 30), (3, 37), (3, 38), (3, 48), (3, 49), (4, 6), (4, 8), (4, 23), (4, 24), (4, 25), (4, 27), (4, 28), (4, 36), (4, 45), (4, 46), (5, 2), (5, 8), (5, 13), (5, 16), (5, 19), (5, 27), (5, 31), (5, 40), (5, 43), (5, 49), (6, 14), (6, 21), (6, 22), (6, 24), (6, 26),
(6, 32), (6, 34), (6, 36), (6, 37), (6, 43), (7, 1), (7, 6), (7, 10), (7, 17), (7, 20), (7, 34), (7, 36), (7, 38), (7, 42), (7, 44), (8, 4), (8, 7), (8, 13), (8, 17), (8, 18), (8, 30), (8, 36), (8, 40), (8, 41), (8, 45), (9, 1), (9, 4), (9, 5), (9, 9), (9, 18), (9, 22), (9, 26), (9, 27), (9, 30), (9, 40), (10, 2), (10, 7), (10, 24), (10, 26), (10, 37), (10, 38), (10, 39), (10, 45), (10, 48), (10, 49), (11, 0), (11, 2), (11, 4), (11, 6), (11, 15), (11, 17), (11, 19), (11, 21), (11, 32), (11, 35), (12, 2), (12, 9), (12, 12), (12, 16), (12, 18), (12, 26), (12, 34), (12, 38), (12, 39), (12, 44), (13, 2), (13, 8), (13, 20), (13, 21), (13, 23), (13, 24), (13, 29), (13, 33), (13, 38), (13, 41), (14, 6), (14, 17), (14, 27), (14, 32), (14, 35), (14, 39), (14, 40), (14, 43), (14, 45), (14, 46), (15, 0), (15, 15), (15, 16), (15, 19), (15, 21), (15, 26), (15, 27), (15, 33), (15, 35), (15, 37), (16, 1), (16, 3), (16, 8), (16, 20), (16, 21), (16, 24), (16, 29), (16, 37), (16, 39), (16, 40), (17, 1), (17, 3), (17, 17), (17, 22), (17, 31), (17, 37), (17, 38), (17, 43), (17, 45), (17, 47), (18, 1), (18, 16), (18, 19), (18, 20), (18, 23), (18, 29), (18, 37), (18, 38), (18, 40), (18, 43), (19, 6), (19, 11), (19, 16), (19, 19), (19, 20), (19, 23), (19, 24), (19, 38), (19, 48), (19, 49), (20, 1), (20, 8), (20, 14), (20, 17), (20, 19), (20, 32), (20, 36), (20, 41), (20, 43), (20, 47), (21, 6), (21, 11), (21, 15), (21, 20), (21, 21), (21, 27), (21, 38), (21, 41), (21, 43), (21, 44), (22, 5), (22, 10), (22, 13), (22, 14), (22, 21), (22, 28), (22, 36), (22, 41), (22, 43), (22, 47), (23, 2), (23, 7), (23, 11), (23, 12), (23, 14), (23, 17), (23, 20), (23, 28), (23, 33), (23, 36), (24, 5), (24, 8), (24, 17), (24, 18), (24, 21), (24, 22), (24, 26), (24, 37), (24, 39), (24, 41), (25, 2), (25, 9), (25, 17), (25, 18), (25, 22), (25, 26), (25, 29), (25, 33), (25, 36), (25, 40), (26, 0), (26, 2), (26, 12), (26, 14), (26, 27), (26, 30), (26, 32), (26, 35), (26, 39), (26, 45), (27, 4), (27, 14), (27, 18), (27, 21), (27, 29), (27, 33), (27, 34), (27, 42), (27, 47), (27, 48), (28, 2), (28, 7), (28, 12), (28, 15), (28, 18), (28, 27), (28, 32), (28, 36), (28, 37), (28, 44), (29, 0), (29, 3), (29, 7), (29, 10), (29, 15), (29, 19), (29, 30), (29, 32), (29, 42), (29, 47), (30, 1), (30, 3), (30, 7), (30, 8), (30, 16), (30, 21), (30, 26), (30, 33), (30, 34), (30, 39), (31, 3), (31, 7), (31, 10), (31, 12), (31, 14), (31, 15), (31, 17), (31, 22), (31, 30), (31, 34), (32, 0), (32, 3), (32, 8), (32, 15), (32, 17), (32, 25), (32, 31), (32, 36), (32, 40), (32, 48), (33, 0), (33, 3), (33, 8), (33, 17), (33, 20), (33, 27), (33, 30), (33, 33), (33, 39), (33, 49), (34, 2), (34, 3), (34, 4), (34, 5), (34, 7), (34, 20), (34, 30), (34, 31), (34, 32), (34, 45), (35, 4), (35, 10), (35, 16), (35, 19), (35, 20), (35, 22), (35, 23), (35, 24), (35, 37), (35, 41), (36, 0), (36, 7), (36, 8), (36, 12), (36, 21), (36, 24), (36, 27), (36, 35), (36, 45), (36, 46), (37, 2), (37, 5), (37, 11), (37, 23), (37, 24), (37, 29), (37, 34), (37, 36), (37, 38), (37, 41), (38, 2),
(38, 3), (38, 20), (38, 23), (38, 27), (38, 31), (38, 39), (38, 40), (38, 44), (38, 48), (39, 1), (39, 4), (39, 13), (39, 15), (39, 17), (39, 26), (39, 29), (39, 34), (39, 37), (39, 44), (40, 1), (40, 7), (40, 8), (40, 14), (40, 16), (40, 20), (40, 23), (40, 27), (40, 29), (40, 35), (41, 6), (41, 7), (41, 20), (41, 24), (41, 33), (41, 34), (41, 36), (41, 42), (41, 44), (41, 46),
(42, 0), (42, 2), (42, 6), (42, 9), (42, 15), (42, 24), (42, 30), (42, 37), (42, 45), (42, 49), (43, 1), (43, 5), (43, 6), (43, 7), (43, 11), (43, 21), (43, 24), (43, 33), (43, 36), (43, 42), (44, 1), (44, 2), (44, 5), (44, 7), (44, 18), (44, 19), (44, 30), (44, 37), (44, 43), (44, 44), (45, 3), (45, 6), (45, 15), (45, 32), (45, 33), (45, 35), (45, 36), (45, 45), (45, 47), (45, 49), (46, 4), (46, 11), (46, 15), (46, 20), (46, 29), (46, 35), (46, 36), (46, 39), (46, 41), (46, 44), (47, 5), (47, 16), (47, 22), (47, 23), (47, 24), (47, 25), (47, 26), (47, 35), (47, 38), (47, 48), (48, 10), (48, 15), (48, 26), (48, 32), (48, 36), (48, 37), (48, 43), (48, 44), (48, 47), (48, 48), (49, 6), (49, 9), (49, 10), (49, 25), (49, 30), (49, 31), (49, 33), (49, 41), (49, 43), (49, 47)]

class TwoDMaze(object):
    FREE_SPACE = 0
    OBSTACLE = 1
    SOURCE = 2
    GOAL = 3
    PATH = 4
    WAYPOINT = 5
    MAX_POINT_TYPE = 6

    def __init__(self, size = 50):
        self.size = size
        self.maze = np.full((size, size), TwoDMaze.FREE_SPACE)
        self.adjacency_list = self.compute_adjacency_list()
        self.colour_map = colors.ListedColormap(['white', 'black', 'red', 'blue', 'green', 'yellow'])
        bounds = [TwoDMaze.FREE_SPACE, TwoDMaze.OBSTACLE, TwoDMaze.SOURCE, TwoDMaze.GOAL, TwoDMaze.PATH, TwoDMaze.WAYPOINT, TwoDMaze.MAX_POINT_TYPE]
        self.norm = colors.BoundaryNorm(bounds, self.colour_map.N)

    def compute_adjacency_list(self):
        adjacency_list = defaultdict(dict)

        for i in range(self.size):
            for j in range(self.size):
                if j + 1 < self.size and self.maze[i, j+1] == TwoDMaze.FREE_SPACE:
                    adjacency_list[(i, j)][(i, j+1)] = 1
                if j > 0 and self.maze[i, j-1] == TwoDMaze.FREE_SPACE:
                    adjacency_list[(i, j)][(i, j-1)] = 1
                if i > 0 and self.maze[i-1, j] == TwoDMaze.FREE_SPACE:
                    adjacency_list[(i, j)][(i-1, j)] = 1
                if i + 1 < self.size and self.maze[i+1, j] == TwoDMaze.FREE_SPACE:
                    adjacency_list[(i, j)][(i+1, j)] = 1

        self.adjacency_list = adjacency_list
        return adjacency_list

    def add_default_obstacles(self):
        for x, y in DEFAULT_MAZE_OBSTACLES:
            self.maze[x, y] = TwoDMaze.OBSTACLE
        self.adjacency_list = self.compute_adjacency_list()

    def random_maze_obstacle_every_grid(self):
        for x in range(self.size):
            for y in range(self.size):
                self.maze[x, y] = random.randint(TwoDMaze.FREE_SPACE, TwoDMaze.OBSTACLE)
        self.compute_adjacency_list()

    def random_maze_obstacle_per_row(self, num_of_obstacle_per_row = 1):
        for x in range(self.size):
            cnt = 0
            while cnt < num_of_obstacle_per_row:
                y = random.randint(0, self.size - 1)
                if self.maze[x, y] == TwoDMaze.OBSTACLE:
                    continue

                self.maze[x, y] = TwoDMaze.OBSTACLE
                cnt += 1
        self.compute_adjacency_list()

    def plot(self):
        fig, ax = plt.subplots()
        ax.imshow(self.maze, cmap=self.colour_map, norm=self.norm)
        # draw gridlines
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
        ax.set_xticks(np.arange(0.5, self.size, 1))
        ax.set_xticklabels(np.array([str(i) for i in range(self.size)]))
        ax.set_yticks(np.arange(0.5, self.size, 1))
        ax.set_yticklabels(np.array([str(i) for i in range(self.size)]))
        # ax.axis('off')
        plt.tick_params(axis='both', labelsize=5, length = 0)

        # fig.set_size_inches((8.5, 11), forward=False)
        plt.show()

    def get_random_free_point(self):
        x = random.randint(0, self.size - 1)
        y = random.randint(0, self.size - 1)
        while self.maze[x, y] != TwoDMaze.FREE_SPACE:
            x = random.randint(0, self.size - 1)
            y = random.randint(0, self.size - 1)

        return (x, y)

    def get_random_point(self):
        x = random.randint(0, self.size - 1)
        y = random.randint(0, self.size - 1)

        return (x, y)

    def add_source(self, source):
        x, y = source
        if self.maze[x, y] == TwoDMaze.OBSTACLE:
            print("TwoDMaze/add_source: unable to add source on obstalce")
            return False

        if self.maze[x, y] == TwoDMaze.GOAL:
            print("TwoDMaze/add_source: unable to add source on goal")
            return False

        self.maze[x, y] = TwoDMaze.SOURCE
        return True

    def add_goal(self, goal):
        x, y = goal
        if self.maze[x, y] == TwoDMaze.OBSTACLE:
            print("TwoDMaze/add_source: unable to add source on obstalce")
            return False

        if self.maze[x, y] == TwoDMaze.SOURCE:
            print("TwoDMaze/add_source: unable to add source on source")
            return False

        self.maze[x, y] = TwoDMaze.GOAL
        return True

    def add_random_source(self):
        source = self.get_random_free_point()
        return self.add_source(source)

    def add_random_goal(self):
        goal = self.get_random_free_point()
        return self.add_goal(goal)

    def add_path(self, path, only_change_free_space=False):
        '''
        @param, a path that includes source and goal
        '''
        for x, y in path[1:-1]:
            if only_change_free_space:
                if self.maze[x, y] == TwoDMaze.FREE_SPACE:
                    self.maze[x, y] = TwoDMaze.PATH
            else:
                self.maze[x, y] = TwoDMaze.PATH

        return True

    def add_point(self, p, p_type):
        x, y = p
        if p_type >= TwoDMaze.FREE_SPACE and p_type < TwoDMaze.MAX_POINT_TYPE:
            self.maze[x, y] = p_type
        else:
            print("TwoDMaze/add_point, invalid point type {}".format(p_type))
            return False