import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import random

class TwoDMaze(object):
    FREE_SPACE = 0
    OBSTACLE = 1
    SOURCE = 2
    GOAL = 3
    PATH = 4

    def __init__(self, size = 50):
        self.size = size
        self.maze = np.full((size, size), 0)
        self.adjacency_list = self.compute_adjacency_list
        self.colour_map = colors.ListedColormap(['white', 'black', 'red', 'blue', 'green'])
        bounds = [0, 1]
        self.norm = colors.BoundaryNorm(bounds, self.colour_map.N)

    def compute_adjacency_list(self):
        adjacency_list = []

        for i in range(self.size):
            for j in range(self.size):
                l = []
                if j + 1 < self.size and self.maze[i, j+1] == TwoDMaze.FREE_SPACE:
                    l.append((i * self.size + j + 1, 1))
                if j > 0 and self.maze[i, j-1] == TwoDMaze.FREE_SPACE:
                    l.append((i * self.size + j - 1, 1))
                if i > 0 and self.maze[i-1, j] == TwoDMaze.FREE_SPACE:
                    l.append(((i - 1) * self.size + j, 1))
                if i + 1 < self.size and self.maze[i+1, j] == TwoDMaze.FREE_SPACE:
                    l.append(((i + 1) * self.size + j, 1))
                adjacency_list.append(l)

        self.adjacency_list = adjacency_list
        return adjacency_list

    def random_maze_obstacle_every_grid(self):
        for x in range(self.size):
            for y in range(self.size):
                self.maze[x, y] = random.randint(TwoDMaze.FREE_SPACE, TwoDMaze.OBSTACLE)
        self.compute_adjacency_list()

    def random_maze_obstacle_per_row(self, num_of_obstacle_per_row = 1, random_seed = None):
        if random_seed is not None:
            random.seed(random_seed)

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
        ax.imshow(self.maze, cmap=self.colour_map)
        # draw gridlines
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
        ax.set_xticks(np.arange(0.5, self.size, 1))
        ax.set_yticks(np.arange(0.5, self.size, 1))
        # ax.axis('off')
        plt.tick_params(axis='both', labelsize=0, length = 0)

        # fig.set_size_inches((8.5, 11), forward=False)
        plt.show()

    def get_random_free_point(self):
        x = random.randint(0, self.size - 1)
        y = random.randint(0, self.size - 1)
        while self.maze[x, y] != TwoDMaze.FREE_SPACE:
            x = random.randint(0, self.size - 1)
            y = random.randint(0, self.size - 1)

        return (x, y)

    def add_source(self, x, y):
        if self.maze[x, y] == TwoDMaze.OBSTACLE:
            print("TwoDMaze/add_source: unable to add source on obstalce")
            return False

        if self.maze[x, y] == TwoDMaze.GOAL:
            print("TwoDMaze/add_source: unable to add source on goal")
            return False

        self.maze[x, y] = TwoDMaze.SOURCE
        return True

    def add_goal(self, x, y):
        if self.maze[x, y] == TwoDMaze.OBSTACLE:
            print("TwoDMaze/add_source: unable to add source on obstalce")
            return False

        if self.maze[x, y] == TwoDMaze.SOURCE:
            print("TwoDMaze/add_source: unable to add source on source")
            return False

        self.maze[x, y] = TwoDMaze.GOAL
        return True

    def add_random_source(self):
        x, y = self.get_random_free_point()
        return self.add_source(x, y)

    def add_random_goal(self):
        x, y = self.get_random_free_point()
        return self.add_goal(x, y)

    def add_path(self, path):
        for x, y in path:
            self.maze[x, y] = TwoDMaze.PATH

        return True
