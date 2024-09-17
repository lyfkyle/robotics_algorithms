import sys
sys.path.append('../../environment/')
sys.path.append('../')

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import random
import time

from two_d_maze import TwoDMaze
from prm import ProbabilisticRoadmap

# Initialize environment
env = TwoDMaze()

# -------- Settings ------------
FIX_MAZE = True

# -------- Helper Functions -------------
def sample_vertex():
    x, y = env.get_random_point()
    return x, y

def check_clear(vertex):
    x, y = vertex
    return env.maze[x, y] == env.FREE_SPACE

def check_link(v1, v2):
    v1_x, v1_y = v1
    v2_x, v2_y = v2

    x1 = min(v1_x, v2_x)
    x2 = max(v1_x, v2_x)
    y1 = min(v1_y, v2_y)
    y2 = max(v1_y, v2_y)

    length = y2 - y1 + x2 - x1 - 1
    # try right then top
    path_exist = True
    for y in range(y1, y2+1):
        if env.maze[x2, y] == env.OBSTACLE:
            path_exist = False
            break

    for x in range(x1, x2+1):
        if env.maze[x, y2] == env.OBSTACLE:
            path_exist = False
            break

    if path_exist:
        return True, length

    # try top then right
    path_exist = True
    for x in range(x1, x2+1):
        if env.maze[x, y1] == env.OBSTACLE:
            path_exist = False
            break

    for y in range(y1, y2+1):
        if env.maze[x1, y] == env.OBSTACLE:
            path_exist = False
            break

    if path_exist:
        return True, length

    return False, 0

def compute_local_path(v1, v2):
    v1_x, v1_y = v1
    v2_x, v2_y = v2

    x1 = min(v1_x, v2_x)
    x2 = max(v1_x, v2_x)
    y1 = min(v1_y, v2_y)
    y2 = max(v1_y, v2_y)

    # try right then top
    local_path = []
    path_exist = True
    for y in range(y1, y2+1):
        if env.maze[x2, y] == env.OBSTACLE:
            path_exist = False
            break
        local_path.append((x2, y))

    for x in range(x1, x2+1):
        if env.maze[x, y2] == env.OBSTACLE:
            path_exist = False
            break
        local_path.append((x, y2))

    if path_exist:
        return local_path

    # try top then right
    local_path.clear()
    path_exist = True
    for x in range(x1, x2+1):
        if env.maze[x, y1] == env.OBSTACLE:
            path_exist = False
            break
        local_path.append((x, y1))

    for y in range(y1, y2+1):
        if env.maze[x1, y] == env.OBSTACLE:
            path_exist = False
            break
        local_path.append((x1, y))

    if path_exist:
        return local_path

    return None

def compute_source():
    for x in reversed(range(env.size)):
        for y in range(env.size):
            if env.maze[x, y] == TwoDMaze.FREE_SPACE:
                source = x, y
                return source

def compute_goal():
    for x in range(env.size):
        for y in reversed(range(env.size)):
            if env.maze[x, y] == TwoDMaze.FREE_SPACE:
                goal = x, y
                return goal


# -------- Main Code ----------
# add default obstacles
env.add_default_obstacles()

# generate source and goal
# source_x, source_y = env.get_random_free_point()
# goal_x, goal_y = env.get_random_free_point()
# while goal_x == source_x and goal_y == source_y:
#     goal_x, goal_y = env.get_random_free_point()


# add source and goal to environment
source = compute_source()
goal = compute_goal()
# goal = (45, 0)
env.add_source(source)
env.add_goal(goal)
# path = compute_local_path(source, goal)
# env.add_path(path)
# env.plot()

# initialize planner
my_path_planner = ProbabilisticRoadmap(number_of_vertices = 1000, K = 10) # 1000 samples out of total 2500 vertex.

# offline portion of PRM
start_time = time.time()
my_path_planner.compute_roadmap(sample_vertex, check_clear, check_link)
end_time = time.time()
print("TestPRM, offline takes {} seconds".format(end_time - start_time))

# run path planner
start_time = time.time()
res, shortest_path, shortest_path_len = my_path_planner.get_path(source, goal, check_link)
end_time = time.time()
print("TestPRM, online takes {} seconds".format(end_time - start_time))

# visualize roadmap
roadmap = my_path_planner.get_roadmap()
for vertex in roadmap:
    # for v in roadmap[vertex]:
    #     path = compute_local_path(vertex, v)
    #     if path is None:
    #         print("!!! this should not happen !!!")
    #         break
    #     else:
    #         env.add_path(path)
    if vertex != goal and vertex != source:
        env.add_point(vertex, env.WAYPOINT)

if not res:
    print("TestPRM, no path is available!")
else:
    print("TestPRM, found path of len {}".format(shortest_path_len))
    # visualize path
    path = []
    v1 = source
    for v2 in shortest_path[1:]:
        path = compute_local_path(v1, v2)
        if path is None:
            print("!!! this should not happen !!!")
            break
        else:
            env.add_path(path)

        v1 = v2


env.plot()


# -------- Legacy Code ----------

"""
EMPTY_CELL = 0
OBSTACLE_CELL = 1
START_CELL = 2
GOAL_CELL = 3
MOVE_CELL = 4
# create discrete colormap
cmap = colors.ListedColormap(['white', 'black', 'green', 'red', 'blue'])
bounds = [EMPTY_CELL, OBSTACLE_CELL, START_CELL, GOAL_CELL, MOVE_CELL ,MOVE_CELL + 1]
norm = colors.BoundaryNorm(bounds, cmap.N)

def plot_grid(data):

    fig, ax = plt.subplots()
    ax.imshow(data, cmap=cmap, norm=norm)
    # draw gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
    ax.set_xticks(np.arange(0.5, rows, 1));
    ax.set_yticks(np.arange(0.5, cols, 1));
    plt.tick_params(axis='both', labelsize=0, length = 0)
    # fig.set_size_inches((8.5, 11), forward=False)
    plt.show()

def generate_moves(grid, startX, startY):
    num_rows = np.size(grid, 0)
    num_cols = np.size(grid, 1)

    # Currently do not support moving diagonally so there is a max
    # of 4 possible moves, up, down, left, right.
    possible_moves = np.zeros(8, dtype=int).reshape(4, 2)
    # Move up
    possible_moves[0, 0] = startX - 1
    possible_moves[0, 1] = startY
    # Move down
    possible_moves[1, 0] = startX + 1
    possible_moves[1, 1] = startY
    # Move left
    possible_moves[2, 0] = startX
    possible_moves[2, 1] = startY - 1
    # Move right
    possible_moves[3, 0] = startX
    possible_moves[3, 1] = startY + 1
    # Change the cell value if the move is valid
    for row in possible_moves:
        if row[0] < 0 or row[0] >= num_rows:
            continue
        if row[1] < 0 or row[1] >= num_cols:
            continue
        grid[row[0], row[1]] = MOVE_CELL



if __name__ == "__main__":
    # rows = 20
    # cols = 20
    # # Randomly create 20 different grids
    # for i in range(0, 20):

    #     data = np.zeros(rows * cols).reshape(rows, cols)
    #     start_x = random.randint(0, rows - 1)
    #     start_y = random.randint(0, cols - 1)
    #     data[start_x, start_y] = START_CELL

    #     goal_x = random.randint(0, rows - 1)
    #     # Dont want the start and end positions to be the same
    #     # so keep changing the goal x until its different.
    #     # If X is different dont need to check Y
    #     while goal_x is start_x:
    #         goal_x = random.randint(0, rows - 1)
    #     goal_y = random.randint(0, cols - 1)

    #     data[goal_x, goal_y] = GOAL_CELL
    #     generate_moves(data, start_x, start_y)
    #     plot_grid(data)


"""