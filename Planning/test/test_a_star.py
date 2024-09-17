import sys
sys.path.append('../../environment/')
sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np
import random
import time
import math

from two_d_maze import TwoDMaze
from a_star import AStar

# Initialize environment
env = TwoDMaze()

# -------- Settings ------------
FIX_MAZE = True

# only applicable if FIX_MAZE is True
RANDOM_SEED = 3 # dont change this, This seed will ensure there is path from source to goal
NUM_OF_OBSTACLE_PER_ROW = 10 # dont change this, This seed will ensure there is path from source to goal

# -------- Helper Functions -------------
def heuristic_func(v, goal):
    # simply the distance between v and goal
    v_x = v // env.size
    v_y = v % env.size
    goal_x = goal // env.size
    goal_y = goal % env.size
    return math.sqrt((goal_x - v_x) ** 2 + (goal_y - v_y) ** 2)

# -------- Main Code ----------

# add random obstacle to environment
if FIX_MAZE:
    env.random_maze_obstacle_per_row(num_of_obstacle_per_row = NUM_OF_OBSTACLE_PER_ROW, random_seed = RANDOM_SEED)
else:
    env.random_maze_obstacle_per_row(num_of_obstacle_per_row = 10)

# env.plot()

# generate source and goal
# source_x, source_y = env.get_random_free_point()
# goal_x, goal_y = env.get_random_free_point()
# while goal_x == source_x and goal_y == source_y:
#     goal_x, goal_y = env.get_random_free_point()
for x in range(env.size):
    for y in range(env.size):
        if env.maze[x, y] == TwoDMaze.FREE_SPACE:
            source_x, source_y = x, y
            break

for x in reversed(range(env.size)):
    for y in reversed(range(env.size)):
        if env.maze[x, y] == TwoDMaze.FREE_SPACE:
            goal_x, goal_y = x, y
            break

# add source and goal to environment
env.add_source(source_x, source_y)
env.add_goal(goal_x, goal_y)
source = source_x * env.size + source_y
goal = goal_x * env.size + goal_y

# initialize planner
my_path_planner = AStar()

# run path planner
start_time = time.time()
res, shortest_path, shortest_path_len = my_path_planner.run(env.adjacency_list, source, goal, heuristic_func)
end_time = time.time()
print("TestAStar, takes {} seconds".format(end_time - start_time))

if not res:
    print("TestAStar, no path is available!")
else:
    print("TestAStar, found path of len {}".format(shortest_path_len))
    # visualize path
    path = []
    for v in shortest_path:
        v_x = v // env.size
        v_y = v % env.size
        path.append([v_x, v_y])

    env.add_path(path)

env.plot()


# -------- Legacy Code ----------
