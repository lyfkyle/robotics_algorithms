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

# -------- Helper Functions -------------
def heuristic_func(v, goal):
    # simply the distance between v and goal
    v_x, v_y = v
    goal_x, goal_y = goal
    return math.sqrt((goal_x - v_x) ** 2 + (goal_y - v_y) ** 2)

# -------- Main Code ----------

# add random obstacle to environment
if FIX_MAZE:
    env.add_default_obstacles()
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
            source = x, y
            break

for x in reversed(range(env.size)):
    for y in reversed(range(env.size)):
        if env.maze[x, y] == TwoDMaze.FREE_SPACE:
            goal = x, y
            break

# add source and goal to environment
env.add_source(source)
env.add_goal(goal)

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
        path.append(v)

    env.add_path(path)

env.plot()


# -------- Legacy Code ----------
