import sys
sys.path.append('../../environment/')
sys.path.append('../')

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import random
import time
import math

from two_d_maze import TwoDMaze
from rrt_connect import RRTConnect

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

def expand(v_source, v_goal, delta):
    v_source_x, v_source_y = v_source
    v_goal_x, v_goal_y = v_goal

    # expand from v_source to v_goal by delta
    theta = math.atan2(v_goal_y - v_source_y, v_goal_x - v_source_x)
    v_x = int(v_source_x + delta * math.cos(theta))
    v_y = int(v_source_y + delta * math.sin(theta))
    v = (v_x, v_y)

    # if expand too little
    if v == v_source:
        v = v_goal

    # if expand too much
    if v_x < 0 or v_x >= env.size or v_y < 0 or v_y > env.size:
        v = v_goal

    clear = check_clear(v)

    if not clear:
        return None

    link, _ = check_link(v_source, v)
    if not link:
        return None

    return v

def check_link(v1, v2):
    local_path = compute_local_path(v1, v2)
    if local_path is None:
        return False, 0
    else:
        return True, len(local_path)

def compute_local_path(v1, v2):
    v1_x, v1_y = v1
    v2_x, v2_y = v2

    local_path = []
    path_exist = True

    # try x first then y
    if v1_x >= v2_x:
        for x in range(v1_x, v2_x-1, -1):
            if env.maze[x, v1_y] == env.OBSTACLE:
                path_exist = False
                break

            local_path.append((x, v1_y))
    else:
        for x in range(v1_x, v2_x+1):
            if env.maze[x, v1_y] == env.OBSTACLE:
                path_exist = False
                break

            local_path.append((x, v1_y))

    if v1_y >= v2_y:
        for y in range(v1_y-1, v2_y-1, -1):
            if env.maze[v2_x, y] == env.OBSTACLE:
                path_exist = False
                break

            local_path.append((v2_x, y))
    else:
        for y in range(v1_y+1, v2_y+1):
            if env.maze[v2_x, y] == env.OBSTACLE:
                path_exist = False
                break

            local_path.append((v2_x, y))

    if path_exist:
        return local_path

    path_exist = True
    local_path.clear()
    # try y first then x
    if v1_y >= v2_y:
        for y in range(v1_y, v2_y-1, -1):
            if env.maze[v1_x, y] == env.OBSTACLE:
                path_exist = False
                break

            local_path.append((v1_x, y))
    else:
        for y in range(v1_y, v2_y+1):
            if env.maze[v1_x, y] == env.OBSTACLE:
                path_exist = False
                break

            local_path.append((v1_x, y))

    if v1_x >= v2_x:
        for x in range(v1_x-1, v2_x-1, -1):
            if env.maze[x, v2_y] == env.OBSTACLE:
                path_exist = False
                break

            local_path.append((x, v2_y))
    else:
        for x in range(v1_x+1, v2_x+1):
            if env.maze[x, v2_y] == env.OBSTACLE:
                path_exist = False
                break

            local_path.append((x, v2_y))

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

# add source and goal to environment
source = compute_source()
goal = compute_goal()
env.add_source(source)
env.add_goal(goal)

# source = (24, 6)
# goal = (23, 5)
# local_path = compute_local_path(source, goal)
# env.add_source(source)
# env.add_goal(goal)
# env.add_path(local_path)
# env.plot()


# initialize planner
my_path_planner = RRTConnect(number_of_samples = 1000) # 1000 samples out of total 2500 vertex.

# run path planner
start_time = time.time()
res, shortest_path, shortest_path_len = my_path_planner.run(source, goal, sample_vertex, expand, delta=1)
end_time = time.time()
print("TestRRT, online takes {} seconds".format(end_time - start_time))

# visualize tree
tree = my_path_planner.get_tree()
for vertex in tree:
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
    print("TestRRT, no path is available!")
else:
    # visualize path
    path = [source]
    v1 = source
    for v2 in shortest_path[1:]:
        # print(v1, v2)
        local_path = compute_local_path(v1, v2)
        if local_path is None:
            print("!!! this should not happen !!!")
            break
        else:
            path += local_path[1:] # ignore source of local path

        v1 = v2

    env.add_path(path)
    print("TestRRT, found path of len {}".format(len(path)))

env.plot()
