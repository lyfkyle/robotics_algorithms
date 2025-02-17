import time
import math
import numpy as np

from robotics_algorithm.env.discrete_world_2d import GridWorldMaze
from robotics_algorithm.planning.path_planning.a_star import AStar

# Initialize environment
env = GridWorldMaze()

# -------- Settings ------------
FIX_MAZE = True


# -------- Helper Functions -------------
def heuristic_func(v, goal):
    # simply the distance between v and goal
    v_x, v_y = v
    goal_x, goal_y = goal
    return math.sqrt((goal_x - v_x) ** 2 + (goal_y - v_y) ** 2)


def state_key_func(state: np.ndarray):
    return tuple(state.tolist())


# -------- Main Code ----------

# add random obstacle to environment
env.reset(random_env=not FIX_MAZE)
env.render()


# initialize planner
planner = AStar(env, heuristic_func, state_key_func)

# run path planner
start = env.start_state
goal = env.goal_state
start_time = time.time()
res, shortest_path, shortest_path_len = planner.run(start, goal)
end_time = time.time()
print("TestAStar, takes {} seconds".format(end_time - start_time))

if not res:
    print("TestAStar, no path is available!")
else:
    print("TestAStar, found path of len {}".format(shortest_path_len))
    # visualize path
    env.add_path(shortest_path)

env.render()
