import time
import math

from robotics_algorithm.env.continuous_world_2d import TwoDWorldDiffDrive
from robotics_algorithm.planning import HybridAStar

# Initialize environment
env = TwoDWorldDiffDrive(discrete_action=True)  # use discrete action for Hybrid A*

# -------- Settings ------------
FIX_MAZE = True


# -------- Helper Functions -------------
def heuristic_func(state, goal):
    # simply the distance between v and goal
    v_x, v_y, _ = state
    goal_x, goal_y, _ = goal
    return math.sqrt((goal_x - v_x) ** 2 + (goal_y - v_y) ** 2)


def state_key_func(state):
    return env.calc_state_key(state)


# -------- Main Code ----------
env.reset(random_env=not FIX_MAZE)
start = env.start_state
goal = env.goal_state
env.render()

# initialize planner
planner = HybridAStar(env, heuristic_func, state_key_func)

# run path planner
start_time = time.time()
res, shortest_path, shortest_path_len = planner.run(start, goal)
end_time = time.time()
print("TestHybridAStar, takes {} seconds".format(end_time - start_time))

if not res:
    print("TestHybridAStar, no path is available!")
else:
    print("TestHybridAStar, found path of len {}".format(shortest_path_len))
    # visualize path
    path = []
    for v in shortest_path:
        path.append(v)

    env.add_action_path(path)

env.render()
