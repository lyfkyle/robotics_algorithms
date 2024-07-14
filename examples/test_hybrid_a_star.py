import time
import math

from robotics_algorithm.env.two_d_maze import TwoDMazeDiffDrive
from robotics_algorithm.planning import HybridAStar

# Initialize environment
env = TwoDMazeDiffDrive()

# -------- Settings ------------
FIX_MAZE = True


# -------- Helper Functions -------------
def heuristic_func(state, goal):
    # simply the distance between v and goal
    v_x, v_y, _ = state
    goal_x, goal_y, _ = goal
    return math.sqrt((goal_x - v_x) ** 2 + (goal_y - v_y) ** 2)


def state_key_func(state):
    return (int(state[0] // 0.25), int(state[1] // 0.25), int(state[2] // math.radians(30)))


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

    env.add_path(path)

env.render()
