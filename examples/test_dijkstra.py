import time
import math

from robotics_algorithm.env.grid_world import DeterministicGridWorld
from robotics_algorithm.planning import Dijkstra


# Initialize environment
env = DeterministicGridWorld()

# -------- Settings ------------
FIX_MAZE = True


# -------- Helper Functions -------------

# -------- Main Code ----------

# add random obstacle to environment
env.reset(random_env=not FIX_MAZE)
env.render()

# initialize planner
planner = Dijkstra(env)

# run path planner
start = env.start_state
goal = env.goal_state
start_time = time.time()
res, shortest_path, shortest_path_len = planner.run(start, goal)
end_time = time.time()
print("TestDijkstra, takes {} seconds".format(end_time - start_time))

if not res:
    print("TestDijkstra, no path is available!")
else:
    print("TestDijkstra, found path of len {}".format(shortest_path_len))
    # visualize path
    env.add_path(shortest_path)

env.render()
