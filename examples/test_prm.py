import time
import numpy as np

from robotics_algorithm.env.two_d_world import TwoDWorldOmni
from robotics_algorithm.planning import ProbabilisticRoadmap

# Initialize environment
env = TwoDWorldOmni()

# -------- Settings ------------
FIX_MAZE = True


# -------- Helper Functions -------------
def sample_func(env):
    random_state = env.sample_state()
    return tuple(random_state)


def is_state_valid(env, state):
    return env.is_state_valid(state)


def is_edge_valid(env, state1, state2):
    return env.is_state_transition_valid(state1, state2)


# -------- Main Code ----------
# add default obstacles
env.reset(random_env=not FIX_MAZE)
env.render()

# initialize planner
planner = ProbabilisticRoadmap(
    env, sample_func, is_state_valid, is_edge_valid, num_of_samples=100, num_neighbors=10
)  # 1000 samples out of total 2500 vertex.

# offline portion of PRM
start_time = time.time()
planner.compute_roadmap()
end_time = time.time()
print("TestPRM, offline takes {} seconds".format(end_time - start_time))

# run path planner
start_time = time.time()
start = env.start_state
goal = env.goal_state
res, shortest_path, shortest_path_len = planner.get_path(start, goal)
end_time = time.time()
print("TestPRM, online takes {} seconds".format(end_time - start_time))

# visualize roadmap
roadmap = planner.get_roadmap()
for state in roadmap.nodes:
    if state != goal and state != start:
        env.add_state_samples(state)

if not res:
    print("TestPRM, no path is available!")
else:
    # visualize path
    env.add_action_path(shortest_path)
    print("TestPRM, found path of len {}".format(shortest_path_len))

env.render()
