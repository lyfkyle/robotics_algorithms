import time
import numpy as np

from robotics_algorithm.env.continuous_world_2d import OmniDriveTwoDEnv
from robotics_algorithm.planning import RRTStar

# Initialize environment
env = OmniDriveTwoDEnv()

# -------- Settings ------------
FIX_MAZE = True


# -------- Helper Functions -------------
def sample_func(env):
    random_state = env.random_state()
    return tuple(random_state)


def is_state_valid(env, state):
    return env.is_state_valid(state)


def is_edge_valid(env, state1, state2):
    return env.is_state_transition_valid(state1, state2)


def vertex_expand_func(env, state1, state2):
    path = env.extend(state1, state2)
    path_len = np.linalg.norm(np.array(path[-1]) - np.array(state1))
    return path[-1], path_len


def distance_func(env, state1, state2):
    return np.linalg.norm(np.array(state1) - np.array(state2))


# -------- Main Code ----------
# add default obstacles
env.reset(random_env=not FIX_MAZE)
env.render()

# initialize planner
planner = RRTStar(env, sample_func, vertex_expand_func, is_edge_valid, distance_func, num_of_samples=500)

# run path planner
start = env.start_state
goal = env.goal_state
start_time = time.time()
res, shortest_path, shortest_path_len = planner.run(start, goal)
end_time = time.time()
print("TestRRTStar, online takes {} seconds".format(end_time - start_time))

# visualize tree
tree = planner.get_tree()
for state in tree.nodes:
    if state != goal and state != start:
        env.add_state_samples(state)

if not res:
    print("TestRRTStar, no path is available!")
else:
    # visualize path
    env.add_action_path(shortest_path)
    print("TestRRTStar, found path of len {}".format(shortest_path_len))

env.render()
