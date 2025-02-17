import time
import numpy as np

from robotics_algorithm.env.base_env import BaseEnv
from robotics_algorithm.env.continuous_2d.omni_2d_planning import OmniDrive2DPlanning
from robotics_algorithm.planning.motion_planning.rrt import RRT

# Initialize environment
env = OmniDrive2DPlanning()

# -------- Settings ------------
FIX_MAZE = True


# -------- Helper Functions -------------
def sample_func(env: BaseEnv):
    random_state = env.random_state()
    return random_state


def is_state_valid(env: BaseEnv, state: np.ndarray):
    return env.is_state_valid(state)


def is_edge_valid(env: BaseEnv, state1: np.ndarray, state2: np.ndarray):
    return env.is_state_transition_valid(state1, None, state2)


def vertex_expand_func(env: BaseEnv, state1: np.ndarray, state2: np.ndarray):
    path = env.extend(np.array(state1), np.array(state2))  # TODO refactor.
    path_len = np.linalg.norm(np.array(path[-1]) - np.array(state1))
    return path[-1], path_len


# -------- Main Code ----------
# add default obstacles
env.reset(random_env=not FIX_MAZE)
env.render()

# initialize planner
planner = RRT(env, sample_func, vertex_expand_func, num_of_samples=500)

# run path planner
start = env.start_state
goal = env.goal_state
start_time = time.time()
res, shortest_path, shortest_path_len = planner.run(start, goal)
end_time = time.time()
print('TestRRT, online takes {} seconds'.format(end_time - start_time))

# visualize tree
tree = planner.get_tree()
for state in tree.nodes:
    if state != tuple(goal.tolist()) and state != tuple(start.tolist()):
        env.add_state_samples(state)

if not res:
    print('TestRRT, no path is available!')
else:
    # visualize path
    env.add_state_path(shortest_path, interpolate=True)
    print('TestRRT, found path of len {}'.format(shortest_path_len))

env.render()
