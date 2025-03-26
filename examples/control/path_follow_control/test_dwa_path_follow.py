import json
import math
import os.path as osp

from robotics_algorithm.control.path_follow_control.dwa import DWA
from robotics_algorithm.env.continuous_2d.diff_drive_2d_control import DiffDrive2DControl

CUR_DIR = osp.join(osp.dirname(osp.abspath(__file__)))

# This path is computed using Hybrid A*
PATH_DT = 0.1
with open(osp.join(CUR_DIR, 'example_path.json'), 'r') as f:
    shortest_path = json.load(f)

# Initialize environment
env = DiffDrive2DControl()
env.reset(shortest_path, empty=False)

controller = DWA(env)

# debug
env.interactive_viz = True

# run controller
state = env.start_state
path = [state]
while True:
    action = controller.run(state)

    # visualize local plan
    local_plan = controller.best_traj  # debug
    env.set_local_plan(local_plan)

    next_state, reward, term, trunc, info = env.step(action)
    print(state, action, next_state, reward)

    # nearest_idx = env._get_nearest_waypoint_to_state(next_state)
    # print(shortest_path[nearest_idx])

    env.render()

    path.append(next_state)
    state = next_state

    if term or trunc:
        break
