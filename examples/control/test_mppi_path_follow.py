import time
import math
import json
import os.path as osp

from robotics_algorithm.env.continuous_2d.diff_drive_2d_control import DiffDrive2DControl
from robotics_algorithm.control.mppi import MPPI

CUR_DIR = osp.join(osp.dirname(osp.abspath(__file__)))

# This path is computed using Hybrid A*
PATH_DT = 0.1
with open(osp.join(CUR_DIR, 'exmaple_path.json'), 'r') as f:
    shortest_path = json.load(f)

# Initialize environment
env = DiffDrive2DControl(action_dt=PATH_DT)
env.reset(random_env=False)
env.set_ref_path(shortest_path)

controller = MPPI(env, action_mean=[0.25, 0], action_std=[0.25, math.radians(30)])

# debug
env.interactive_viz = True

# run controller
state = env.start_state
path = [state]
while True:
    action = controller.run(state)

    # visualize local plan
    local_plan = [state]
    best_actions = controller.prev_actions.tolist()  # debug
    new_state = env.sample_state_transition(state, action)[0]
    for future_action in best_actions:
        new_state = env.sample_state_transition(new_state, future_action)[0]
        local_plan.append(new_state)
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
