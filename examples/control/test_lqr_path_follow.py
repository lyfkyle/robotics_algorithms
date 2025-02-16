import time
import math
import json
import os.path as osp
import numpy as np

from robotics_algorithm.env.continuous_2d.diff_drive_2d_control import DiffDrive2DControl
from robotics_algorithm.control.lqr import LQR

CUR_DIR = osp.join(osp.dirname(osp.abspath(__file__)))

# This path is computed using Hybrid A*
PATH_DT = 0.1
with open(osp.join(CUR_DIR, 'example_path.json'), 'r') as f:
    shortest_path = json.load(f)
env = DiffDrive2DControl()
env.reset(shortest_path, empty=True)
env.interactive_viz = True

# initialize controller
controller = LQR(env, horizon=100, discrete_time=True, solve_by_iteration=True)

# Local stabilization
# state = [0.5, 0.5, 1.0]
# env.start_state = state
# env.cur_state = state
# goal_state = [2.0, 2.0, 0.0]
# # env.start_state = state
# # env.cur_state = state
# env.goal_state = goal_state
# # env._env_impl.start_state[2] = 2
# # env._env_impl.cur_state[2] = 2
# ref_action = [0.0, 0.0]
# path = [state]

# while True:
#     # print(state, env._cur_ref_state, env._env_impl.cur_state)

#     state_error = (np.array(env.cur_state) - np.array(env.goal_state)).tolist()
#     action_error = controller.run(state_error, ref_action)
#     action = (np.array(ref_action) + np.array(action_error)).tolist()

#     # print(action)
#     # visualize local plan
#     # local_plan = [state]
#     # best_actions = controller.prev_actions.tolist()  # debug
#     # new_state = env.sample_state_transition(state, action)[0]
#     # for future_action in best_actions:
#     #     new_state = env.sample_state_transition(new_state, future_action)[0]
#     #     local_plan.append(new_state)
#     # env.set_local_plan(local_plan)

#     next_state, reward, term, trunc, info = env.step(action)

#     print(state, action, next_state)
#     env.render()

#     path.append(next_state)
#     state = next_state

#     if term or trunc:
#         break

# Time-varying local trajectory stabilization (tracking)
state = env.cur_state
ref_action = [0.0, 0.0]
env.render()

while True:
    # print(state, env._cur_ref_state, env._env_impl.cur_state)

    cur_ref_pose = env.get_cur_lookahead_state()
    state_error = (np.array(env.cur_state) - np.array(cur_ref_pose)).tolist()
    action_error = controller.run(state_error, ref_action)
    action = (np.array(ref_action) + np.array(action_error)).tolist()

    # print(action)
    # visualize local plan
    # local_plan = [state]
    # best_actions = controller.prev_actions.tolist()  # debug
    # new_state = env.sample_state_transition(state, action)[0]
    # for future_action in best_actions:
    #     new_state = env.sample_state_transition(new_state, future_action)[0]
    #     local_plan.append(new_state)
    # env.set_local_plan(local_plan)

    next_state, reward, term, trunc, info = env.step(action)

    print(state, action, next_state)
    env.render()

    state = next_state

    if term or trunc:
        break