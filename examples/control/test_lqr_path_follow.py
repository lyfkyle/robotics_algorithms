import time
import math
import json
import os.path as osp

from robotics_algorithm.env.continuous_2d.diff_drive_2d_control import DiffDrive2DControlRelative
from robotics_algorithm.control.lqr import LQR

CUR_DIR = osp.join(osp.dirname(osp.abspath(__file__)))

# This path is computed using Hybrid A*
PATH_DT = 0.1
with open(osp.join(CUR_DIR, 'example_path.json'), 'r') as f:
    shortest_path = json.load(f)
env = DiffDrive2DControlRelative()
env.reset(shortest_path)

# initialize controller
controller = LQR(env, horizon=100, discrete_time=True, solve_by_iteration=True)

# run controller
state = env.start_state
path = [state]
while True:
    # print(state, env._cur_ref_state, env._env_impl.cur_state)

    action = controller.run(state)

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

    path.append(next_state)
    state = next_state

    if term or trunc:
        break
