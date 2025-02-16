import json
import os.path as osp
import numpy as np

from robotics_algorithm.env.continuous_2d.diff_drive_2d_control import DiffDrive2DControl
from robotics_algorithm.control.optimal_control.convex_mpc import ConvexMPC

CUR_DIR = osp.join(osp.dirname(osp.abspath(__file__)))

PATH_DT = 0.1
with open(osp.join(CUR_DIR, 'example_path.json'), 'r') as f:
    shortest_path = json.load(f)
env = DiffDrive2DControl()
env.reset(shortest_path)
env.interactive_viz = True

# initialize controller
controller = ConvexMPC(env, horizon=20)
state = env.cur_state
ref_action = np.zeros(env.action_space.state_size)
env.render()

# run controller
while True:
    cur_ref_pose = env.get_cur_lookahead_state()
    controller.set_ref_state_action(cur_ref_pose, ref_action)

    state_error = env.cur_state - cur_ref_pose
    action_error = controller.run(state_error)
    action = ref_action + action_error

    next_state, reward, term, trunc, info = env.step(action)

    print(state, action, next_state, reward, term, trunc)
    env.render()

    state = next_state

    if term or trunc:
        break