import json
import os.path as osp
import numpy as np

from robotics_algorithm.env.continuous_2d.diff_drive_2d_control import DiffDrive2DControl
from robotics_algorithm.control.convex_mpc import ConvexMPC

CUR_DIR = osp.join(osp.dirname(osp.abspath(__file__)))

# Time-varying local trajectory stabilization (tracking)
# discrete time model
PATH_DT = 0.1
with open(osp.join(CUR_DIR, 'example_path.json'), 'r') as f:
    shortest_path = json.load(f)
env = DiffDrive2DControl()
env.reset(shortest_path, empty=True)
env.interactive_viz = True

# initialize controller
controller = ConvexMPC(env)
state = env.cur_state
ref_action = np.array([0.0, 0.0])
env.render()

# run controller
while True:
    cur_ref_pose = env.get_cur_lookahead_state()
    state_error = env.cur_state - cur_ref_pose
    action_error = controller.run(state_error, ref_action)
    action = ref_action + action_error

    next_state, reward, term, trunc, info = env.step(action)

    print(state, action, next_state, reward, term, trunc)
    env.render()

    state = next_state

    if term or trunc:
        break