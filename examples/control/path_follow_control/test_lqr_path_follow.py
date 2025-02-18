import json
import os.path as osp
import numpy as np

from robotics_algorithm.env.continuous_2d.diff_drive_2d_control import DiffDrive2DControl
from robotics_algorithm.control.optimal_control.lqr import LQR

CUR_DIR = osp.join(osp.dirname(osp.abspath(__file__)))

# This path is computed using Hybrid A*
PATH_DT = 0.1
with open(osp.join(CUR_DIR, 'example_path.json'), 'r') as f:
    shortest_path = json.load(f)
env = DiffDrive2DControl(lookahead_dist=5, has_kinematics_constraint=False)  # ! LQR can't deal with constraints
env.reset(shortest_path)
env.interactive_viz = True

# initialize controller
controller = LQR(env, horizon=100, discrete_time=True, solve_by_iteration=True)

state = env.cur_state
# ! Here we set reference action to be zero velocity, implying that each time LQR is trying to bring robot to stop
# ! at the lookahead state. Optionally, if planned path contains velocity information, reference action can be read
# ! from the path.
ref_action = np.zeros(env.action_space.state_size)
env.render()

# Following reference path falls into category of iterative time-varying local trajectory stabilization.
# In this case, we iteratively use LQR to stabilize around a reference state that is changing as the robot moves.
# In each iteration, LQR is solved with linearized dynamics around current reference state and reference action.
while True:
    # Set current reference state and action to obtain new linearized dynamics
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