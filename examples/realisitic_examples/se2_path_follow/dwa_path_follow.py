import json
import math
import os.path as osp

from se2_dwa import SE2PathFollowDWA
from env import DiffDriveSE2PathFollow

CUR_DIR = osp.join(osp.dirname(osp.abspath(__file__)))

# This path is computed using Hybrid A*
waypoints = [
    [1.0, 1.0, 0.0],  # Start position
    [3.0, 1.0, 0.0],  # Move forward 1 meter
    [3.0, 1.0, math.pi / 2],  # Rotate in place 90 degrees
    [3.0, 3.0, math.pi / 2],  # Move forward 1 meter in the new direction
    [3.0, 3.0, math.pi],  # Move forward 1 meter in the new direction
    [5.0, 3.0, math.pi],  # Move forward 1 meter in the new direction
]

# Initialize environment
env = DiffDriveSE2PathFollow(lookahead_index=100)
dense_waypoints = env.discretize_se2_trajectory(waypoints, resolution=0.005)
print(dense_waypoints)
env.reset(dense_waypoints, empty=True)
env.start_state = [1.0, 1.2, 0.0]
env.cur_state = [1.0, 1.2, 0.0]

controller = SE2PathFollowDWA(env, min_lin_vel=-0.3, max_lin_vel=0.3, lin_vel_samples=7, ang_vel_samples=10)

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
