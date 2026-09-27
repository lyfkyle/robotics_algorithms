"""Mobile manipulator end-effector trajectory following.

A mobile manipulator (differential-drive circular base + planar 2-link arm) must move its
end-effector along a figure-8 (lemniscate) loop. The whole-body trajectory (base motion + arm
motion) is optimized jointly using either direct shooting or direct collocation with L-BFGS.

- Direct shooting always produces dynamically feasible (and therefore non-holonomic-consistent)
  trajectories since actions are forward-simulated through the true kinematics.
- Direct collocation treats intermediate states as decision variables and enforces the dynamics
  (including the non-holonomic base constraint) via a quadratic penalty cost.
"""

import numpy as np
import matplotlib.pyplot as plt

from robotics_algorithm.control.trajectory_optimisation.direct_shooting import DirectShooting
from robotics_algorithm.env.continuous_2d.mobile_manipulator_eef_traj_follow import MMEefTrajFollowEnv


def _solve_2link_ik(target: np.ndarray, l1: float, l2: float) -> tuple[float, float]:
    """Solve planar 2-link IK for a base fixed at the origin with theta=0.

    Returns the elbow-up solution (q1, q2) such that the end-effector reaches `target`.
    """
    d = np.linalg.norm(target)
    assert abs(l1 - l2) <= d <= l1 + l2, f'target {target} is unreachable with l1={l1}, l2={l2}'

    cos_q2 = np.clip((d**2 - l1**2 - l2**2) / (2 * l1 * l2), -1.0, 1.0)
    q2 = np.arccos(cos_q2)  # elbow-up solution
    q1 = np.arctan2(target[1], target[0]) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))

    return q1, q2


HORIZON = 50
DT = 0.1

# Figure-8 (lemniscate) reference trajectory for the end-effector, in world frame.
CENTER = np.array([0.4, 0.0])
AMPLITUDE = 0.3
t = np.linspace(0, 2 * np.pi, HORIZON + 1)
ref_eef_traj = np.column_stack([CENTER[0] + AMPLITUDE * np.sin(t), CENTER[1] + 0.5 * AMPLITUDE * np.sin(2 * t)])

LINK1_LENGTH = 0.3
LINK2_LENGTH = 0.2

# Initialize the arm so the end-effector starts exactly on the reference path, with the base at
# the origin with zero heading.
q1_0, q2_0 = _solve_2link_ik(ref_eef_traj[0], LINK1_LENGTH, LINK2_LENGTH)
start_state = np.array([0.0, 0.0, 0.0, q1_0, q2_0])  # x, y, theta, q1, q2

env = MMEefTrajFollowEnv(start_state, ref_eef_traj, link1_length=LINK1_LENGTH, link2_length=LINK2_LENGTH, dt=DT)

optimizer = DirectShooting(env, horizon=HORIZON, path_cost_w=1.0, terminal_cost_w=1.0)
# Seed with a small non-zero initial guess: since the reference trajectory is a closed loop
# (goal == start), the zero-action guess sits exactly at the terminal-cost's non-differentiable
# point (||state - goal|| == 0), which makes L-BFGS-B's finite-difference gradient unstable.
rng = np.random.default_rng(0)
initial_action_path = rng.normal(scale=1e-3, size=(optimizer.horizon, env.action_space.state_size))
success, state_path, action_path, initial_cost, final_cost = optimizer.run(
    start_state, env.goal_state, initial_action_path
)
state_path = np.array(state_path)

print('success:', success)
print('initial cost:', initial_cost)
print('final cost:', final_cost)

# Compute the end-effector trajectory from the optimized state path.
eef_path = np.array([env.robot_model.eef_pos(s) for s in state_path])

# Interactively animate the optimized trajectory: the base and arm are redrawn at each time step,
# with the traveled path traced so far, so the trajectory sequence is visually clear.
plt.ion()
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

action_ax = axes[1]
action_ax.plot(action_path[:, 0], label='v (base linear vel)')
action_ax.plot(action_path[:, 1], label='omega (base angular vel)')
action_ax.plot(action_path[:, 2], label='q1_dot')
action_ax.plot(action_path[:, 3], label='q2_dot')
action_ax.set_xlabel('Time Step')
action_ax.set_ylabel('Action')
action_ax.set_title('Optimized Actions')
action_ax.legend()
action_ax.grid(True)

traj_ax = axes[0]
for i in range(len(state_path)):
    traj_ax.cla()
    traj_ax.plot(ref_eef_traj[:, 0], ref_eef_traj[:, 1], 'r--', label='Reference EEF trajectory')
    traj_ax.plot(eef_path[: i + 1, 0], eef_path[: i + 1, 1], 'b-', label='Optimized EEF trajectory')
    traj_ax.plot(state_path[: i + 1, 0], state_path[: i + 1, 1], 'k-', label='Base trajectory')
    env.render(traj_ax, state_path[i])

    traj_ax.set_xlabel('x (m)')
    traj_ax.set_ylabel('y (m)')
    traj_ax.set_title('Mobile Manipulator End-Effector Trajectory Following')
    traj_ax.legend()
    traj_ax.axis('equal')
    traj_ax.grid(True)

    plt.pause(0.05)

plt.ioff()
plt.tight_layout()
plt.show(block=True)
