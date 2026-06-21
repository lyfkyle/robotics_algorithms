import numpy as np
import matplotlib.pyplot as plt

from robotics_algorithm.control.trajectory_optimisation.ilqr import iLQR
from robotics_algorithm.env.planar_quadrotor_hover import PlanarQuadrotorHoverEnv


env = PlanarQuadrotorHoverEnv(
    hover_pos=0.5, hover_height=1.0, quadratic_reward=True, term_if_constraints_violated=False
)
env.reset()
print('cur_state: ', env.cur_state)

start = env.cur_state.copy()
goal = env.goal_state.copy()

# For trajectory optimization, the most important parameter is the horizon and initial path guess
optimizer = iLQR(env, horizon=300, max_iter=100)
# Construct a good initial action path: ramp up to hover thrust, then maintain
initial_action_path = np.zeros((optimizer.horizon, env.action_space.state_size))
ramp_steps = 50  # Steps to ramp up to hover thrust
for i in range(optimizer.horizon):
    if i < ramp_steps:
        # Smoothly ramp up from 0 to hover thrust (with slight overshoot to rise)
        alpha = i / ramp_steps
        ramp_factor = 1.1  # 10% overshoot to ensure rising
        initial_action_path[i] = alpha * ramp_factor * env.goal_action
    else:
        # Maintain hover thrust after ramp
        initial_action_path[i] = env.goal_action

state_path, action_path = optimizer.run(start, initial_action_path)

print('start state:', state_path[0])
print('goal state:', state_path[-1])
print('first action:', action_path[0])
# Check for issues in trajectory
print('\n' + '=' * 60)
print('OPTIMIZED TRAJECTORY (RAW ANALYSIS)')
print('=' * 60)
print(f'Theta range: [{np.min(state_path[:, 2]):.4f}, {np.max(state_path[:, 2]):.4f}]')
print(f'State contains NaN: {np.any(np.isnan(state_path))}')
print(f'State contains inf: {np.any(np.isinf(state_path))}')
print(f'Action range: [{np.min(action_path):.4f}, {np.max(action_path):.4f}]')
print(f'Actions violate bounds: {np.any((action_path < env.action_space.low) | (action_path > env.action_space.high))}')
print(f'Action contains NaN: {np.any(np.isnan(action_path))}')
print(f'Action contains inf: {np.any(np.isinf(action_path))}')

input('execute?')

# Open loop execution of the optimized trajectory in the environment
env.render()

print(env.max_steps)

actual_state_path = [start.copy()]
for action in action_path:
    next_state, reward, term, trunc, info = env.step(action)
    actual_state_path.append(env.cur_state.copy())
    print(env.cur_state, action, reward, term, trunc, info, env.step_cnt)
    env.render()

    if term or trunc:
        break

# Plot imagined vs actual state paths
actual_state_path = np.array(actual_state_path)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Angle plot
axes[0].plot(actual_state_path[:, 0], label='Actual', marker='s', markersize=4)
axes[0].set_xlabel('Time Step')
axes[0].set_ylabel('Angle (rad)')
axes[0].set_title('Angle')
axes[0].legend()
axes[0].grid(True)

# Angular velocity plot
axes[1].plot(actual_state_path[:, 1], label='Actual', marker='s', markersize=4)
axes[1].set_xlabel('Time Step')
axes[1].set_ylabel('Angular Velocity (rad/s)')
axes[1].set_title('Angular Velocity')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show(block=True)
