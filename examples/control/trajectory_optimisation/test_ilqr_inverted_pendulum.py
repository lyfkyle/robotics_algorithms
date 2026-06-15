import numpy as np
import matplotlib.pyplot as plt

from robotics_algorithm.control.trajectory_optimisation.ilqr import iLQR
from robotics_algorithm.env.inverted_pendulum import InvertedPendulumEnv


env = InvertedPendulumEnv(mode='swing_up')
env.reset()

start = np.array([3.0, 0.0])  # start from downward position
goal = np.array([0.0, 0.0])
env.goal_state = goal
env.goal_action = np.zeros(1)

# For trajectory optimization, the most important parameter is the horizon and initial path guess
optimizer = iLQR(env, horizon=400, max_iter=100)
# Here we use  zero action as the initial action guess. This should produce a one-swing solution
initial_action_path = np.zeros((optimizer.horizon, env.action_space.state_size))
# * Below will create a two-swing solution
# initial_action_path = 5 * np.sin(np.linspace(0, 3 * np.pi, optimizer.horizon)).reshape(-1, 1)
state_path, action_path = optimizer.run(start, goal, initial_action_path)

print('start state:', state_path[0])
print('goal state:', state_path[-1])
print('first action:', action_path[0])

# Open loop execution of the optimized trajectory in the environment
env.cur_state = start.copy()
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
axes[0].plot(state_path[:, 0], label='Imagined', marker='o', markersize=4)
axes[0].plot(actual_state_path[:, 0], label='Actual', marker='s', markersize=4)
axes[0].set_xlabel('Time Step')
axes[0].set_ylabel('Angle (rad)')
axes[0].set_title('Angle: Imagined vs Actual')
axes[0].legend()
axes[0].grid(True)

# Angular velocity plot
axes[1].plot(state_path[:, 1], label='Imagined', marker='o', markersize=4)
axes[1].plot(actual_state_path[:, 1], label='Actual', marker='s', markersize=4)
axes[1].set_xlabel('Time Step')
axes[1].set_ylabel('Angular Velocity (rad/s)')
axes[1].set_title('Angular Velocity: Imagined vs Actual')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show(block=True)
