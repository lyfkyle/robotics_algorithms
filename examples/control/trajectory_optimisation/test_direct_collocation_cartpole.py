import numpy as np
import matplotlib.pyplot as plt

from robotics_algorithm.control.trajectory_optimisation.direct_collocation import DirectCollocation
from robotics_algorithm.env.cartpole_balance import CartPoleEnv

# Use quadratic reward for trajectory optimization
# Increase dt to balance for longer time.
env = CartPoleEnv(quadratic_reward=True, dt=0.02)
env.reset()

start = env.cur_state.copy()
goal = env.goal_state.copy()

# For trajectory optimization, the most important parameter is the horizon, the initial path guess and the cost function weights.
# * Due to large number of decision variables in direct collocation, we use shorter horizon than direct shooting.
optimizer = DirectCollocation(env, horizon=25, path_cost_w=1.0, dynamics_cost_w=1e9)
# Here we use linearly interpolated path between start and goal as the initial path guess, and zero action as the initial action guess.
initial_state_path = np.linspace(start, goal, optimizer.horizon + 1)
initial_action_path = np.zeros((optimizer.horizon, env.action_space.state_size))
success, state_path, action_path, initial_cost, final_cost = optimizer.run(
    start, goal, initial_state_path, initial_action_path
)

print('success:', success)
print('initial cost:', initial_cost)
print('final cost:', final_cost)
print('start state:', state_path[0])
print('goal state:', state_path[-1])
print('first action:', action_path[0])

# Open loop execution of the optimized trajectory in the environment
env.cur_state = start.copy()
env.render()

actual_state_path = [start.copy()]
for action in action_path:
    next_state, reward, term, trunc, info = env.step(action)
    actual_state_path.append(env.cur_state.copy())
    print(env.cur_state, action, reward, term, trunc, info)
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
