import numpy as np

from robotics_algorithm.control.trajectory_optimisation.direct_collocation import DirectCollocation
from robotics_algorithm.env.inverted_pendulum import InvertedPendulumEnv


env = InvertedPendulumEnv()
env.reset()

start = np.array([1.0, 0.0])
goal = np.array([0.0, 0.0])
env.goal_state = goal
env.goal_action = np.zeros(1)

optimizer = DirectCollocation(env, horizon=10)
success, state_path, cost = optimizer.run(start, goal)

print('success:', success)
print('cost:', cost)
print('start state:', state_path[0])
print('goal state:', state_path[-1])
print('first action:', optimizer.action_path[0])

env.cur_state = start.copy()
env.render()

for action in optimizer.action_path:
    next_state, reward, term, trunc, info = env.step(action)
    print(env.cur_state, action, reward, term, trunc, info)
    env.render()

    if term or trunc:
        break
