import numpy as np

from robotics_algorithm.env.continuous_1d.double_integrator_env import DoubleIntegratorEnv
from robotics_algorithm.control.optimal_control.lqr import LQR

# Test 1
# discrete time model, solve dare using scipy
env = DoubleIntegratorEnv(observation_noise_std=0, state_transition_noise_std=0)

env.reset()
print('cur_state: ', env.start_state)
env.render()

# initialize controller
controller = LQR(env, discrete_time=True)

# run controller
state = env.start_state
path = [state]
while True:
    action = controller.run(state)
    next_state, reward, term, trunc, info = env.step(action)

    print(state, action, next_state)

    path.append(next_state)
    state = next_state

    if term or trunc:
        break

env.add_path(path)
env.render()

# Test 2
# Solve dare by iteration for finite-horizon case
env = DoubleIntegratorEnv(observation_noise_std=0, state_transition_noise_std=0)

env.reset()
print('cur_state: ', env.start_state)
env.render()

# initialize controller
controller = LQR(env, discrete_time=True, horizon=2000, solve_by_iteration=True)

# run controller
state = env.start_state
path = [state]
while True:
    action = controller.run(state)
    next_state, reward, term, trunc, info = env.step(action)

    print(state, action, next_state)

    path.append(next_state)
    state = next_state

    if term or trunc:
        break

env.add_path(path)
env.render()
