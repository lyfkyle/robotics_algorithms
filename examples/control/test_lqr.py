from robotics_algorithm.env.continuous_world_1d import DoubleIntegratorEnv
from robotics_algorithm.control.lqr import LQR

# Test 1
# Continuous time model
#  - Since we have not incorporated state estimation, set noises to small value as if the env is deterministic and
#    fully-observable
env = DoubleIntegratorEnv(use_discrete_time_model=False, observation_noise_std=1e-5, state_transition_noise_std=1e-5)

env.reset()
print("cur_state: ", env.start_state)
env.render()

# initialize controller
controller = LQR(env, discrete_time=False)

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
# discrete time model
env = DoubleIntegratorEnv(use_discrete_time_model=True, observation_noise_std=0, state_transition_noise_std=0)

env.reset()
print("cur_state: ", env.start_state)
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

# Test 3
# Now test solve by iteration for finite-horizon case
env = DoubleIntegratorEnv(use_discrete_time_model=True, observation_noise_std=0, state_transition_noise_std=0)

env.reset()
print("cur_state: ", env.start_state)
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
