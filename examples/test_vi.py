import numpy as np

# from robotics_algorithm.env.windy_gridworld import WindyGridWorld
from robotics_algorithm.env.cliff_walking import CliffWalking
from robotics_algorithm.planning import ValueIteration


env = CliffWalking()
env.reset()
env.render()

vi = ValueIteration()
Q, policy = vi.run(env)

state = env.reset()
path = []
while True:
    # choose action according to epsilon-greedy policy
    action_probs = policy(state)
    action = np.random.choice(env.action_space, p=action_probs)  # choose action
    next_state, reward, term, trunc, info = env.step(action)

    print(state)
    print(action)

    path.append(state)
    state = next_state

    if term or trunc:
        break

env.add_path(path)
env.render()

# env = WindyGridWorld()
# Q, policy = vi.run(env)

# state = env.reset()
# path = []
# while True:
#     # choose action according to epsilon-greedy policy
#     action_probs = policy(state)
#     action = np.random.choice(env.actions, p=action_probs)  # choose action
#     next_state, reward, done, _ = env.step(action)

#     path.append(state)
#     state = next_state

#     print(state)
#     print(action)

#     if done:
#         break

# env.add_path(path)
# env.render()
