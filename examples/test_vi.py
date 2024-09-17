import numpy as np

from robotics_algorithm.env.windy_gridworld import WindyGridWorld
from robotics_algorithm.env.cliff_walking import CliffWalking
from robotics_algorithm.planning import ValueIteration

planner = ValueIteration()

env = CliffWalking()
state = env.reset()
env.render()

# Plan
Q, policy = planner.run(env)

# Execute
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


env = WindyGridWorld()
state = env.reset()
env.render()

# Plan
Q, policy = planner.run(env)

# Execute
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
