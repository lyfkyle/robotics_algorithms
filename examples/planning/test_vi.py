import numpy as np

from robotics_algorithm.env.windy_grid_world import WindyGridWorld
from robotics_algorithm.env.cliff_walking import CliffWalking
from robotics_algorithm.planning import ValueIteration

env = CliffWalking()
state, _ = env.reset()
env.render()

# Plan
planner = ValueIteration(env)
Q, policy = planner.run()

# Execute
path = []
while True:
    # choose action according to epsilon-greedy policy
    action_probs = policy(state)
    action = np.random.choice(env.action_space.get_all(), p=action_probs)  # choose action
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
state, _ = env.reset()
env.render()

# Plan
planner = ValueIteration(env)
Q, policy = planner.run()

# Execute
path = []
while True:
    # choose action according to epsilon-greedy policy
    action_probs = policy(state)
    action = np.random.choice(env.action_space.get_all(), p=action_probs)  # choose action
    next_state, reward, term, trunc, info = env.step(action)

    print(state)
    print(action)

    path.append(state)
    state = next_state

    if term or trunc:
        break

env.add_path(path)
env.render()
