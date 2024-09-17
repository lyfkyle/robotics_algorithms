import numpy as np

from robotics_algorithm.env.frozen_lake import FrozenLake
from robotics_algorithm.planning import PolicyTreeSearch


env = FrozenLake(dense_reward=True)
state = env.reset(random_env=False)
env.render()

# Plan
planner = PolicyTreeSearch(env, max_depth=5)

# Execute
path = []
while True:
    # choose action according to the computed policy
    action = planner.run(state)
    next_state, reward, term, trunc, info = env.step(action)
    env.render()

    print(state)
    print(action)
    print(reward)

    path.append(state)
    state = next_state

    if term or trunc:
        break

# env.add_path(path)
env.render()
