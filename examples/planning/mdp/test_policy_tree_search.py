import numpy as np

from robotics_algorithm.env.classic_mdp.frozen_lake import FrozenLake
from robotics_algorithm.planning.mdp.policy_tree_search import PolicyTreeSearch


env = FrozenLake(dense_reward=True)
state, _ = env.reset()
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

    print(state, action, next_state, reward)

    path.append(state)
    state = next_state

    if term or trunc:
        break

# env.add_path(path)
env.render()
