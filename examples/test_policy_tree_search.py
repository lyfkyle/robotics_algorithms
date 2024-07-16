import numpy as np

from robotics_algorithm.env.grid_world import GridWorld
from robotics_algorithm.planning import PolicyTreeSearch

planner = PolicyTreeSearch()

env = GridWorld()
state = env.reset(random_env=False)
env.render()

# Plan
policy = planner.run(env, max_depth=5)

# Execute
path = []
while True:
    # choose action according to epsilon-greedy policy
    action_probs = policy(state)
    action = np.random.choice(env.action_space, p=action_probs)  # choose action
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