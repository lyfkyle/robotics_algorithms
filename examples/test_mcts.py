import numpy as np

from robotics_algorithm.env.frozen_lake import FrozenLake
from robotics_algorithm.env.cliff_walking import CliffWalking
from robotics_algorithm.planning import MCTS

env = FrozenLake(dense_reward=True)  # For online tree search, dense reward needs to be enabled.
# env = CliffWalking(dense_reward=True)  # For online tree search, dense reward needs to be enabled.
state, _ = env.reset(random_env=False)
env.render()

planner = MCTS(env)

# Execute
path = []
while True:
    # Call tree search online
    action = planner.run(state)
    new_state, reward, term, trunc, info = env.step(action)

    print(state)
    print(action)
    print(reward)
    print(new_state)

    env.render()

    path.append(state)
    state = new_state

    if term or trunc:
        break

# env.add_path(path)
env.render()
