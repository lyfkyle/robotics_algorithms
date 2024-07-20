import numpy as np

# from robotics_algorithm.env.grid_world import GridWorld
from robotics_algorithm.env.cliff_walking import CliffWalking
from robotics_algorithm.planning import MCTS


env = CliffWalking(dense_reward=True)  # For online tree search, dense reward needs to be enabled.
state = env.reset()
env.render()

planner = MCTS(env)

# Plan
# policy = planner.run(env, max_depth=5)

# Execute
path = []
while True:
    # choose action according to epsilon-greedy policy
    action = planner.run(state)
    # action = np.random.choice(env.action_space, p=action_probs)  # choose action
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