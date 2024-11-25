import numpy as np
import matplotlib.pyplot as plt

from robotics_algorithm.env.frozen_lake import FrozenLake
from robotics_algorithm.learning.reinforcement_learning.q_learning import QLearning

env = FrozenLake(dense_reward=True)
state, _ = env.reset()
env.render()

# Plan
learner = QLearning(env)
Q, policy = learner.run(num_episodes=5000)

# Execute
state, _ = env.reset()
path = []
while True:
    # choose action according to epsilon-greedy policy
    action_probs = policy(state)
    action = np.random.choice(env.action_space.get_all(), p=action_probs)  # choose action
    next_state, reward, term, trunc, info = env.step(action)
    env.render()

    print(state)
    print(action)

    path.append(state)
    state = next_state

    if term or trunc:
        break

env.render()