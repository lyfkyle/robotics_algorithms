import numpy as np
import matplotlib.pyplot as plt

from robotics_algorithm.env.frozen_lake import FrozenLake
from robotics_algorithm.learning.reinforcement_learning.q_learning import QLearning
from robotics_algorithm.utils.math_utils import smooth
from robotics_algorithm.utils.mdp_utils import make_greedy_policy

env = FrozenLake(dense_reward=True)
state, _ = env.reset()
env.render()

# Plan
learner = QLearning(env)
Q, _ = learner.run(num_episodes=10000)
# Plot cumulative rewards over episodes
episodes, learning_curve = learner.get_learning_curve()
plt.plot(episodes, smooth(learning_curve, weight=0.95), label="mc")
plt.ylabel("episode_reward")
plt.xlabel("episodes")
plt.show()
print(Q)

# Execute
policy = make_greedy_policy(Q, env.action_space.size)  # after learning, switch to use greedy policy
state, _ = env.reset()
path = []
while True:
    # choose action according to epsilon-greedy policy
    action_probs = policy(state)
    action = np.random.choice(env.action_space.get_all(), p=action_probs)  # choose action
    next_state, reward, term, trunc, info = env.step(action)

    print(state, action, next_state, reward)
    env.render()

    path.append(state)
    state = next_state

    if term or trunc:
        break

env.render()