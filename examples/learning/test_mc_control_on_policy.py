import numpy as np
import matplotlib.pyplot as plt

from robotics_algorithm.env.classic_mdp.frozen_lake import FrozenLake
from robotics_algorithm.learning.reinforcement_learning.mc_control_on_policy import MCControlOnPolicy
from robotics_algorithm.utils.math_utils import smooth
from robotics_algorithm.utils.mdp_utils import make_greedy_policy

env = FrozenLake(dense_reward=True)
state, _ = env.reset()
env.render()

# Plan
learner = MCControlOnPolicy(env)
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
all_actions = env.action_space.get_all()
while True:
    # choose action according to epsilon-greedy policy
    action_probs = policy(state)
    action_idx = np.random.choice(np.arange(len(all_actions)), p=action_probs)  # choose action
    action = all_actions[action_idx]
    next_state, reward, term, trunc, info = env.step(action)

    print(state, action, next_state, reward)
    env.render()

    path.append(state)
    state = next_state

    if term or trunc:
        break

env.render()