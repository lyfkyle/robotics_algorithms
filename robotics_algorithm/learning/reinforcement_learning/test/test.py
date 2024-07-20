import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from mc_control_on_policy import MCControlOnPolicy
from sarsa_lamda import SARSALamda

from environment.windy_gridworld import WindyGridWorld
from environment.cliff_walking import CliffWalking

mc = MCControlOnPolicy()
sarsa_zero = SARSALamda(0)
# sarsa_one = SARSALamda(1)
sarsa_five = SARSALamda(5)
sarsa_ten = SARSALamda(10)

def get_success_rate(env, policy, num_of_attempt=100):
    success_cnt = 0
    for _ in range(num_of_attempt):
        state = env.reset()
        path = []
        while True:
            ## choose action according to epsilon-greedy policy
            action_probs = policy(state)
            # print(Q[state])
            action = np.argmax(action_probs)
            next_state, reward, done, goal_reached = env.step(action)

            path.append(state)
            # print(state)
            # print(action)
            # print(reward)

            state = next_state

            if goal_reached:
                success_cnt += 1

            if done:
                break

    return success_cnt / 100.0

env = WindyGridWorld()

EPISODE_NUM = 200

Q, policy = mc.learn(env, num_episodes = EPISODE_NUM, epsilon = 0.1)
episodes, learning_curve_1 = mc.get_learning_curve()

Q, policy = sarsa_zero.learn(env, num_episodes = EPISODE_NUM, epsilon = 0.1)
_, learning_curve_2 = sarsa_zero.get_learning_curve()

# Q, policy = sarsa_one.learn(env, num_episodes = 100, epsilon = 0.1)
# _, learning_curve_3 = sarsa_one.get_learning_curve()

Q, policy = sarsa_five.learn(env, num_episodes = EPISODE_NUM, epsilon = 0.1)
_, learning_curve_4 = sarsa_five.get_learning_curve()

Q, policy = sarsa_ten.learn(env, num_episodes = EPISODE_NUM, epsilon = 0.1)
_, learning_curve_5 = sarsa_ten.get_learning_curve()

plt.plot(episodes, learning_curve_1, label="mc")
plt.plot(episodes, learning_curve_2, label="sarsa_zero")
# plt.plot(episodes, learning_curve_3, label="sarsa_one")
plt.plot(episodes, learning_curve_4, label="sarsa_five")
plt.plot(episodes, learning_curve_5, label="sarsa_ten")
plt.legend()
plt.ylabel('reward')
plt.xlabel('episodes')
plt.xlim([0, 50])
plt.ylim([-500, 0])
plt.show()

env = CliffWalking()

EPISODE_NUM = 800

Q, policy = mc.learn(env, num_episodes = EPISODE_NUM, epsilon = 0.1)
# episodes, learning_curve_1 = mc.get_learning_curve()
mc_success_cnt = get_success_rate(env, policy)

Q, policy = sarsa_zero.learn(env, num_episodes = EPISODE_NUM, epsilon = 0.1)
# _, learning_curve_2 = sarsa_zero.get_learning_curve()
success_cnt_1 = get_success_rate(env, policy)

# Q, policy = sarsa_one.learn(env, num_episodes = 100, epsilon = 0.1)
# _, learning_curve_3 = sarsa_one.get_learning_curve()
Q, policy = sarsa_five.learn(env, num_episodes = EPISODE_NUM, epsilon = 0.1)
# _, learning_curve_4 = sarsa_five.get_learning_curve()
success_cnt_2 = get_success_rate(env, policy)

Q, policy = sarsa_ten.learn(env, num_episodes = EPISODE_NUM, epsilon = 0.1)
# _, learning_curve_5 = sarsa_ten.get_learning_curve()
success_cnt_3 = get_success_rate(env, policy)

plt.bar(["sarsa_zero", "sarsa_five", "sarsa_ten", "mc"], [success_cnt_1, success_cnt_2, success_cnt_3, mc_success_cnt])
plt.ylabel("success rate")
plt.show()