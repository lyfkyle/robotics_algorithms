#!/usr/bin/evn python

import sys
import os.path as osp

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "../../"))

import numpy as np
import random
import matplotlib.pyplot as plt

from robotics_algorithm.env.continuous_world_1d import DoubleIntegratorEnv
from robotics_algorithm.state_estimation.kalman_filter import KalmanFilter

# Env
env = DoubleIntegratorEnv(observation_noise_std=0.1, state_transition_noise_std=0.1)
env.reset()

# Create filter
filter = KalmanFilter(env)
filter.set_initial_state(env.cur_state)

# Step env with random actions
true_state = []
filter_state = []
max_steps = 100
for i in range(max_steps):
    action = [random.uniform(-1.0, 1.0)]
    new_obs, reward, term, trunc, info = env.step(action)

    filter.run(action, new_obs)

    if term or trunc:
        break

    # print("True state: {}".format(env.state))
    # print("Filter belief : {}".format(my_filter.state))

    true_state.append(env.cur_state)
    filter_state.append(filter.get_state())


# calculate RMSE
true_state = np.array(true_state)
filter_state = np.array(filter_state)
rmse = np.sqrt(np.mean((true_state - filter_state) ** 2, axis=0))
print("RMSE: {}".format(rmse))

# plot result
t = np.arange(max_steps)
fig = plt.figure()
plt.plot(t, [true_state[i][0] for i in range(max_steps)], "k", label="groundtruth")
plt.plot(t, [filter_state[i][0] for i in range(max_steps)], "b", label="predicted")
plt.ylabel("position")
plt.legend()
plt.show()
