#!/usr/bin/evn python

import sys
import os.path as osp
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../../'))

import numpy as np
import random
import matplotlib.pyplot as plt
import math

from robotics_algorithm.env.continuous_world_2d import DiffDrive2DEnvComplex
from robotics_algorithm.state_estimation.extended_kalman_filter import ExtendedKalmanFilter

# Initialize environment
env = DiffDrive2DEnvComplex()
obs, _ = env.reset()
env.start_state = [env.size / 2, env.size / 2, 0]
env.cur_state = [env.size / 2, env.size / 2, 0]
env.render()

# Initialize filter
filter = ExtendedKalmanFilter(env)
filter.set_initial_state(env.cur_state)

# Add initial state
# Step env with random actions
true_states = []
filter_states = []
obss = []
true_states.append(env.cur_state)
filter_states.append(filter.get_state())
obss.append(obs)
max_steps = 100
for i in range(max_steps):
    action = [random.uniform(0.0, 0.5), random.uniform(0, 0.5)]
    new_obs, reward, term, trunc, info = env.step(action)

    filter.run(action, new_obs)

    true_states.append(env.cur_state)
    filter_states.append(filter.get_state())
    obss.append(new_obs)

    if term or trunc:
        break

print(true_states)
print(filter_states)

# calculate RMSE
true_states = np.array(true_states)
filter_states = np.array(filter_states)
rmse = np.sqrt(np.mean((true_states - filter_states) ** 2, axis=0))
print("RMSE: {}".format(rmse))

env.add_ref_path(true_states)
env.add_state_path(filter_states)
env.render()