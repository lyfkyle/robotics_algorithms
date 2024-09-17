#!/usr/bin/evn python

import sys
sys.path.append('../../environment/')
sys.path.append('../')

import numpy as np
import random
import matplotlib.pyplot as plt
import math

from two_d_localization_with_feature import TwoDLocalizationWithFeature
from extended_kalman_filter import ExtendedKalmanFilter

# Initialize environment
env = TwoDLocalizationWithFeature()

# -------- Settings ------------
NUM_OF_TIMESTAMP = 10

# -------- Helper Functions -------------
def compute_R(state, control):
    V = env.process_noise_jacobian(state, control)
    return V @ env.R @ V.transpose()

feature_idx = 0
def measurement_func(state):
    global feature_idx
    return env.measurement_func_each_feature(state, feature_idx)

def compute_jacobian_H(state):
    global feature_idx
    return env.meas_jacobian_each_feature(state, feature_idx)

# -------- Main Code ----------

# Initialize filter
initial_covariance = np.eye(3)
my_filter = ExtendedKalmanFilter(env.state, initial_covariance, env.state_transition, env.control_jacobian, compute_R, measurement_func, compute_jacobian_H, env.Q)

# Add initial state
true_state = []
filter_state = []
true_state.append(env.state)
filter_state.append(my_filter.state)

# Run test
for i in range(NUM_OF_TIMESTAMP):
    print("timestamp : {}".format(i))
    control = [random.uniform(-1.0, 1.0), random.uniform(-math.pi / 2, math.pi / 2)]

    meas = env.control_and_measure(control)
    # print("True state: {}".format(env.state))
    # print("meas: {}".format(meas))

    my_filter.predict(control)
    # print("Filter belief before meas: {}".format(my_filter.state))
    for idx, meas_feature in enumerate(meas):
        feature_idx = idx
        my_filter.update(meas_feature)
    # print("Filter belief after meas: {}".format(my_filter.state))

    true_state.append(env.state)
    filter_state.append(my_filter.state)


# plot result
NUM_OF_TIMESTAMP += 1
t = np.arange(NUM_OF_TIMESTAMP)

fig, (ax1, ax2, ax3) = plt.subplots(nrows = 3, ncols = 1)
ax1.plot(t, [true_state[i][0, 0] for i in range(NUM_OF_TIMESTAMP)], 'k')
ax1.plot(t, [filter_state[i][0, 0] for i in range(NUM_OF_TIMESTAMP)], 'bo')
ax1.set_ylabel('x')
ax2.plot(t, [true_state[i][1, 0] for i in range(NUM_OF_TIMESTAMP)], 'k')
ax2.plot(t, [filter_state[i][1, 0] for i in range(NUM_OF_TIMESTAMP)], 'bo')
ax2.set_ylabel('y')
ax3.plot(t, [true_state[i][2, 0] for i in range(NUM_OF_TIMESTAMP)], 'k')
ax3.plot(t, [filter_state[i][2, 0] for i in range(NUM_OF_TIMESTAMP)], 'bo')
ax3.set_ylabel('theta')
plt.show()