#!/usr/bin/evn python

import sys
sys.path.append('../../environment/')
sys.path.append('../')

import numpy as np
import random
import matplotlib.pyplot as plt

from two_d_localization_with_feature import TwoDLocalizationWithFeature
from extended_kalman_filter import ExtendedKalmanFilter

env = TwoDLocalizationWithFeature()

initial_covariance = np.eye(3)

def compute_R(state, control):
    V = env.process_noise_jacobian(state, control)
    return V @ env.R @ V.transpose()

my_filter = ExtendedKalmanFilter(env.state, initial_covariance, env.control, env.control_jacobian, compute_R, env.meas_jacobian, env.Q)

true_state = []
filter_state = []

for i in range(10):
    control = random.uniform(-1.0, 1.0)
    meas = env.control_and_sense(accel)

    my_filter.predict(np.array([[accel]]))
    my_filter.update(meas)

    # print("True state: {}".format(env.state))
    # print("Filter belief : {}".format(my_filter.state))

    true_state.append(env.state)
    filter_state.append(my_filter.state)


# plot result
t = np.arange(10)

fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1)
ax1.plot(t, [true_state[i][0, 0] for i in range(10)], 'k')
ax1.plot(t, [filter_state[i][0, 0] for i in range(10)], 'bo')
ax1.set_ylabel('position')
ax2.plot(t, [true_state[i][1, 0] for i in range(10)], 'k')
ax2.plot(t, [filter_state[i][1, 0] for i in range(10)], 'bo')
ax2.set_ylabel('velocity')
plt.show()