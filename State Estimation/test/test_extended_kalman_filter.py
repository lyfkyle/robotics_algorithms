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

env = TwoDLocalizationWithFeature()

initial_covariance = np.eye(3)

def compute_R(state, control):
    V = env.process_noise_jacobian(state, control)
    return V @ env.R @ V.transpose()

my_filter = ExtendedKalmanFilter(env.state, initial_covariance, env.state_transition, env.control_jacobian, compute_R, env.measurement_func, env.meas_jacobian, env.Q)

true_state = []
filter_state = []
true_state.append(env.state)
filter_state.append(my_filter.state)

for _ in range(10):
    control = [random.uniform(-1.0, 1.0), random.uniform(-math.pi / 2, math.pi / 2)]

    meas = env.control_and_sense(control)
    # print("True state: {}".format(env.state))
    # print("meas: {}".format(meas))

    my_filter.predict(control)
    # print("Filter belief before meas: {}".format(my_filter.state))
    for meas_feature in meas:
        my_filter.update(meas_feature)
    # print("Filter belief after meas: {}".format(my_filter.state))

    true_state.append(env.state)
    filter_state.append(my_filter.state)


# plot result
t = np.arange(11)

fig, (ax1, ax2, ax3) = plt.subplots(nrows = 3, ncols = 1)
ax1.plot(t, [true_state[i][0, 0] for i in range(11)], 'k')
ax1.plot(t, [filter_state[i][0, 0] for i in range(11)], 'bo')
ax1.set_ylabel('x')
ax2.plot(t, [true_state[i][1, 0] for i in range(11)], 'k')
ax2.plot(t, [filter_state[i][1, 0] for i in range(11)], 'bo')
ax2.set_ylabel('y')
ax3.plot(t, [true_state[i][2, 0] for i in range(11)], 'k')
ax3.plot(t, [filter_state[i][2, 0] for i in range(11)], 'bo')
ax3.set_ylabel('theta')
plt.show()