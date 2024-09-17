#!/usr/bin/evn python

import sys
sys.path.append('../../environment/')
sys.path.append('../')

import numpy as np
import random
import matplotlib.pyplot as plt

from one_d_localization_continuous import OneDLocalizationContinuous
from kalman_filter import KalmanFilter

env = OneDLocalizationContinuous()

initial_covariance = np.eye(2)

my_filter = KalmanFilter(env.state, initial_covariance, env.A, env.B, env.R, env.H, env.Q)

true_state = []
filter_state = []

for i in range(10):
    accel = random.uniform(-1.0, 1.0)
    meas = env.control_and_sense(accel)

    my_filter.predict(np.array([[accel]]))
    my_filter.update(meas)

    # print("True state: {}".format(env.state))
    # print("Filter belief : {}".format(my_filter.state))

    true_state.append(env.state)
    filter_state.append(my_filter.state)


# plot result
t = np.arange(10)

plt.figure()
plt.subplot(211)
plt.plot(t, [true_state[i][0, 0] for i in range(10)], 'bo', t, [filter_state[i][0, 0] for i in range(10)], 'k')
plt.ylabel('position')
plt.subplot(212)
plt.plot(t, [true_state[i][1, 0] for i in range(10)], 'bo', t, [filter_state[i][1, 0] for i in range(10)], 'k')
plt.ylabel('velocity')
plt.show()
