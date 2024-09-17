#!/usr/bin/evn python

import sys
sys.path.append('../../environment/')
sys.path.append('../')

import numpy as np
import random
import matplotlib.pyplot as plt

from one_d_localization_continuous import OneDLocalizationContinuous
from lqr import LQR

# Initialize environment
env = OneDLocalizationContinuous()

# -------- Settings ------------
NUM_OF_TIMESTAMP = 10

# -------- Main Code ----------
pos_setpoint = 5
vel_setpoint = 0
Q = np.array([[1, 0],
              [0, 20]]) # controls how much state difference cost
R = 0.01 # controls how much control cost.

state_setpoint = np.array([[pos_setpoint], [vel_setpoint]])

initial_covariance = np.eye(2)

my_controller = LQR(env.A, env.B, Q, R)
gain = my_controller.get_K(10) # note that the minus sign is omitted here because of the way we calculate error.
print('gain: {}'.format(gain))

true_state = [env.state]
accel_cmd = [0]

for i in range(10):
    accel = np.dot(gain, (state_setpoint - env.state))[0, 0]
    meas = env.control_and_sense(accel)

    print("True state: {}".format(env.state))
    print("accel : {}".format(accel))

    true_state.append(env.state)
    accel_cmd.append(accel)

# print(accel_cmd)

# plot result
NUM_OF_TIMESTAMP += 1
t = np.arange(NUM_OF_TIMESTAMP)

fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1)
ax1.plot(t, [true_state[i][0, 0] for i in range(NUM_OF_TIMESTAMP)], 'k')
ax1.plot(t, [true_state[i][1, 0] for i in range(NUM_OF_TIMESTAMP)], 'bo')
ax1.set_ylabel('state')
ax2.plot(t, accel_cmd, 'k')
ax2.set_ylabel('control')
plt.show()