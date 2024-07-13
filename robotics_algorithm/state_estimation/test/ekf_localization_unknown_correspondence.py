#!/usr/bin/evn python

import sys
import os.path as osp
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../../'))

import numpy as np
import random
import matplotlib.pyplot as plt
import math

from env.two_d_localization_with_feature import TwoDLocalizationWithFeature
from state_estimation.extended_kalman_filter import ExtendedKalmanFilter

# Initialize environment
env = TwoDLocalizationWithFeature()
env.meas_correspondence = False # disable correspondence
num_of_feature = env.num_of_features

# -------- Settings ------------
NUM_OF_TIMESTAMP = 10

# -------- Helper Functions -------------
def control_noise_matrix(state, control):
    V = env.process_noise_jacobian(state, control)
    return V @ env.R @ V.transpose()

feature_idx = 0
def measurement_func(state):
    global feature_idx
    return env.measurement_func_each_feature(state, feature_idx)

def compute_meas_jacobian(state):
    global feature_idx
    return env.meas_jacobian_each_feature(state, feature_idx)

def MLE_feature_idx(state, meas_feature):
    meas_llh = []
    for idx in range(num_of_feature):
        meas_jacobian_feat = env.meas_jacobian_each_feature(state, idx)
        S = meas_jacobian_feat @ my_filter.covariance @ meas_jacobian_feat.transpose() + env.Q
        expected_meas_feature = env.measurement_func_each_feature(state, idx)

        # print(idx, expected_meas_feature, meas_feature)

        # likelihood
        llh = (meas_feature - expected_meas_feature).transpose() @ np.linalg.inv(S) @ (meas_feature - expected_meas_feature)

        meas_llh.append(llh)

    # MLE
    mle_feat_idx = np.argmin(np.array(meas_llh))

    return mle_feat_idx

# -------- Main Code ----------
# Initialize filter
initial_covariance = np.eye(3)
my_filter = ExtendedKalmanFilter(env.state, initial_covariance, env.state_transition, env.control_jacobian, control_noise_matrix, measurement_func, compute_meas_jacobian, env.Q)

# Add initial state
true_state = []
filter_state = []
true_state.append(env.state)
filter_state.append(my_filter.state)

# Run test
for i in range(NUM_OF_TIMESTAMP):
    print("timestamp : {}".format(i))
    control = [random.uniform(-5.0, 5.0), random.uniform(-math.pi / 2, math.pi / 2)]

    meas = env.control_and_measure(control)
    # print("True state: {}".format(env.state))
    # print("meas: {}".format(meas))

    my_filter.predict(control)
    # print("Filter belief before meas: {}".format(my_filter.state))
    for idx, meas_feature in enumerate(meas):
        # feature_idx = idx

        feature_idx = MLE_feature_idx(my_filter.state, meas_feature)

        print("idx: {}, mle_feature_idx: {}".format(idx, feature_idx))
        my_filter.update(meas_feature)
    # print("Filter belief after meas: {}".format(my_filter.state))

    true_state.append(env.state)
    filter_state.append(my_filter.state)


# plot result
NUM_OF_TIMESTAMP += 1
t = np.arange(NUM_OF_TIMESTAMP)

fig, (ax1, ax2, ax3) = plt.subplots(nrows = 3, ncols = 1)
ax1.plot(t, [true_state[i][0] for i in range(NUM_OF_TIMESTAMP)], 'k')
ax1.plot(t, [filter_state[i][0] for i in range(NUM_OF_TIMESTAMP)], 'bo')
ax1.set_ylabel('x')
ax2.plot(t, [true_state[i][1] for i in range(NUM_OF_TIMESTAMP)], 'k')
ax2.plot(t, [filter_state[i][1] for i in range(NUM_OF_TIMESTAMP)], 'bo')
ax2.set_ylabel('y')
ax3.plot(t, [true_state[i][2] for i in range(NUM_OF_TIMESTAMP)], 'k')
ax3.plot(t, [filter_state[i][2] for i in range(NUM_OF_TIMESTAMP)], 'bo')
ax3.set_ylabel('theta')
plt.show()

