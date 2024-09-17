import sys
import os.path as osp
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../../'))
# sys.path.append('../environment/')
# sys.path.append('../')

import numpy as np
import math
import random
import matplotlib.pyplot as plt

from state_estimation.extended_kalman_filter import ExtendedKalmanFilter
from env.two_d_localization_with_feature import TwoDLocalizationWithFeature

# Initialize environment
env = TwoDLocalizationWithFeature()
env.meas_correspondence = False

# state
num_of_feature = 4
feature_size = 3
robot_state_size = 3
robot_pos = env.state
feature_pos = np.zeros(feature_size * num_of_feature)
state = np.concatenate((robot_pos, feature_pos), axis = 0)
state_size = state.shape[0]
feature_state_size = feature_pos.shape[0]

# state_covariance
state_covariance = np.eye(state_size)

# helper matrix
F = np.concatenate((np.eye(robot_state_size), np.zeros((robot_state_size, feature_state_size))), axis = 1) # the matrix to map robot_state to full state.

def state_transition(state, control):
    robot_state = state[:robot_state_size]
    feature_state = state[robot_state_size:]
    new_robot_state = env.state_transition(robot_state, control) # only robot state is affected by control
    new_state = np.concatenate((new_robot_state, feature_state))

    return new_state

def control_jacobian(state, control):
    # jacobian wrt robot_state at control
    A = env.control_jacobian(state, control)

    # map to full_state
    A = A - np.eye(A.shape[0])
    G = np.eye(state_size) + F.transpose() @ A @ F

    return G

def control_noise_matrix(state, control):
    # robot_state space process noise
    V = env.process_noise_jacobian(state, control)
    R = V @ env.R @ V.transpose()

    # map to full state
    full_state_R = F.transpose() @ R @ F

    return full_state_R

feature_idx = 0
def measurement_func(state):
    global feature_idx
    return env.measurement_func_each_feature(state, feature_idx)

def measurement_jacobian_feat(state, feat_idx):
    feat_state_idx = robot_state_size + feature_size * feat_idx
    robot_state = state[:robot_state_size]
    feat_state = state[feat_state_idx : feat_state_idx + feature_size]

    temp = (feat_state[0] - robot_state[0]) ** 2 + (feat_state[1] - robot_state[1]) ** 2
    # partial derivative against robot_state
    H1 = np.array([[-(feat_state[0] - robot_state[0]) / math.sqrt(temp), -(feat_state[1] - robot_state[0]) / math.sqrt(temp), 0],
                   [(feat_state[1] - robot_state[1]) / temp, -(feat_state[0] - robot_state[0]) / math.sqrt(temp), -1],
                   [0, 0, 0]])
    # partial derivative against feat_state
    H2 = np.array([[(feat_state[0] - state[0]) / math.sqrt(temp), (feat_state[1] - state[1]) / math.sqrt(temp), 0],
                   [-(feat_state[1] - state[1]) / temp, -(feat_state[0] - state[0]) / math.sqrt(temp), 0],
                   [0, 0, 1]])
    H = np.concatenate((H1, H2), axis=1) # stack horizontally

    # transform to full state
    F_j = np.zeros((robot_state_size + feature_size, state_size))
    F_j[:robot_state_size, :robot_state_size] = np.eye(robot_state_size)
    F_j[robot_state_size:, feat_state_idx:feat_state_idx+feature_size] = np.eye(feature_size)

    full_state_H = H @ F_j

    return full_state_H

def measurement_jacobian(state):
    global feature_idx
    return measurement_jacobian_feat(state, feature_idx)

def compute_feat_location(state, meas):
    r = meas[0]
    bearing = meas[1]
    idx = meas[2]
    theta = state[2]
    robot_state = state[:robot_state_size]
    feat_state = np.array([robot_state[0], robot_state[1], idx]) + np.array([r * math.cos(bearing + theta),
                                                                             r * math.sin(bearing + theta),
                                                                             0])
    return feat_state

def MLE_feature_idx(state, meas_feature):
    robot_state = state[:robot_state_size]

    meas_llh = []
    for idx in range(min(num_of_feature, num_feature_observed + 1)):
        meas_jacobian_feat = measurement_jacobian_feat(state, idx)
        S = meas_jacobian_feat @ my_filter.covariance @ meas_jacobian_feat.transpose() + env.Q
        expected_meas_feature = env.measurement_func_each_feature(robot_state, idx)

        # print(idx, expected_meas_feature, meas_feature)

        # likelihood
        llh = (meas_feature - expected_meas_feature).transpose() @ np.linalg.inv(S) @ (meas_feature - expected_meas_feature)

        meas_llh.append(llh)

    # print(meas_llh)
    meas_llh[-1] += 0.01 # manually decrease the likelihood of hypothesized feature. Only create new feature if the llh of feat_measurement is
                         # different from all existing llh

    # MLE
    mle_feat_idx = np.argmin(np.array(meas_llh))

    return mle_feat_idx

# EKF
my_filter = ExtendedKalmanFilter(state, state_covariance, state_transition, control_jacobian, control_noise_matrix, measurement_func, measurement_jacobian, env.Q)

# Add initial state
true_states = []
filter_states = []

true_feat_state = []
for feat in env.features:
    true_feat_state.append([feat["pos"][0], feat["pos"][1], feat["ind"]])
true_feat_state = np.array(true_feat_state).reshape(-1)

true_state = np.concatenate((env.state, true_feat_state))
true_states.append(true_state)
filter_states.append(my_filter.state)

# Run test
num_feature_observed = 0
NUM_OF_TIMESTAMP = 50
for i in range(NUM_OF_TIMESTAMP):
    print("timestamp : {}".format(i))
    control = [random.uniform(-5.0, 5.0), random.uniform(-math.pi / 2, math.pi / 2)]

    meas = env.control_and_measure(control)
    # print("True state: {}".format(env.state))
    # print("meas: {}".format(meas))

    my_filter.predict(control)
    # print("Filter belief before meas: {}".format(my_filter.state))
    for idx, meas_feature in enumerate(meas):

        if num_feature_observed < num_of_feature:
            # expected location of the new feature
            feat_state = compute_feat_location(state, meas_feature)
            feat_state_idx = robot_state_size + feature_size * num_feature_observed
            my_filter.state[feat_state_idx : feat_state_idx + feature_size] = feat_state

        # MLE feature idx estimate
        feature_idx = MLE_feature_idx(my_filter.state, meas_feature)
        # print("idx: {}, mle_feature_idx: {}".format(idx, feature_idx))

        # create new feature
        if feature_idx >= num_feature_observed:
            num_feature_observed += 1

        # for feature first observed, compute its expected position

        my_filter.update(meas_feature)

    true_state = np.concatenate((env.state, true_feat_state))
    true_states.append(true_state)
    filter_states.append(my_filter.state)

print("Filter belief after meas: {}".format(my_filter.state))

# plot result
NUM_OF_TIMESTAMP += 1
t = np.arange(NUM_OF_TIMESTAMP)

# plot localization result
fig, (ax1, ax2, ax3) = plt.subplots(nrows = 3, ncols = 1)
ax1.plot(t, [true_states[i][0] for i in range(NUM_OF_TIMESTAMP)], 'k')
ax1.plot(t, [filter_states[i][0] for i in range(NUM_OF_TIMESTAMP)], 'bo')
ax1.set_ylabel('x')
ax2.plot(t, [true_states[i][1] for i in range(NUM_OF_TIMESTAMP)], 'k')
ax2.plot(t, [filter_states[i][1] for i in range(NUM_OF_TIMESTAMP)], 'bo')
ax2.set_ylabel('y')
ax3.plot(t, [true_states[i][2] for i in range(NUM_OF_TIMESTAMP)], 'k')
ax3.plot(t, [filter_states[i][2] for i in range(NUM_OF_TIMESTAMP)], 'bo')
ax3.set_ylabel('theta')
plt.show()

# plot mapping result
fig, axes = plt.subplots(nrows = 3, ncols = 4)
for x in range(3):
    for y in range(4):
        axes[x, y].plot(t, [true_states[i][robot_state_size + y*3 + x] for i in range(NUM_OF_TIMESTAMP)], 'k')
        axes[x, y].plot(t, [filter_states[i][robot_state_size + y*3 + x] for i in range(NUM_OF_TIMESTAMP)], 'bo')
        axes[x, y].set_ylabel('feat_{}_{}'.format(y, x))
plt.show()