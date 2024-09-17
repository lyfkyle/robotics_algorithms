import copy
import math
import numpy as np
from numpy.random import randn
import scipy.stats

class TwoDLocalizationWithFeature(object):
    '''
    This environment involves an agent trying to localize itself in a 2d grid of size 10*10
    There are four features/landmarks, each at each corner.
    The agent is able to measure distance to each landmark

    State : [[x], [y], [theta]]
    Control ：[velocity, angular velocity] # Refer to velocity model in Probabilistic Robotics Chapter 5.
    Measurement: a list of size 4, each entry consists measurement to a feature. Each measurement has format
                 [[distance to feature], [bearing angle of feature](the amount agent must rotate to face feature), [feature index]]
    '''
    def __init__(self, initial_state=[[5], [5], [0]], control_noise=[0.01, 0.01], meas_noise=[0.01, 0.01, 1e-10]):
        '''
        @param initial_state, initial_state of agent, in [x, y, theta] format
        @param control_noise, noise associated with control signal
        @param meas_noise, noise associated with measurement signal
        '''

        self.state = np.array(initial_state)
        self.dt = 1
        self.num_of_features = 4
        self.features = [{'pos': [0, 0], 'theta': 0, 'ind': 0},
                         {'pos': [0, 10], 'theta': 0, 'ind': 1},
                         {'pos': [10, 10], 'theta': 0, 'ind': 2},
                         {'pos': [10, 0], 'theta': 0, 'ind': 3}]
        self.control_noise = control_noise
        self.meas_noise = meas_noise
        self.R = np.array([[control_noise[0], 0],
                           [0, control_noise[1]]], dtype=np.float32)
        self.Q = np.array([[meas_noise[0], 0 ,0],
                           [0, meas_noise[1], 0],
                           [0, 0, meas_noise[2]]], dtype=np.float32)

    def control(self, control):
        '''
        Run control on agent in simulation.
        Agent's state will be updated according to transition function
        '''
        new_state = self.state_transition(self.state, control)
        self.state = new_state

    def measure(self):
        '''
        Measure agent's location in simulation
        @return, measurement from agent
        '''
        meas = []
        for i in range(self.num_of_features):
            feature = self.features[i]
            meas_feature = np.array([[math.sqrt((feature['pos'][0] - self.state[0, 0]) ** 2 + (feature['pos'][1] - self.state[1, 0]) ** 2)],
                                     [math.atan2(feature['pos'][1] - self.state[1, 0], feature['pos'][0] - self.state[0, 0]) - self.state[2, 0]],
                                     [feature['ind']]])
            # add random noise
            meas_feature += np.array([[randn() * self.meas_noise[0]],
                                      [randn() * self.meas_noise[1]],
                                      [0]])
            meas.append(meas_feature)

        return meas

    def control_and_measure(self, control):
        '''
        Run control on agent and then measure agent location
        @return, measurement from agent
        '''
        self.control(control)
        return self.measure()

    def state_transition(self, state, control):
        '''
        State transition function， given agent's state and control signal, compute agent's next state
        @return new_state, agent's new state
        '''
        # unpack input
        vel = control[0]
        ang_vel = control[1]

        # add gaussian noise to velocity and ang_vel
        true_vel = vel + randn() * self.control_noise[0]
        true_ang_vel = ang_vel + randn() * self.control_noise[1]
        theta = self.state[2, 0]

        # calculate new state
        if true_ang_vel != 0:
            temp = true_vel / true_ang_vel
            new_state = state + np.array([[-temp * math.sin(theta) + temp * math.sin(theta + true_ang_vel * self.dt)],
                                          [temp * math.cos(theta) - temp * math.cos(theta + true_ang_vel * self.dt)],
                                          [true_ang_vel * self.dt]])
        else:
            new_state = state + np.array([[true_vel * math.cos(theta)],
                                          [true_vel * math.sin(theta)],
                                          [0]])

        return new_state

    def measurement_func_each_feature(self, state, feature_idx):
        '''
        Measurement function. Given agent's state, return measurement.
        @param state, state
        @param feature_idx, index of feature
        @return meas, measurement.
        '''
        feature = self.features[feature_idx]

        meas = np.array([[math.sqrt((feature['pos'][0] - state[0, 0]) ** 2 + (feature['pos'][1] - state[1, 0]) ** 2)],
                         [math.atan2(feature['pos'][1] - state[1, 0], feature['pos'][0] - state[0, 0]) - state[2, 0]],
                         [feature['ind']]])

        return meas

    def measurement_func(self, state):
        '''
        Measurement function. Given agent's state, return measurement.
        @param state, state
        @return meas, measurement.
        '''
        meas = []
        for idx in range(self.num_of_features):
            meas_feature = self.measurement_func_each_feature(state, idx)
            meas.append(meas_feature)

        return meas

    def measuremnt_prob_each_feature(self, state, meas_feature):
        '''
        A function to return measurement probability(likelihood).
        @param state, state
        @param meas_feature, measurement to a feature.
        @return the probability of getting that measurement in state
        '''
        feature = self.features[int(meas_feature[2, 0])]

        meas = np.array([[math.sqrt((feature['pos'][0] - state[0, 0]) ** 2 + (feature['pos'][1] - state[1, 0]) ** 2)],
                         [math.atan2(feature['pos'][1] - state[1, 0], feature['pos'][0] - state[0, 0]) - state[2, 0]],
                         [feature['ind']]])

        error_1 = meas_feature[0, 0] - meas[0, 0]
        error_2 = meas_feature[1, 0] - meas[1, 0]
        # meas_prob_1 = scipy.stats.norm(loc=0, scale=math.sqrt(self.meas_noise[0]))
        # meas_prob_2 = scipy.stats.norm(loc=0, scale=math.sqrt(self.meas_noise[1]))
        # prob = meas_prob_1.pdf(meas_feature[0, 0] - meas[0, 0]) * meas_prob_2.pdf(meas_feature[1, 0] - meas[1, 0]) # ignore prob of ind. Assume ind is always correct
        prob = math.e ** -(error_1 ** 2 / (2 * math.sqrt(self.meas_noise[0]))) * math.e ** -(error_2 ** 2 / (2 * math.sqrt(self.meas_noise[1]))) # ignore prob of ind. Assume ind is always correct
        # print(prob)
        return prob

    def measuremnt_prob(self, state, meas):
        '''
        A function to return measurement probability(likelihood).
        @param state, state
        @param meas, the combined measurement to each feature
        @return the probability of getting that measurement in state
        '''
        prob = 1
        for meas_feature in meas:
            prob *= self.measuremnt_prob_each_feature(state, meas_feature)

        return prob

    def control_jacobian(self, state, control):
        '''
        Return jocobian matrix of transition function
        @state, state
        @control, control
        @return A, jacobian matrix
        '''

        vel = control[0]
        ang_vel = control[1]
        theta = state[2, 0]

        temp = vel / ang_vel
        self.A = np.array([[1, 0, -temp * math.cos(theta) + temp * math.sin(theta + ang_vel * self.dt)],
                           [0, 1, -temp * math.sin(theta) + temp * math.sin(theta + ang_vel * self.dt)],
                           [0, 0, 1]])
        return self.A

    def process_noise_jacobian(self, state, control):
        '''
        Return process_noise_jocobian matrix.
        Note that process noise is defined in control space, this jacobian matrix can be used to transform process noise into state space
        @state, state
        @control, control
        @return V, jacobian matrix
        '''
        vel = control[0]
        ang_vel = control[1]
        theta = state[2, 0]

        self.V = np.array([[(-math.sin(theta) + math.sin(theta + ang_vel * self.dt)) / ang_vel, vel * (math.sin(theta) - math.sin(theta + ang_vel * self.dt)) / (ang_vel ** 2) + vel * math.cos(theta + ang_vel * self.dt) * self.dt / ang_vel],
                           [(math.cos(theta) - math.cos(theta + ang_vel * self.dt)) / ang_vel, -vel * (math.cos(theta) - math.cos(theta + ang_vel * self.dt)) / (ang_vel ** 2) + vel * math.sin(theta + ang_vel * self.dt) * self.dt / ang_vel],
                           [0, self.dt]])
        return self.V

    def meas_jacobian(self, state, meas):
        '''
        Return meas_jacobian matrix.
        Note, since there are four features, this function will return a list of size 4, each entry is a meas_jacobian matrix to one feature
        @state, state
        @meas, measurement
        @return H_list, a list of meas_jacobian matrix
        '''
        H_list = []
        for feature in self.features:
            temp = (feature['pos'][0] - state[0, 0]) ** 2 + (feature['pos'][1] - state[1, 0]) ** 2
            H = np.array([[-(feature['pos'][0] - state[0, 0]) / math.sqrt(temp), -(feature['pos'][1] - state[1, 0]) / math.sqrt(temp), 0],
                            [(feature['pos'][1] - state[1, 0]) / temp, -(feature['pos'][0] - state[0, 0]) / math.sqrt(temp), -1],
                            [0, 0, 0]])
            H_list.append(H)

        return H_list

    def meas_jacobian_each_feature(self, state, feature_idx):
        '''
        Return meas_jacobian matrix for measurement to a single feature
        @state, state
        @feature_idx, index of feature
        @return H, the meas jacobian matrix
        '''
        # print(meas_feature)
        feature = self.features[feature_idx]

        temp = (feature['pos'][0] - state[0, 0]) ** 2 + (feature['pos'][1] - state[1, 0]) ** 2
        H = np.array([[-(feature['pos'][0] - state[0, 0]) / math.sqrt(temp), -(feature['pos'][1] - state[1, 0]) / math.sqrt(temp), 0],
                        [(feature['pos'][1] - state[1, 0]) / temp, -(feature['pos'][0] - state[0, 0]) / math.sqrt(temp), -1],
                        [0, 0, 0]])

        return H

