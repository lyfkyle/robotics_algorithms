import copy
import math
import numpy as np
from numpy.random import randn

class TwoDLocalizationWithFeature(object):
    def __init__(self, control_noise=[0.01, 0.05], meas_noise=[0.01, 0.05, 0.001]):
        self.state = np.array([[5], [5], [0]])
        self.dt = 1
        self.num_of_features = 4
        self.features = [{'pos': [0, 0], 'theta': 0, 'ind': 0},
                         {'pos': [0, 10], 'theta': 0, 'ind': 1},
                         {'pos': [10, 10], 'theta': 0, 'ind': 2},
                         {'pos': [10, 0], 'theta': 0, 'ind': 3}]
        self.control_noise = control_noise
        self.meas_noise = meas_noise
        self.R = np.array([[control_noise[0], 0], [0, control_noise[1]]], dtype=np.float32)
        self.Q = np.array([[meas_noise[0], 0 ,0],
                           [0, meas_noise[1], 0],
                           [0, 0, meas_noise[2]]])

    def state_transition(self, state, control):
        vel = control[0]
        ang_vel = control[1]

        # add gaussian noise to velocity and ang_vel
        true_vel = vel + randn() * self.control_noise[0]
        true_ang_vel = ang_vel + randn() * self.control_noise[1]

        # temp variables
        theta = self.state[2, 0]
        temp = true_vel / true_ang_vel
        new_state = state + np.array([[-temp * math.sin(theta) + temp * math.sin(theta + true_ang_vel * self.dt)],
                                           [temp * math.cos(theta) - temp * math.cos(theta + true_ang_vel * self.dt)],
                                           [true_ang_vel * self.dt]])

        return new_state

    def control(self, control):
        '''

        '''
        new_state = self.state_transition(self.state, control)
        self.state = new_state

    def measurement_func(self, state, meas_feature):
        feature = self.features[int(meas_feature[2, 0])]

        meas = np.array([[math.sqrt((feature['pos'][0] - state[0, 0]) ** 2 + (feature['pos'][1] - state[1, 0]) ** 2)],
                         [math.atan2(feature['pos'][1] - state[1, 0], feature['pos'][0] - state[0, 0]) - state[2, 0]],
                         [feature['ind']]])
        return meas

    def sense(self):
        meas = []
        for i in range(self.num_of_features):
            feature = self.features[i]
            meas_feature = np.array([[math.sqrt((feature['pos'][0] - self.state[0, 0]) ** 2 + (feature['pos'][1] - self.state[1, 0]) ** 2)],
                                     [math.atan2(feature['pos'][1] - self.state[1, 0], feature['pos'][0] - self.state[0, 0]) - self.state[2, 0]],
                                     [feature['ind']]])
            meas_feature += np.array([[randn() * self.meas_noise[0]],
                                      [randn() * self.meas_noise[1]],
                                      [0]])
            meas.append(meas_feature)

        return meas

    def control_and_sense(self, control):
        self.control(control)
        return self.sense()

    def control_jacobian(self, state, control):
        vel = control[0]
        ang_vel = control[1]
        theta = state[2, 0]

        temp = vel / ang_vel
        self.A = np.array([[1, 0, -temp * math.cos(theta) + temp * math.sin(theta + ang_vel * self.dt)],
                           [0, 1, -temp * math.sin(theta) + temp * math.sin(theta + ang_vel * self.dt)],
                           [0, 0, 1]])
        return self.A

    def process_noise_jacobian(self, state, control):
        vel = control[0]
        ang_vel = control[1]
        theta = state[2, 0]

        self.V = np.array([[(-math.sin(theta) + math.sin(theta + ang_vel * self.dt)) / ang_vel, vel * (math.sin(theta) - math.sin(theta + ang_vel * self.dt)) / (ang_vel ** 2) + vel * math.cos(theta + ang_vel * self.dt) * self.dt / ang_vel],
                           [(math.cos(theta) - math.cos(theta + ang_vel * self.dt)) / ang_vel, -vel * (math.cos(theta) - math.cos(theta + ang_vel * self.dt)) / (ang_vel ** 2) + vel * math.sin(theta + ang_vel * self.dt) * self.dt / ang_vel],
                           [0, self.dt]])
        return self.V

    def meas_jacobian(self, state, meas_feature):
        # print(meas_feature)
        feature = self.features[int(meas_feature[2])]

        temp = (feature['pos'][0] - state[0, 0]) ** 2 + (feature['pos'][1] - state[1, 0]) ** 2
        self.H = np.array([[-(feature['pos'][0] - state[0, 0]) / math.sqrt(temp), -(feature['pos'][1] - state[1, 0]) / math.sqrt(temp), 0],
                           [(feature['pos'][1] - state[1, 0]) / temp, -(feature['pos'][0] - state[0, 0]) / math.sqrt(temp), -1],
                           [0, 0, 0]])
        return self.H