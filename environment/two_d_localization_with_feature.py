import copy
import math
import numpy as np
from numpy.random import randn

class TwoDLocalizationWithFeature(object):
    def __init__(self, control_noise=[0.01, 0.01], meas_noise=[0.01, 0.01, 0]):
        self.state = np.array([[5, 5, 0]])
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


    def control(self, control):
        '''

        '''
        vel = control[0]
        ang_vel = control[1]

        # add gaussian noise to velocity and ang_vel
        true_vel = vel + randn() * self.control_noise[0]
        true_ang_vel = vel + randn() * self.control_noise[1]

        # temp variables
        theta = self.state[0, 2]
        temp = true_vel / true_ang_vel
        new_state = self.state + np.array([[-temp * math.sin(theta) + temp * math.sin(theta + true_ang_vel * self.dt)],
                                           [temp * math.cos(theta) - temp * math.cos(theta + true_ang_vel * self.dt)],
                                           [true_ang_vel * self.dt]])

        self.state = new_state

    def sense(self):
        meas = np.array([])
        for i in range(self.num_of_features):
            feature = self.features[i]
            meas_feature = np.array([[math.sqrt((feature['loc'][0] - self.state[0, 0]) ** 2 + (feature['loc'][1] - self.state[0, 1]) ** 2)],
                                     [math.atan2(feature['loc'][1] - self.state[0, 1], feature['loc'][0] - self.state[0, 0]) - self.state[0, 2]],
                                     [feature['ind']]])
            meas_feature += np.array([[randn() * self.meas_noise[0]],
                                      [randn() * self.meas_noise[1]],
                                      [0]])
            meas = np.hstack([meas, meas_feature]) if meas.size else meas_feature

        return meas

    def control_and_sense(self, control):
        self.control(control)
        return self.sense()

    def control_jacobian(self, state, control):
        vel = control[0]
        ang_vel = control[1]
        theta = state[0, 2]

        temp = vel / ang_vel
        self.A = np.array([[1, 0, -temp * math.cos(theta) + temp * math.sin(theta + ang_vel * self.dt)],
                           [0, 1, -temp * math.sin(theta) + temp * math.sin(theta + ang_vel * self.dt)],
                           [0, 0, 1]])
        return self.A

    def process_noise_jacobian(self, state, control):
        vel = control[0]
        ang_vel = control[1]
        theta = state[0, 2]

        self.V = np.array([[(-math.sin(theta) + math.sin(theta + ang_vel * self.dt)) / ang_vel, vel * (math.sin(theta) - math.sin(theta + ang_vel * self.dt)) / (ang_vel ** 2) + vel * math.cos(theta + ang_vel * self.dt) * self.dt / ang_vel],
                           [(math.cos(theta) - math.cos(theta + ang_vel * self.dt)) / ang_vel, -vel * (math.cos(theta) - math.cos(theta + ang_vel * self.dt)) / (ang_vel ** 2) + vel * math.sin(theta + ang_vel * self.dt) * self.dt / ang_vel],
                           [0, self.dt]])
        return self.V

    def meas_jacobian(self, state, meas_feature):
        feature = self.features[meas_feature[2]]

        temp = (feature['loc'][0] - state[0, 0]) ** 2 + (feature['loc'][1] - state[0, 1]) ** 2
        self.H = np.array([[-(feature['loc'][0] - state[0, 0]) / math.sqrt(temp), -(feature['loc'][1] - state[0, 1]) / math.sqrt(temp), 0],
                           [(feature['loc'][1] - state[0, 1]) / temp, -(feature['loc'][0] - state[0, 0]) / math.sqrt(temp), -1],
                           [0, 0, 0]])
        return self.H