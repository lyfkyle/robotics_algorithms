import math
import numpy as np

class KalmanFilter:
    def __init__(self, initial_state, initial_covariance, A, B, R, H, Q):
        '''
        State Transition: X = AX + BU + sigma
        Measurement Z = HX + delta
        R: Process covariance matrix from sigma
        Q. Measurement covariance matrix from delta
        '''

        self.state = initial_state
        self.covariance = initial_covariance

        # state_transition_func: x = Ax + bu + epsilon
        self.A = A
        self.B = B
        self.R = R

        # measurement function z = Hx + delta
        self.H = H
        self.Q = Q

    def predict(self, control):
        '''
        @param control, control
        '''

        new_state = self.A @ self.state + self.B @ control
        new_covariance = self.A @ self.covariance @ self.A.transpose() + self.R

        self.state = new_state
        self.covariance = new_covariance

    def update(self, measurement):
        '''
        @param measurement, measurement
        '''

        K = self.covariance @ self.H.transpose() @ np.linalg.inv(self.H @ self.covariance @ self.H.transpose() + self.Q)
        new_state = self.state + K @ (measurement - self.H @ self.state)
        tmp_matrix = K @ self.H
        new_covariance = (np.eye(tmp_matrix.shape[0]) - tmp_matrix) @ self.covariance

        self.state = new_state
        self.covariance = new_covariance



