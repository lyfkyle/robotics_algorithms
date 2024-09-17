import math
import numpy as np

class ExtendedKalmanFilter:
    def __init__(self, initial_state, initial_covariance, transition_func, compute_jacobian_A, compute_R, measurement_func, compute_jacobian_H, Q):
        '''
        State Transition: X = AX + BU + sigma
        Measurement Z = HX + delta
        R: Process covariance matrix from sigma
        Q. Measurement covariance matrix from delta
        '''

        self.state = initial_state
        self.covariance = initial_covariance

        # state_transition_func: x = f(x, u)
        self._transition_func = transition_func
        self._compute_jacobian_A = compute_jacobian_A # jacobian of transition_func
        self._compute_R = compute_R

        # measurement function z = f(x)
        self._measurement_func = measurement_func
        self._compute_jacobian_H = compute_jacobian_H # jacobian of measurement_func
        self.Q = Q

    def predict(self, control):
        '''
        @param control, control
        '''
        A = self._compute_jacobian_A(self.state, control)
        new_state = self._transition_func(self.state, control) # non-linear transition_func
        new_covariance = A @ self.covariance @ A.transpose() + self._compute_R(self.state, control)

        self.state = new_state
        self.covariance = new_covariance

    def update(self, measurement):
        '''
        @param measurement, measurement
        '''
        H = self._compute_jacobian_H(self.state, measurement)
        K = self.covariance @ H.transpose() @ np.linalg.inv(H @ self.covariance @ H.transpose() + self.Q)
        new_state = self.state + K @ (measurement - self._measurement_func(self.state, measurement))
        tmp_matrix = K @ H
        new_covariance = (np.eye(tmp_matrix.shape[0]) - tmp_matrix) @ self.covariance

        self.state = new_state
        self.covariance = new_covariance