import math
import numpy as np

class ExtendedKalmanFilter:
    def __init__(self, initial_state, initial_covariance, transition_func, compute_control_jacobian, compute_control_noise_matrix, measurement_func, compute_meas_jacobian, meas_noise_matrix):
        '''
        State Transition: X = transition_func(x, u) + sigma
        Measurement Z = measurement_func(x) + delta
        R: Process covariance matrix from sigma
        Q. Measurement covariance matrix from delta
        '''

        self.state = initial_state
        self.covariance = initial_covariance

        # state_transition_func: x = f(x, u)
        self._transition_func = transition_func
        self._compute_control_jacobian = compute_control_jacobian # jacobian of transition_func
        self._compute_control_noise_matrix = compute_control_noise_matrix

        # measurement function z = f(x)
        self._measurement_func = measurement_func
        self._compute_meas_jacobian = compute_meas_jacobian # jacobian of measurement_func
        self.meas_noise_matrix = meas_noise_matrix

    def predict(self, control):
        '''
        @param control, control
        '''
        A = self._compute_control_jacobian(self.state, control)
        new_state = self._transition_func(self.state, control) # non-linear transition_func
        new_covariance = A @ self.covariance @ A.transpose() + self._compute_control_noise_matrix(self.state, control)

        self.state = new_state
        self.covariance = new_covariance

    def update(self, measurement):
        '''
        @param measurement, measurement
        '''
        H = self._compute_meas_jacobian(self.state)
        K = self.covariance @ H.transpose() @ np.linalg.inv(H @ self.covariance @ H.transpose() + self.meas_noise_matrix)
        new_state = self.state + K @ (measurement - self._measurement_func(self.state))
        tmp_matrix = K @ H
        new_covariance = (np.eye(tmp_matrix.shape[0]) - tmp_matrix) @ self.covariance

        self.state = new_state
        self.covariance = new_covariance