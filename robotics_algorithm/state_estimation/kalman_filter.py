import math
import numpy as np

from robotics_algorithm.env.base_env import BaseEnv, EnvType, FunctionType, DistributionType


class KalmanFilter:
    def __init__(self, env: BaseEnv):
        """
        Kalman filter.

        When environment has linear state transtion and linear observation function with Gaussian noise, bayesian
        filter rule can be simplified.

        State Transition: X = AX + BU + sigma
        Measurement Z = HX + delta
        """
        assert env.state_transition_type == EnvType.STOCHASTIC.value
        assert env.observability == EnvType.PARTIALLY_OBSERVABLE.value
        assert env.state_transition_func_type == FunctionType.LINEAR.value
        assert env.observation_func_type == FunctionType.LINEAR.value
        assert env.state_transition_dist_type == DistributionType.GAUSSIAN.value
        assert env.observation_dist_type == DistributionType.GAUSSIAN.value

        # state_transition_func: x = Ax + Bu + sigma
        self.A, self.B = env.linearize_state_transition(None, None)  # As state transition is linear, A and B are constant
        self.R = env.state_transition_covariance_matrix

        # measurement function z = Hx + delta
        self.H = env.linearize_observation(None, None)  # As observation is linear, H are constant
        self.Q = env.observation_covariance_matrix

        self.state = np.array(env.cur_state)
        self.covariance = np.eye(env.state_space.state_size)

    def set_initial_state(self, state: np.ndarray):
        """
        Set the initial state of filter.
        @param state, initial state
        """
        self.state = np.array(state)

    def get_state(self) -> np.ndarray:
        return self.state

    def run(self, action: np.ndarray, obs: np.ndarray):
        """
        Run one iteration of the Kalman filter.

        @param action, control
        @param obs, observation
        """
        self.predict(action)
        self.update(obs)

    def predict(self, action: np.ndarray):
        """
        Predict the state of the Kalman filter.

        @param action, applied action
        """
        new_state = self.A @ self.state + self.B @ action
        new_covariance = self.A @ self.covariance @ self.A.transpose() + self.R

        self.state = new_state
        self.covariance = new_covariance

    def update(self, obs: np.ndarray):
        """
        Update the state of the Kalman filter.

        @param obs, obtained observation.
        """
        K = self.covariance @ self.H.transpose() @ np.linalg.inv(self.H @ self.covariance @ self.H.transpose() + self.Q)
        new_state = self.state + K @ (obs - self.H @ self.state)
        tmp_matrix = K @ self.H
        new_covariance = (np.eye(tmp_matrix.shape[0]) - tmp_matrix) @ self.covariance

        self.state = new_state
        self.covariance = new_covariance
