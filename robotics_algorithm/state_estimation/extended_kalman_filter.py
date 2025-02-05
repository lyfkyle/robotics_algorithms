import numpy as np

from robotics_algorithm.env.base_env import BaseEnv, EnvType, FunctionType, DistributionType


class ExtendedKalmanFilter:
    def __init__(self, env: BaseEnv):
        """
        State Transition: X = transition_func(x, u) + sigma
        Measurement Z = measurement_func(x) + delta
        R: Process covariance matrix from sigma
        Q. Measurement covariance matrix from delta
        """

        assert env.state_transition_type == EnvType.STOCHASTIC.value
        assert env.observability == EnvType.PARTIALLY_OBSERVABLE.value
        assert env.state_transition_func_type == FunctionType.GENERAL.value
        assert env.observation_func_type == FunctionType.GENERAL.value
        assert env.state_transition_dist_type == DistributionType.GAUSSIAN.value
        assert env.observation_dist_type == DistributionType.GAUSSIAN.value

        self.env = env
        self.state = np.array(env.state_space.sample())
        self.covariance = np.eye(env.state_space.state_size)

    def set_initial_state(self, state: list):
        """
        Set the initial state of filter.

        @param state, initial state
        """
        self.state = np.array(state)

    def get_state(self) -> list:
        return self.state.tolist()

    def run(self, action: list, obs: list):
        """
        Run one iteration of the Kalman filter.

        @param action, control
        @param obs, observation
        """
        self.predict(action)
        self.update(obs)

    def predict(self, action: list):
        """
        @param control, control
        """
        A = self.env.state_transition_jacobian(self.state.tolist(), action)
        mean_new_state, _ = self.env.state_transition_func(
            self.state.tolist(), action
        )  # non-linear transition_func
        new_covariance = A @ self.covariance @ A.transpose() + self.env.state_transition_cov_matrix

        self.state = np.array(mean_new_state)
        self.covariance = new_covariance

    def update(self, obs: list):
        """
        @param obs, observation
        """
        obs_np = np.array(obs)
        mean_obs, _ = self.env.observation_func(self.state.tolist())
        mean_obs_np = np.array(mean_obs)

        H = self.env.observation_jacobian(self.state.tolist(), obs)
        K = (
            self.covariance
            @ H.transpose()
            @ np.linalg.inv(H @ self.covariance @ H.transpose() + self.env.obs_cov_matrix)
        )
        new_state = self.state + K @ (obs_np - mean_obs_np)
        tmp_matrix = K @ H
        new_covariance = (np.eye(tmp_matrix.shape[0]) - tmp_matrix) @ self.covariance

        self.state = new_state
        self.covariance = new_covariance
