import numpy as np
import scipy
import scipy.linalg

from robotics_algorithm.env.base_env import BaseEnv, FunctionType, SpaceType


class LQR:
    """Implements Linear-quadratic regulator.

    Given a system with linear state transition function and quadratic cost function, the optimal control strategy
    can be computed analytically and has the form: u = -Kx
    """
    def __init__(self, env: BaseEnv, discrete_time=True, horizon=float("inf"), solve_by_iteration=False):
        """
        State transition:
        x_dot = A * X + B * U
        y = x (full state feedback)
        J = sum_over_time(X_T*Q*X + U_T*R*U)
        """
        assert env.state_space.type == SpaceType.CONTINUOUS.value
        assert env.action_space.type == SpaceType.CONTINUOUS.value
        # Can be general because we can linearize dynamics around fixed point.
        assert env.state_transition_func_type == FunctionType.LINEAR.value or env.state_transition_func_type == FunctionType.GENERAL.value
        assert env.reward_func_type == FunctionType.QUADRATIC.value

        self.env = env
        self.discrete_time = discrete_time
        self.horizon = horizon
        self.solve_by_iteration = solve_by_iteration

        # By default, default reference state and action is env.goal and zero action
        self.A, self.B = self.env.linearize_state_transition(env.goal_state, env.goal_action)

    def set_ref_state_action(self, ref_state, ref_action):
        self.A, self.B = self.env.linearize_state_transition(ref_state, ref_action)

    def run(self, state: np.ndarray) -> np.ndarray:
        """Compute the current action based on the current state.

        Args:
            state (np.ndarray): current state.

        Returns:
            np.ndarray: current action
        """
        A, B = self.A, self.B
        Q, R = self.env.Q, self.env.R
        state = np.array(state).reshape(-1, 1)

        if self.discrete_time:
            P = self._solve_dare(A, B, Q, R)
            K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P
        else:
            P = self._solve_care(A, B, Q, R)
            K = np.linalg.inv(R) @ B.T @ P

        u = -K @ state
        return u.reshape(-1)

    def _solve_care(self, A, B, Q, R):
        return scipy.linalg.solve_continuous_are(A, B, Q, R)

    def _solve_dare(self, A, B, Q, R):
        # we can call scipy library.
        if self.horizon == float("inf") or not self.solve_by_iteration:
            return scipy.linalg.solve_discrete_are(A, B, Q, R)

        # Alternatively, discrete-time algebraic Riccati equation can be solved by iteration

        # lists for P that contains P_t
        P = [Q]

        # work backwards in time from horizon to 0
        for _ in range(self.horizon):
            # calculate K_t
            # K_t = -(R+B^T*P_(t+1)*B)^(-1)*B^T*P_(t+1)*A
            K_t = np.linalg.inv(R + B.T @ P[-1] @ B) @ B.T @ P[-1] @ A

            # calculate P_t
            # P_t  = Q + A^T*P_(t+1)*A - A^T*P_(t+1)*B*(R + B^T  P_(t+1) B)^(-1) B^T*P_(t+1)*A
            P_t = Q + A.T @ P[-1] @ A - A.T @ P[-1] @ B @ K_t

            # add our calculations to the beginning of the lists (since we are working backwards)
            P.append(P_t)

        # P_N = Q_f
        # P.insert(0, Q)

        # K_N = 0
        # K.insert(0, 0)

        return P[-1]
