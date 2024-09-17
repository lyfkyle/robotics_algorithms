import numpy as np
import scipy
import scipy.linalg

from robotics_algorithm.env.base_env import DeterministicEnv


class LQR(object):
    """Implements Linear-quadratic regulator.

    # Given a system with linear state transition function and quadratic cost function.
    # the optimal control strategy is u = -Kx
    """

    def __init__(self, env: DeterministicEnv):
        """
        State transition:
        x_dot = A * X + B * U
        y = x (full state feedback)
        J = sum_over_time(X_T*Q*X + U_T*R*U)
        """
        assert env.state_transition_type == "linear"
        assert env.reward_func_type == "quadratic"

        self.env = env

    def run(self, state: list) -> list:
        """Compute the current action based on the current state.

        Args:
            state (list): current state.

        Returns:
            list: current action
        """
        state = np.array(state)
        Q, R, A, B = self.env.Q, self.env.R, self.env.A, self.env.B

        P = self._solve_care(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P

        u = -K @ state
        return u.tolist()

    def _solve_care(self, A, B, Q, R):
        return scipy.linalg.solve_continuous_are(A, B, Q, R)

    def _solve_dare(self, A, B, Q, R):
        # we can call scipy library.
        # return scipy.linalg.solve_discrete_are(A, B, Q, R)

        # Alternatively, discrete-time algebraic Riccati equation can be solved by iteration
        # lists for P and K where element t represents P_t and K_t, respectively
        P = []
        K = []

        # work backwards from N-1 to 0
        for _ in range(1000 - 1, -1, -1):
            # calculate K_t
            # K_t = -(R+B^T*P_(t+1)*B)^(-1)*B^T*P_(t+1)*A
            temp1 = np.linalg.inv(R + np.dot(B.T, np.dot(P[0], B)))
            temp2 = np.dot(B.T, np.dot(P[0], A))
            K_t = np.dot(temp1, temp2)
            # K_t = np.dot(K_t, -1)

            # calculate P_t
            # P_t  = Q + A^T*P_(t+1)*A - A^T*P_(t+1)*B*(R + B^T  P_(t+1) B)^(-1) B^T*P_(t+1)*A
            P_t = Q + np.dot(A.T, np.dot(P[0], A)) - np.dot(A.T, np.dot(P[0], B)) * K_t

            # add our calculations to the beginning of the lists (since we are working backwards)
            K.insert(0, K_t)
            P.insert(0, P_t)

        # P_N = Q_f
        P.insert(0, Q)

        # K_N = 0
        K.insert(0, 0)

        return K[0]