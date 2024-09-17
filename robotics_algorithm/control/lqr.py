import numpy as np

# Given a model governed by state-space equation and a cost function that is quadratic to action and state,
# the optimal control strategy is u = -Kx

class LQR(object):
    def __init__(self, A, B, Q, R):
        '''
        State transition:
        x_dot = A * X + B * U
        y = x (full state feedback)
        J = sum_over_time(X_T*Q*X + U_T*R*U)
        '''
        self.Q = Q
        self.R = R
        self.A = A
        self.B = B

    def solve(self, N):
        Q, R, A, B = self.Q, self.R, self.A, self.B

        # lists for P and K where element t represents P_t and K_t, respectively
        P = []
        K = []

        # P_N = Q_f
        P.insert(0, Q)

        # K_N = 0
        K.insert(0, 0)

        # work backwards from N-1 to 0
        for _ in range(N-1, -1, -1):
            # calculate K_t
            # K_t = -(R+B^T*P_(t+1)*B)^(-1)*B^T*P_(t+1)*A
            temp1 = np.linalg.inv(R+np.dot(B.T, np.dot(P[0], B)))
            temp2 = np.dot(B.T, np.dot(P[0], A))
            K_t = np.dot(temp1, temp2)
            # K_t = np.dot(K_t, -1)

            # calculate P_t
            # P_t  = Q + A^T*P_(t+1)*A - A^T*P_(t+1)*B*(R + B^T  P_(t+1) B)^(-1) B^T*P_(t+1)*A
            P_t = Q + np.dot(A.T, np.dot(P[0], A)) - np.dot(A.T, np.dot(P[0], B)) * K_t

            # add our calculations to the beginning of the lists (since we are working backwards)
            K.insert(0, K_t)
            P.insert(0, P_t)

        return P, K

    def get_K(self, N):
        _, K = self.solve(N)
        return K[0]