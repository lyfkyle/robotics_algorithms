import numpy as np
import cvxpy
import time

from robotics_algorithm.env.base_env import BaseEnv, FunctionType, SpaceType


class ConvexMPC:

    def __init__(self, env: BaseEnv, horizon=100):
        assert env.state_space.type == SpaceType.CONTINUOUS.value
        assert env.action_space.type == SpaceType.CONTINUOUS.value
        assert env.state_transition_func_type == FunctionType.LINEAR.value
        assert env.reward_func_type == FunctionType.QUADRATIC.value

        self.env = env
        self.T = horizon

    def run(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Compute the current action based on the current state.

        Args:
            state (list): current state.

        Returns:
            list: current action
        """
        A, B = self.env.linearize_state_transition(self.env.goal_state, action)
        Q, R = self.env.Q, self.env.R

        x = cvxpy.Variable((self.T + 1, state.shape[0]))
        u = cvxpy.Variable((self.T + 1, action.shape[0]))

        cost = 0.0
        constr = []
        for t in range(self.T):
            cost += cvxpy.quad_form(x[t], Q)
            cost += cvxpy.quad_form(u[t], R)
            constr += [x[t + 1].T == A @ x[t] + B @ u[t]]

        constr += [x[0] == state]
        prob = cvxpy.Problem(cvxpy.Minimize(cost), constr)

        start = time.time()
        prob.solve(verbose=False)
        elapsed_time = time.time() - start
        print(f"calc time:{elapsed_time:.6f} [sec]")

        print("status:", prob.status)
        # print("optimal value", prob.value)
        # print("optimal var", x.value, u.value)

        return np.array(u.value[0])