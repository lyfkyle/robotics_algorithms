import numpy as np
import cvxpy
import time

from robotics_algorithm.env.base_env import BaseEnv, FunctionType, SpaceType
from .lqr import LQR


class ConvexMPC:
    """Implements Convex Model Predictive Control

    Use convex optimization to explicitly solve the optimal sequence of action that minimizes cost of future trajectory,
    subject to linear state transition constraints.
    """
    def __init__(self, env: BaseEnv, horizon=100):
        """
        Constructor

        Args:
            env (BaseEnv): the env
            horizon (int, optional): the lookahead distance. Defaults to 100.

        Raises:
            AssertionError: if env does not satisfy the assumptions of ConvexMPC
        """
        assert env.state_space.type == SpaceType.CONTINUOUS.value
        assert env.action_space.type == SpaceType.CONTINUOUS.value
        assert env.state_transition_func_type == FunctionType.LINEAR.value
        assert env.reward_func_type == FunctionType.QUADRATIC.value

        self.env = env
        self.T = horizon

        # By default, default reference state and action is env.goal and zero action
        self.A, self.B = self.env.linearize_state_transition(env.goal_state, np.zeros(env.action_space.state_size))

    def set_ref_state_action(self, ref_state, ref_action):
        """
        Set the reference state and action for the current iteration.

        Args:
            ref_state (np.ndarray): the reference state
            ref_action (np.ndarray): the reference action
        """
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

        x = cvxpy.Variable((self.T + 1, self.env.state_space.state_size))
        u = cvxpy.Variable((self.T + 1, self.env.action_space.state_size))

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
