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
        assert (
            env.state_transition_func_type == FunctionType.LINEAR.value
            or env.state_transition_func_type == FunctionType.GENERAL.value
        )
        assert env.reward_func_type == FunctionType.QUADRATIC.value

        self.env = env
        self.horizon = horizon

        # ! Use LQR to calculate terminal cost. This makes sense as we assume after horizon, the system is brought
        # ! to good state where all constraints should be satisfied.
        self._lqr = LQR(env)

        # By default, default reference state and action is env.goal and zero action
        self.A, self.B = self.env.linearize_state_transition(env.goal_state, env.goal_action)

        # Set up default constraints on input and state
        self.default_constraints = []
        self.x = cvxpy.Variable((self.horizon + 1, self.env.state_space.state_size))
        self.u = cvxpy.Variable((self.horizon + 1, self.env.action_space.state_size))

    def set_ref_state_action(self, ref_state, ref_action):
        """
        Set the reference state and action for the current iteration.

        Args:
            ref_state (np.ndarray): the reference state
            ref_action (np.ndarray): the reference action
        """
        self.A, self.B = self.env.linearize_state_transition(ref_state, ref_action)

    def get_decision_variables(self):
        return self.x, self.u

    def add_constraints(self, constraint):
        if isinstance(constraint, list):
            self.default_constraints += constraint
        else:
            self.default_constraints.append(constraint)

    def run(self, state: np.ndarray) -> np.ndarray:
        """Compute the current action based on the current state.

        Args:
            state (np.ndarray): current state.

        Returns:
            np.ndarray: current action
        """
        A, B = self.A, self.B
        Q, R = self.env.Q, self.env.R

        # use lqr to get terminal cost
        P = self._lqr._solve_dare(A, B, Q, R)

        constr = self.default_constraints.copy()
        cost = 0.0
        for t in range(self.horizon):
            cost += cvxpy.quad_form(self.x[t], Q)
            cost += cvxpy.quad_form(self.u[t], R)
            constr += [self.x[t + 1].T == A @ self.x[t] + B @ self.u[t]]

            # default constraints on state space and action space size
            # constr += [self.x[t] <= self.env.state_space.high, self.x[t] >= self.env.state_space.low]
            # constr += [self.u[t] <= self.env.action_space.high, self.u[t] >= self.env.action_space.low]

        cost += cvxpy.quad_form(self.x[self.horizon], P)  # LQR terminal cost
        constr += [self.x[0] == state]
        prob = cvxpy.Problem(cvxpy.Minimize(cost), constr)

        start = time.time()
        prob.solve(verbose=False)
        elapsed_time = time.time() - start
        print(f'calc time:{elapsed_time:.6f} [sec]')

        print('status:', prob.status)
        # print("optimal value", prob.value)
        # print('optimal var')
        # print(self.x.value)
        # print(self.u.value)

        return self.u.value[0]
