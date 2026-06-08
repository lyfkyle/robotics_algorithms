import numpy as np
import scipy

from robotics_algorithm.env.base_env import BaseEnv, EnvType, SpaceType


class DirectCollocation:
    def __init__(
        self,
        env: BaseEnv,
        horizon=100,
        max_iter=1000,
        path_cost_w=1.0,
        dynamics_cost_w=1e9,
        constraint_tolerance=1e-3,
    ):
        """Constructor.

        Args:
            env (BaseEnv): A planning env.
            horizon (int): number of control intervals in the trajectory.
            max_iter (int): maximum number of optimizer iterations.
            path_cost_w (float): weight for path cost.
            dynamics_cost_w (float): weight for dynamics constraint violations.
            constraint_tolerance (float): maximum accepted constraint violation.
        """
        assert horizon >= 1
        assert env.action_space.type == SpaceType.CONTINUOUS.value
        assert env.state_space.type == SpaceType.CONTINUOUS.value
        assert env.state_transition_type == EnvType.DETERMINISTIC.value
        assert env.observability == EnvType.FULLY_OBSERVABLE.value

        self.env = env
        self.horizon = horizon
        self.max_iter = max_iter
        self.path_cost_w = path_cost_w
        self.dynamics_cost_w = dynamics_cost_w
        self.constraint_tolerance = constraint_tolerance
        self.optimization_result = None

    def run(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        initial_state_path: np.ndarray = None,
        initial_action_path: np.ndarray = None,
    ) -> tuple[bool, np.ndarray, float]:
        """Run algorithm.

        Args:
            start (np.ndarray): the start state
            goal (np.ndarray): the goal state
            initial_state_path (np.ndarray): initial guess for the state trajectory
            initial_action_path (np.ndarray): initial guess for the action trajectory

        Returns:
            res (bool): return true if optimisation is found, return false otherwise.
            state_path (np.ndarray): optimized state trajectory.
            cost (float): optimized trajectory cost.
        """
        start = np.asarray(start)
        goal = np.asarray(goal)
        state_size = self.env.state_space.state_size
        action_size = self.env.action_space.state_size

        # * scipy expects a flat array
        # * The first few variables represent state guess, followed by action guess
        intermediate_state_var_size = (self.horizon - 1) * state_size

        def unpack(decision_vars):
            intermediate_states = decision_vars[:intermediate_state_var_size].reshape(self.horizon - 1, state_size)
            states = np.vstack([start, intermediate_states, goal])
            actions = decision_vars[intermediate_state_var_size:].reshape(self.horizon, action_size)
            return states, actions

        # Define cost function
        def cost_func(decision_vars):
            states, actions = unpack(decision_vars)

            path_cost = 0.0
            for i in range(self.horizon):
                path_cost += self.env.reward_func(states[i], actions[i], states[i + 1]) * -1.0

            dynamics_cost = calc_dynamics_cost(states, actions)

            total_cost = self.path_cost_w * path_cost + self.dynamics_cost_w * dynamics_cost

            return total_cost

        def calc_dynamics_cost(states, actions):
            dynamic_cost = 0
            for i in range(self.horizon):
                predicted_state = self.env.state_transition_func(states[i], actions[i])
                diff = states[i + 1] - predicted_state
                dynamic_cost += np.dot(diff, diff)

            return dynamic_cost

        def variable_bounds():
            def bound_pair(low, high):
                low_bound = None if np.isneginf(low) else float(low)
                high_bound = None if np.isposinf(high) else float(high)
                return low_bound, high_bound

            state_low = np.asarray(self.env.state_space.low).reshape(-1)
            state_high = np.asarray(self.env.state_space.high).reshape(-1)
            action_low = np.asarray(self.env.action_space.low).reshape(-1)
            action_high = np.asarray(self.env.action_space.high).reshape(-1)

            bounds = []
            for _ in range(self.horizon - 1):
                for low, high in zip(state_low, state_high):
                    bounds.append(bound_pair(low, high))

            for _ in range(self.horizon):
                for low, high in zip(action_low, action_high):
                    bounds.append(bound_pair(low, high))

            return bounds

        # initial guess
        if initial_action_path is not None:
            assert initial_action_path.shape == (self.horizon, action_size)
            action_guess = initial_action_path
        else:
            action_guess = np.zeros((self.horizon, action_size))

        if initial_state_path is not None:
            assert initial_state_path.shape == (self.horizon + 1, state_size)
            state_guess = initial_state_path
        else:
            state_guess = np.linspace(start, goal, self.horizon + 1)

        initial_guess = np.concatenate([state_guess[1:-1].reshape(-1), action_guess.reshape(-1)])
        initial_cost = cost_func(initial_guess)
        print('Initial cost:', initial_cost)

        # Convert equality constraints to quadratic penalties. L-BFGS-B handles box bounds directly.
        res = scipy.optimize.minimize(
            cost_func,
            initial_guess,
            method='L-BFGS-B',
            bounds=variable_bounds(),
            options={'maxiter': self.max_iter, 'ftol': 1e-6, 'maxfun': 1000000},
        )
        self.optimization_result = res
        print(res)

        # Extract result
        state_path, action_path = unpack(res.x)
        total_dynamic_cost = calc_dynamics_cost(state_path, action_path)
        success = res.success and total_dynamic_cost < self.constraint_tolerance

        final_cost = res.fun
        print('Final cost:', res.fun, 'final_dynamic_cost:', total_dynamic_cost)

        return success, state_path, action_path, initial_cost, final_cost
