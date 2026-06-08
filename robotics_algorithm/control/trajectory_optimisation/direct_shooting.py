import numpy as np
import scipy

from robotics_algorithm.env.base_env import BaseEnv, EnvType, SpaceType


class DirectShooting:
    def __init__(
        self,
        env: BaseEnv,
        horizon=100,
        max_iter=1000,
        path_cost_w=1.0,
        terminal_cost_w=1e9,
    ):
        """Constructor.

        Args:
            env (BaseEnv): A planning env.
            horizon (int): number of control intervals in the trajectory.
            max_iter (int): maximum number of optimizer iterations.
            path_cost_w (float): weight for path cost.
            terminal_cost_w (float): weight for terminal cost.
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
        self.terminal_cost_w = terminal_cost_w
        self.optimization_result = None

    def run(
        self,
        start: np.ndarray,
        goal: np.ndarray,
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
        action_size = self.env.action_space.state_size

        # Define cost function
        def cost_func(decision_vars):
            state = start
            actions = decision_vars.reshape(self.horizon, action_size)

            path_cost = 0.0
            for i in range(self.horizon):
                next_state = self.env.state_transition_func(state, actions[i])
                path_cost += self.env.reward_func(state, actions[i], next_state) * -1.0
                state = next_state

            # Add terminal state cost to encourage reaching the goal
            terminal_cost = np.linalg.norm(state - goal)
            total_cost = self.path_cost_w * path_cost + self.terminal_cost_w * terminal_cost

            return total_cost

        def variable_bounds():
            def bound_pair(low, high):
                low_bound = None if np.isneginf(low) else float(low)
                high_bound = None if np.isposinf(high) else float(high)
                return low_bound, high_bound

            action_low = np.asarray(self.env.action_space.low).reshape(-1)
            action_high = np.asarray(self.env.action_space.high).reshape(-1)

            bounds = []
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

        initial_cost = cost_func(action_guess)
        print('Initial cost:', initial_cost)

        # Convert equality constraints to quadratic penalties. L-BFGS-B handles box bounds directly.
        res = scipy.optimize.minimize(
            cost_func,
            action_guess.flatten(),
            method='L-BFGS-B',
            bounds=variable_bounds(),
            options={'maxiter': self.max_iter, 'ftol': 1e-6, 'maxfun': 1000000},
        )
        self.optimization_result = res
        print(res)

        # Extract result
        action_path = res.x.reshape(self.horizon, action_size)
        success = res.success

        print('Final cost:', res.fun)

        state_path = [start]
        for action in action_path:
            next_state = self.env.state_transition_func(state_path[-1], action)
            state_path.append(next_state)

        return success, state_path, action_path, initial_cost, res.fun
