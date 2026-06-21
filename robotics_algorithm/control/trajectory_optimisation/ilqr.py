import numpy as np

from robotics_algorithm.env.base_env import BaseEnv, EnvType, SpaceType


class iLQR:
    def __init__(
        self,
        env: BaseEnv,
        horizon=100,
        max_iter=100,
        enforce_bounds=True,
    ):
        """Constructor.

        Args:
            env (BaseEnv): A planning env.
            horizon (int): number of control intervals in the trajectory.
            max_iter (int): maximum number of optimizer iterations.
            enforce_bounds (bool): if True, clip actions to environment bounds in forward pass.
        """
        assert horizon >= 1
        assert env.action_space.type == SpaceType.CONTINUOUS.value
        assert env.state_space.type == SpaceType.CONTINUOUS.value
        assert env.state_transition_type == EnvType.DETERMINISTIC.value
        assert env.observability == EnvType.FULLY_OBSERVABLE.value

        self.env = env
        self.horizon = horizon
        self.max_iter = max_iter
        self.enforce_bounds = enforce_bounds

        self.init_regu_mod_factor = 2.0
        self.min_regu = 1e-6

        if self.enforce_bounds:
            print(
                '[W]/iQR: enforce bounds will clip new action and new state during forward pass. This might affect'
                'correctness of optimisation'
            )

    def run(
        self,
        start: np.ndarray,
        initial_action_path: np.ndarray = None,
    ) -> tuple[bool, np.ndarray, float]:
        """Run algorithm.

        Args:
            start (np.ndarray): the start state
            initial_action_path (np.ndarray, optional): the initial action path

        Returns:

        """
        assert len(initial_action_path) == self.horizon

        state_path = np.empty((self.horizon + 1, self.env.state_space.state_size))
        action_path = initial_action_path.copy()

        # Compute old cost and initial state path based on the initial action path
        old_cost = 0
        state_path[0] = start.copy()
        for n in range(self.horizon):
            state_path[n + 1] = self.env.state_transition_func(state_path[n], action_path[n])
            old_cost -= self.env.reward_func(state_path[n], action_path[n], state_path[n + 1])

        print('[I]/iLQR: Initial traj cost is ', old_cost)

        regu = 0.1
        regu_mod_factor = self.init_regu_mod_factor
        ks = np.empty(action_path.shape)
        Ks = np.empty((action_path.shape[0], action_path.shape[1], state_path.shape[1]))

        # Iterate until convergence
        for _ in range(self.max_iter):
            iter_success = True

            # Quadratic approximation of the cost-to-go function
            # V(x0 + dx) = V(x0) + V_x.T @ dx + 0.5 * dx.T @ V_xx @ dx
            # At terminal state, V(x) = l(x), so V_x = l_x, V_xx = l_xx
            V_x, _ = self.env.reward_jacobian(state_path[-1], action_path[-1])
            V_xx, _ = self.env.reward_hessian(state_path[-1], action_path[-1])
            V_x = -V_x  # flip due to reward to cost
            V_xx = -V_xx  # flip due to reward to cost

            # backward pass
            regu_I = regu * np.eye(V_xx.shape[0])
            expected_delta_V = 0
            for n in range(self.horizon - 1, -1, -1):
                f_x, f_u = self.env.state_transition_jacobian(state_path[n], action_path[n])
                # reward depends on the next state in this env API, so evaluate at state_path[n+1]
                l_x, l_u = self.env.reward_jacobian(state_path[n + 1], action_path[n])
                l_x = -l_x  # flip due to reward to cost
                l_u = -l_u  # flip due to reward to cost
                l_xx, l_uu = self.env.reward_hessian(state_path[n + 1], action_path[n])
                l_xx = -l_xx  # flip due to reward to cost
                l_uu = -l_uu  # flip due to reward to cost
                l_ux = np.zeros((l_u.shape[0], l_x.shape[0]))
                # print(f_x.shape, f_u.shape, l_x.shape, l_u.shape, l_xx.shape, l_uu.shape)

                # Q_terms for the cost-to-go function
                # This is due to Bellman Equation
                # Q(u, x) = L(u, x) + V'(f(x, u))
                Q_x = l_x + f_x.T @ V_x
                Q_u = l_u + f_u.T @ V_x
                Q_xx = l_xx + f_x.T @ V_xx @ f_x
                Q_ux = l_ux + f_u.T @ V_xx @ f_x
                Q_uu = l_uu + f_u.T @ V_xx @ f_u

                # ensure symmetry
                # ! Important for numerical stability
                Q_xx = 0.5 * (Q_xx + Q_xx.T)
                Q_uu = 0.5 * (Q_uu + Q_uu.T)

                # Apply regularisation.
                # * Regularisation is applied to V not Q according to original paper
                # * https://roboti.us/lab/papers/TassaIROS12.pdf
                f_u_dot_regu = f_u.T @ regu_I
                Q_ux_regu = Q_ux + f_u_dot_regu @ f_x
                Q_uu_regu = Q_uu + f_u_dot_regu @ f_u
                try:
                    Q_uu_inv = np.linalg.inv(Q_uu_regu)
                except np.linalg.LinAlgError:
                    print('Q_uu_regu is not invertible, increase regularization')
                    iter_success = False
                    break

                # Gains
                k = -Q_uu_inv @ Q_u
                K = -Q_uu_inv @ Q_ux_regu
                ks[n], Ks[n] = k, K

                # V_terms
                V_x = Q_x + K.T @ Q_u + Q_ux.T @ k + K.T @ Q_uu @ k
                V_xx = Q_xx + K.T @ Q_ux + Q_ux.T @ K + K.T @ Q_uu @ K

                # ensure symmetry
                # ! Important for numerical stability
                V_xx = 0.5 * (V_xx + V_xx.T)

                # expected cost reduction
                expected_delta_V += Q_u.T @ k + 0.5 * k.T @ Q_uu @ k

            # early stop if expected cost reduction is small
            if expected_delta_V > 0:
                print(expected_delta_V)
                print('Expected cost is found to increase!!.')
                break
            if expected_delta_V > -1e-6:
                print('Expected cost reduction is small, stopping optimization.')
                break

            # forward pass
            # line search, if cost does not reduce, reduce alpha
            alpha = 1.0
            while iter_success and alpha > 1e-4:
                new_state_path = np.empty(state_path.shape)
                new_action_path = np.empty(action_path.shape)

                new_state_path[0] = start
                new_cost = 0
                for n in range(self.horizon):
                    new_action_path[n] = action_path[n] + alpha * ks[n] + Ks[n] @ (new_state_path[n] - state_path[n])

                    # Enforce action bounds if requested
                    if self.enforce_bounds:
                        new_action_path[n] = np.clip(
                            new_action_path[n], self.env.action_space.low, self.env.action_space.high
                        )
                    elif (
                        new_action_path[n] < self.env.action_space.low
                        or new_action_path[n] > self.env.action_space.high
                    ):
                        print('[W]/iLQR: action exceeds bounds in forward pass!')

                    new_state_path[n + 1] = self.env.state_transition_func(new_state_path[n], new_action_path[n])

                    # Enforce action bounds if requested
                    if self.enforce_bounds:
                        new_state_path[n + 1] = np.clip(
                            new_state_path[n + 1], self.env.state_space.low, self.env.state_space.high
                        )
                    elif (
                        new_state_path[n + 1] < self.env.state_space.low
                        or new_state_path[n + 1] > self.env.state_space.high
                    ):
                        print('[W]/iLQR: state exceeds bounds in forward pass!')

                    new_cost -= self.env.reward_func(new_state_path[n], new_action_path[n], new_state_path[n + 1])

                # compute cost reduction
                actual_reduction = old_cost - new_cost

                if actual_reduction > 0:
                    state_path, action_path = new_state_path, new_action_path
                    break
                else:
                    alpha *= 0.5
            else:
                iter_success = False

            print(
                f'Iter success: {iter_success}, old_cost: {old_cost:.4f}, new_cost: {new_cost:.4f}, expected reduction: {-expected_delta_V:.4f}, actual reduction: {actual_reduction:.4f}, alpha: {alpha:.4f}, regu: {regu:.4f}'
            )

            # update regularization
            if iter_success:
                # decrease regularization if iteration is successful
                old_cost = new_cost
                regu_mod_factor = min(regu_mod_factor, 1.0) / self.init_regu_mod_factor
                regu *= regu_mod_factor
                if regu < self.min_regu:
                    regu = 0
            else:
                # increase regularization
                regu_mod_factor = max(self.init_regu_mod_factor, regu_mod_factor * 2.0)
                regu = max(self.min_regu, regu * regu_mod_factor)

        return state_path, action_path
