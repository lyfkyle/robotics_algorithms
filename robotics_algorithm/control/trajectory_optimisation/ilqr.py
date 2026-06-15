import numpy as np
import scipy

from robotics_algorithm.env.base_env import BaseEnv, EnvType, SpaceType


class iLQR:
    def __init__(
        self,
        env: BaseEnv,
        horizon=100,
        max_iter=100,
        debug: bool = False,
    ):
        """Constructor.

        Args:
            env (BaseEnv): A planning env.
            horizon (int): number of control intervals in the trajectory.
            max_iter (int): maximum number of optimizer iterations.
        """
        assert horizon >= 1
        assert env.action_space.type == SpaceType.CONTINUOUS.value
        assert env.state_space.type == SpaceType.CONTINUOUS.value
        assert env.state_transition_type == EnvType.DETERMINISTIC.value
        assert env.observability == EnvType.FULLY_OBSERVABLE.value

        self.env = env
        self.horizon = horizon
        self.max_iter = max_iter
        self.debug = debug

        self.init_regu_mod_factor = 2.0
        self.min_regu = 1e-6

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

        # Quadratic approximation of the cost-to-go function
        # V(x0 + dx) = V(x0) + V_x.T @ dx + 0.5 * dx.T @ V_xx @ dx
        # At terminal state, V(x) = l(x), so V_x = l_x, V_xx = l_xx
        V_x, _ = self.env.reward_jacobian(state_path[-1], action_path[-1])
        V_xx, _ = self.env.reward_hessian(state_path[-1], action_path[-1])
        V_x = -V_x  # flip due to reward to cost
        V_xx = -V_xx  # flip due to reward to cost

        regu = 0.1
        regu_mod_factor = self.init_regu_mod_factor
        ks = np.empty(action_path.shape)
        Ks = np.empty((action_path.shape[0], action_path.shape[1], state_path.shape[1]))

        for _ in range(self.max_iter):
            iter_success = True

            # backward pass
            regu_I = regu * np.eye(V_xx.shape[0])
            expected_delta_V = 0
            for n in range(self.horizon - 1, -1, -1):
                f_x, f_u = self.env.state_transition_jacobian(state_path[n], action_path[n])
                # reward depends on the next state in this env API, so evaluate at state_path[n+1]
                l_x, l_u = self.env.reward_jacobian(state_path[n + 1], action_path[n])
                # flip first-order reward derivatives to cost derivatives
                l_x = -l_x
                l_u = -l_u
                l_xx, l_uu = self.env.reward_hessian(state_path[n + 1], action_path[n])
                l_xx = -l_xx  # flip due to reward to cost
                l_uu = -l_uu  # flip due to reward to cost
                l_ux = np.zeros((l_u.shape[0], l_x.shape[0]))
                # print(f_x.shape, f_u.shape, l_x.shape, l_u.shape, l_xx.shape, l_uu.shape)

                # Q_terms for the cost-to-go function
                Q_x = l_x + f_x.T @ V_x
                Q_u = l_u + f_u.T @ V_x
                Q_xx = l_xx + f_x.T @ V_xx @ f_x
                Q_ux = l_ux + f_u.T @ V_xx @ f_x
                Q_uu = l_uu + f_u.T @ V_xx @ f_u

                # gains
                f_u_dot_regu = f_u.T @ regu_I
                Q_ux_regu = Q_ux + f_u_dot_regu @ f_x
                Q_uu_regu = Q_uu + f_u_dot_regu @ f_u
                try:
                    Q_uu_inv = np.linalg.inv(Q_uu_regu)
                except np.linalg.LinAlgError:
                    # if self.debug:
                    #     try:
                    #         Q_sym = 0.5 * (Q_uu_regu + Q_uu_regu.T)
                    #         eigs = np.linalg.eigvalsh(Q_sym)
                    #         print(f'Backward n={n}: Q_uu_regu symmetric eigenvalues: {eigs}')
                    #     except Exception:
                    #         print(f'Backward n={n}: Q_uu_regu not invertible (no eigenvalues)')
                    # print('Q_uu_regu is not invertible, increase regularization')
                    iter_success = False
                    break

                k = -Q_uu_inv @ Q_u
                K = -Q_uu_inv @ Q_ux_regu
                ks[n], Ks[n] = k, K

                # # debug diagnostics for backward pass
                # if self.debug:
                #     try:
                #         q_u_norm = float(np.linalg.norm(Q_u))
                #     except Exception:
                #         q_u_norm = float('nan')
                #     try:
                #         Q_uu_sym = 0.5 * (Q_uu + Q_uu.T)
                #         min_eig_Q_uu = float(np.min(np.linalg.eigvalsh(Q_uu_sym)))
                #     except Exception:
                #         min_eig_Q_uu = float('nan')
                #     try:
                #         Q_uu_regu_sym = 0.5 * (Q_uu_regu + Q_uu_regu.T)
                #         min_eig_Q_uu_reg = float(np.min(np.linalg.eigvalsh(Q_uu_regu_sym)))
                #         cond_Q_uu_reg = float(np.linalg.cond(Q_uu_regu_sym))
                #     except Exception:
                #         min_eig_Q_uu_reg = float('nan')
                #         cond_Q_uu_reg = float('nan')
                #     try:
                #         k_norm = float(np.linalg.norm(k))
                #     except Exception:
                #         k_norm = float('nan')
                #     try:
                #         step_reduction = float(Q_u.T @ k + 0.5 * k.T @ Q_uu @ k)
                #     except Exception:
                #         step_reduction = float('nan')

                #     print(
                #         f'Backward n={n}: ||Q_u||={q_u_norm:.4e}, min_eig(Q_uu)={min_eig_Q_uu:.4e}, '
                #         f'min_eig(Q_uu_reg)={min_eig_Q_uu_reg:.4e}, cond(Q_uu_reg)={cond_Q_uu_reg:.4e}, '
                #         f'||k||={k_norm:.4e}, step_red={step_reduction:.4e}'
                #     )

                # V_terms
                V_x = Q_x + K.T @ Q_u + Q_ux.T @ k + K.T @ Q_uu @ k
                V_xx = Q_xx + K.T @ Q_ux + Q_ux.T @ K + K.T @ Q_uu @ K
                # V_xx = 0.5 * (V_xx + V_xx.T)  # ensure symmetry

                # expected cost reduction
                expected_delta_V += Q_u.T @ k + 0.5 * k.T @ Q_uu @ k

            # early stop if expected cost reduction is small
            if expected_delta_V > -1e-6:
                print('Expected cost reduction is small, stopping optimization.')
                break

            # forward pass
            # Alpha line search (allow smaller steps)
            alpha = 1.0
            while iter_success and alpha > 1e-4:
                new_state_path = np.empty(state_path.shape)
                new_action_path = np.empty(action_path.shape)

                new_state_path[0] = start
                new_cost = 0
                for n in range(self.horizon):
                    new_action_path[n] = action_path[n] + alpha * ks[n] + Ks[n] @ (new_state_path[n] - state_path[n])
                    new_state_path[n + 1] = self.env.state_transition_func(new_state_path[n], new_action_path[n])
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
                # decrease regularization
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
