import numpy as np
from scipy.optimize import minimize


class ArcLengthYawTrajOpt:
    def __init__(
        self,
        M=5,
        K=10,
        robot_radius=0.25,      # Radius of the two approximation circles
        circle_dist=0.25,       # Distance of each circle center from robot center along heading
        obstacles=None,
        max_v=2.0,
        max_w=2.0,
        max_a=2.0,
        max_dw=2.0,
        collision_weight=100.0,
        smooth_weight=1.0,
        limit_weight=10.0,
        time_weight=1.0,
        goal_pos_weight=100.0,  # Initial rho value for Augmented Lagrangian
    ):
        """Constructor for ArcLengthYawTrajOpt.

        Args:
            M (int): Number of trajectory segments/pieces.
            K (int): Number of integration intervals per segment for Simpson's rule.
            robot_radius (float): Radius of the two circles approximating the rectangular robot.
            circle_dist (float): Distance of each circle center along the robot centerline from its center.
            obstacles (list of tuple): List of circular obstacles as (x, y, r).
            max_v (float): Maximum linear velocity limit.
            max_w (float): Maximum angular velocity limit.
            max_a (float): Maximum linear acceleration limit.
            max_dw (float): Maximum angular acceleration limit.
            collision_weight (float): Weight on obstacle collision cost.
            smooth_weight (float): Weight on smoothness (squared acceleration/jerk) cost.
            limit_weight (float): Weight on dynamic limits violation cost.
            time_weight (float): Weight on segment durations.
            goal_pos_weight (float): Initial value of rho for Augmented Lagrangian.
        """
        self.M = M
        self.K = K
        self.robot_radius = robot_radius
        self.circle_dist = circle_dist
        self.obstacles = [] if obstacles is None else obstacles
        self.max_v = max_v
        self.max_w = max_w
        self.max_a = max_a
        self.max_dw = max_dw

        self.collision_weight = collision_weight
        self.smooth_weight = smooth_weight
        self.limit_weight = limit_weight
        self.time_weight = time_weight
        self.goal_pos_weight = goal_pos_weight

        # Precompute Simpson weights for 2K + 1 points
        self.simpson_w = np.ones(2 * K + 1)
        self.simpson_w[1:-1:2] = 4.0
        self.simpson_w[2:-1:2] = 2.0

        # Powell-Hestenes-Rockafellar Augmented Lagrangian Method (PHR ALM)
        self.alm_lambda = np.zeros(2)
        self.alm_rho = np.ones(2) * goal_pos_weight
        self.last_final_xy = np.zeros(2)

    @staticmethod
    def _boundary_to_coeff(x0, v0, a0, x1, v1, a1, T):
        """Analytical mapping from boundary conditions to quintic coefficients.

        Returns:
            coeffs (np.ndarray): Shape (6,) coefficients c0, c1, c2, c3, c4, c5.
            grads (dict): Gradients of coeffs with respect to inputs (x0, v0, a0, x1, v1, a1, T).
        """
        T = max(T, 1e-4)  # Prevent division by zero

        c0 = x0
        c1 = v0
        c2 = 0.5 * a0

        # Compute intermediate differences
        dx = x1 - x0 - v0 * T - 0.5 * a0 * T**2
        dv = v1 - v0 - a0 * T
        da = a1 - a0

        inv_T = 1.0 / T
        inv_T2 = inv_T**2
        inv_T3 = inv_T**3
        inv_T4 = inv_T**4
        inv_T5 = inv_T**5
        inv_T6 = inv_T**6

        # Analytical coefficients c3, c4, c5
        c3 = (10.0 * dx - 4.0 * T * dv + 0.5 * T**2 * da) * inv_T3
        c4 = (-15.0 * dx + 7.0 * T * dv - T**2 * da) * inv_T4
        c5 = (6.0 * dx - 3.0 * T * dv + 0.5 * T**2 * da) * inv_T5

        coeffs = np.array([c0, c1, c2, c3, c4, c5])

        # Compute partial derivatives of c3, c4, c5 wrt dx, dv, da
        dc3_ddx = 10.0 * inv_T3
        dc3_ddv = -4.0 * inv_T2
        dc3_dda = 0.5 * inv_T

        dc4_ddx = -15.0 * inv_T4
        dc4_ddv = 7.0 * inv_T3
        dc4_dda = -inv_T2

        dc5_ddx = 6.0 * inv_T5
        dc5_ddv = -3.0 * inv_T4
        dc5_dda = 0.5 * inv_T3

        # Compute partial derivatives of c3, c4, c5 wrt T (holding dx, dv, da constant)
        dc3_dT_const = -30.0 * inv_T4 * dx + 8.0 * inv_T3 * dv - 0.5 * inv_T2 * da
        dc4_dT_const = 60.0 * inv_T5 * dx - 21.0 * inv_T4 * dv + 2.0 * inv_T3 * da
        dc5_dT_const = -30.0 * inv_T6 * dx + 12.0 * inv_T5 * dv - 1.5 * inv_T4 * da

        # Derivatives of dx, dv, da wrt inputs
        ddx_dx0, ddx_dx1, ddx_dv0, ddx_da0 = -1.0, 1.0, -T, -0.5 * T**2
        ddv_dv0, ddv_dv1, ddv_da0 = -1.0, 1.0, -T
        dda_da0, dda_da1 = -1.0, 1.0

        ddx_dT = -v0 - a0 * T
        ddv_dT = -a0

        # Backpropagate to inputs
        # Jacobian wrt boundary inputs: shape (6, 7) for [x0, v0, a0, x1, v1, a1, T]
        jac = np.zeros((6, 7))

        # c0, c1, c2 depend directly on x0, v0, a0
        jac[0, 0] = 1.0  # dc0/dx0
        jac[1, 1] = 1.0  # dc1/dv0
        jac[2, 2] = 0.5  # dc2/da0

        # Helper function to accumulate derivatives for c3, c4, c5
        def set_row_grads(row_idx, dc_ddx, dc_ddv, dc_dda, dc_dT_const):
            # wrt x0, x1
            jac[row_idx, 0] = dc_ddx * ddx_dx0
            jac[row_idx, 3] = dc_ddx * ddx_dx1
            # wrt v0, v1
            jac[row_idx, 1] = dc_ddx * ddx_dv0 + dc_ddv * ddv_dv0
            jac[row_idx, 4] = dc_ddv * ddv_dv1
            # wrt a0, a1
            jac[row_idx, 2] = dc_ddx * ddx_da0 + dc_ddv * ddv_da0 + dc_dda * dda_da0
            jac[row_idx, 5] = dc_dda * dda_da1
            # wrt T
            jac[row_idx, 6] = dc_dT_const + dc_ddx * ddx_dT + dc_ddv * ddv_dT

        set_row_grads(3, dc3_ddx, dc3_ddv, dc3_dda, dc3_dT_const)
        set_row_grads(4, dc4_ddx, dc4_ddv, dc4_dda, dc4_dT_const)
        set_row_grads(5, dc5_ddx, dc5_ddv, dc5_dda, dc5_dT_const)

        return coeffs, jac

    def _evaluate_trajectory(self, boundary_vars, start_cartesian, start_state, goal_state):
        """Evaluate the trajectory cost and its analytical gradients.

        Args:
            boundary_vars (np.ndarray): Flat vector containing durations, step increments, and waypoint states.
            start_cartesian (np.ndarray): Shape (2,) initial position (x0, y0).
            start_state (np.ndarray): Shape (5,) initial state (theta0, v0, w0, a0, alpha0).
            goal_state (np.ndarray): Shape (7,) final target state (xf, yf, thetaf, vf, wf, af, alphaf).

        Returns:
            cost (float): Total cost.
            grad (np.ndarray): Gradient wrt boundary_vars.
        """
        # 1. Unpack variables
        M = self.M
        K = self.K

        T_vars = boundary_vars[:M]
        delta_s_vars = boundary_vars[M:2*M]
        wps_vars = boundary_vars[2*M:].reshape(M - 1, 5)

        # Reconstruct full boundary conditions
        theta = np.zeros(M + 1)
        w = np.zeros(M + 1)
        alpha_acc = np.zeros(M + 1)
        s = np.zeros(M + 1)
        v = np.zeros(M + 1)
        a = np.zeros(M + 1)

        # Start boundary (fixed)
        theta[0] = start_state[0]
        w[0] = start_state[2]  # w0
        alpha_acc[0] = start_state[4]  # alpha0
        s[0] = 0.0
        v[0] = start_state[1]  # v0
        a[0] = start_state[3]  # a0

        # Intermediate boundaries (optimized)
        theta[1:M] = wps_vars[:, 0]
        w[1:M] = wps_vars[:, 1]
        alpha_acc[1:M] = wps_vars[:, 2]
        v[1:M] = wps_vars[:, 3]
        a[1:M] = wps_vars[:, 4]

        # Arc-lengths cumulative sum
        s[1:] = np.cumsum(delta_s_vars)

        # End boundary (fixed)
        theta[M] = goal_state[2]
        w[M] = goal_state[4]
        alpha_acc[M] = goal_state[6]
        v[M] = goal_state[3]
        a[M] = goal_state[5]

        # 2. Map boundary variables to segment coefficients
        coeffs_theta = []
        coeffs_s = []
        jac_theta_list = []
        jac_s_list = []

        for i in range(M):
            # Seg i boundary values
            c_th, j_th = self._boundary_to_coeff(
                theta[i], w[i], alpha_acc[i],
                theta[i+1], w[i+1], alpha_acc[i+1], T_vars[i]
            )
            c_s, j_s = self._boundary_to_coeff(
                s[i], v[i], a[i],
                s[i+1], v[i+1], a[i+1], T_vars[i]
            )
            coeffs_theta.append(c_th)
            coeffs_s.append(c_s)
            jac_theta_list.append(j_th)
            jac_s_list.append(j_s)

        # Initialize costs and accumulator gradients
        cost = 0.0
        # Gradients wrt polynomial coefficients of all pieces
        g_coeff_theta = np.zeros((M, 6))
        g_coeff_s = np.zeros((M, 6))
        g_dT_direct = np.zeros(M)

        # 3. Trajectory evaluation & Numerical Integration
        # Cartesian integration uses Simpson's rule.
        pos = np.zeros((M, 2 * K + 1, 2))  # Positions at evaluated points
        current_xy = np.array(start_cartesian, dtype=float)

        vx = np.zeros((M, 2 * K + 1))
        vy = np.zeros((M, 2 * K + 1))
        theta_val = np.zeros((M, 2 * K + 1))
        v_val = np.zeros((M, 2 * K + 1))
        w_val = np.zeros((M, 2 * K + 1))
        a_val = np.zeros((M, 2 * K + 1))
        alpha_val = np.zeros((M, 2 * K + 1))

        simpson_steps_x = np.zeros((M, K))
        simpson_steps_y = np.zeros((M, K))

        for i in range(M):
            T = T_vars[i]
            h = T / K
            half_h = h / 2.0
            coeff = h / 6.0

            # Evaluate polynomials at 2K + 1 points
            for j in range(2 * K + 1):
                t = j * half_h
                # Basis vectors
                beta0 = np.array([1.0, t, t**2, t**3, t**4, t**5])
                beta1 = np.array([0.0, 1.0, 2.0 * t, 3.0 * t**2, 4.0 * t**3, 5.0 * t**4])
                beta2 = np.array([0.0, 0.0, 2.0, 6.0 * t, 12.0 * t**2, 20.0 * t**3])
                beta3 = np.array([0.0, 0.0, 0.0, 6.0, 24.0 * t, 60.0 * t**2])

                # Evaluate state and derivatives
                th = np.dot(coeffs_theta[i], beta0)
                wd = np.dot(coeffs_theta[i], beta1)
                al = np.dot(coeffs_theta[i], beta2)

                sc = np.dot(coeffs_s[i], beta0)
                vel = np.dot(coeffs_s[i], beta1)
                acc = np.dot(coeffs_s[i], beta2)

                theta_val[i, j] = th
                v_val[i, j] = vel
                w_val[i, j] = wd
                a_val[i, j] = acc
                alpha_val[i, j] = al

                # Cartesian velocities
                vx[i, j] = vel * np.cos(th)
                vy[i, j] = vel * np.sin(th)

                # Segment-wise Simpson interval accumulation
                # Odd points accumulate in their respective intervals
                if j > 0 and j % 2 == 1:
                    simpson_steps_x[i, j // 2] += 4.0 * coeff * vx[i, j]
                    simpson_steps_y[i, j // 2] += 4.0 * coeff * vy[i, j]
                # Even points accumulate in both adjacent intervals
                elif j % 2 == 0:
                    if j > 0:
                        simpson_steps_x[i, j // 2 - 1] += coeff * vx[i, j]
                        simpson_steps_y[i, j // 2 - 1] += coeff * vy[i, j]
                    if j < 2 * K:
                        simpson_steps_x[i, j // 2] += coeff * vx[i, j]
                        simpson_steps_y[i, j // 2] += coeff * vy[i, j]

            # Reconstruct positions at even (interval junction) points
            pos[i, 0] = current_xy
            for k in range(K):
                pos[i, 2 * k + 2] = pos[i, 2 * k] + np.array([simpson_steps_x[i, k], simpson_steps_y[i, k]])

            current_xy = pos[i, 2 * K]

        # 4. Evaluate cost and backpropagate derivatives
        g_pos = np.zeros((M, 2 * K + 1, 2))
        g_theta_eval = np.zeros((M, 2 * K + 1))

        # (A) Final position goal cost using Augmented Lagrangian Method (ALM) exactly matching TopAY C++
        goal_xy = goal_state[:2]
        final_xy = pos[-1, 2 * K]
        self.last_final_xy = np.copy(final_xy)
        goal_err = final_xy - goal_xy

        cost_endp = 0.5 * (self.alm_rho[0] * (goal_err[0] + self.alm_lambda[0] / self.alm_rho[0])**2 + 
                           self.alm_rho[1] * (goal_err[1] + self.alm_lambda[1] / self.alm_rho[1])**2)
        cost += cost_endp

        g_pos[-1, 2 * K, 0] += self.alm_rho[0] * (goal_err[0] + self.alm_lambda[0] / self.alm_rho[0])
        g_pos[-1, 2 * K, 1] += self.alm_rho[1] * (goal_err[1] + self.alm_lambda[1] / self.alm_rho[1])

        # (B) Collision Cost using Two Overlapping Circles approximation for rectangular robot
        # Circles are aligned along robot's longitudinal heading axis.
        d = self.circle_dist
        R_c = self.robot_radius

        for i in range(M):
            # Checked only at even evaluated points (boundaries of K intervals)
            for k in range(K + 1):
                j = 2 * k
                pt = pos[i, j]
                th = theta_val[i, j]
                cos_th = np.cos(th)
                sin_th = np.sin(th)

                # Center of front and rear circles
                p_front = pt + d * np.array([cos_th, sin_th])
                p_rear = pt - d * np.array([cos_th, sin_th])

                # Check both circles against each obstacle
                for ox, oy, r in self.obstacles:
                    # 1. Front Circle
                    d_front = np.sqrt((p_front[0] - ox)**2 + (p_front[1] - oy)**2)
                    viol_front = R_c + r - d_front
                    if viol_front > 0.0:
                        coll_cost = 0.5 * self.collision_weight * viol_front**2
                        cost += coll_cost
                        if d_front > 1e-4:
                            grad_dist = (p_front - np.array([ox, oy])) / d_front
                            g_front = -self.collision_weight * viol_front * grad_dist
                            # Propagate to robot center position (x, y)
                            g_pos[i, j] += g_front
                            # Propagate to robot heading (theta) via rotation Jacobian
                            dp_dtheta = d * np.array([-sin_th, cos_th])
                            g_theta_eval[i, j] += np.dot(g_front, dp_dtheta)

                    # 2. Rear Circle
                    d_rear = np.sqrt((p_rear[0] - ox)**2 + (p_rear[1] - oy)**2)
                    viol_rear = R_c + r - d_rear
                    if viol_rear > 0.0:
                        coll_cost = 0.5 * self.collision_weight * viol_rear**2
                        cost += coll_cost
                        if d_rear > 1e-4:
                            grad_dist = (p_rear - np.array([ox, oy])) / d_rear
                            g_rear = -self.collision_weight * viol_rear * grad_dist
                            # Propagate to robot center position (x, y)
                            g_pos[i, j] += g_rear
                            # Propagate to robot heading (theta) via rotation Jacobian
                            dp_dtheta = -d * np.array([-sin_th, cos_th])
                            g_theta_eval[i, j] += np.dot(g_rear, dp_dtheta)

        # (C) Dynamic Limit Costs
        # Penalize exceeding: max_v, max_w, max_a, max_dw
        for i in range(M):
            T = T_vars[i]
            h = T / K
            coeff = h / 6.0

            for j in range(2 * K + 1):
                omg_w = 0.5 if (j == 0 or j == 2 * K) else 1.0
                weight_multiplier = omg_w * (T / self.K)  # Weight scale as per TopAY step integration

                # Linear velocity v (allowing symmetric forward and backward speed limits [-max_v, max_v])
                v_j = v_val[i, j]
                v_viol_pos = v_j - self.max_v
                v_viol_neg = -v_j - self.max_v
                if v_viol_pos > 0.0:
                    cost += 0.5 * self.limit_weight * v_viol_pos**2 * weight_multiplier
                    dv_coeff = self.limit_weight * v_viol_pos * weight_multiplier
                    g_coeff_s[i] += dv_coeff * np.array([0.0, 1.0, 2.0 * (j * h / 2.0), 3.0 * (j * h / 2.0)**2, 4.0 * (j * h / 2.0)**3, 5.0 * (j * h / 2.0)**4])
                    g_dT_direct[i] += 0.5 * self.limit_weight * v_viol_pos**2 * (omg_w / self.K)
                if v_viol_neg > 0.0:
                    cost += 0.5 * self.limit_weight * v_viol_neg**2 * weight_multiplier
                    dv_coeff = -self.limit_weight * v_viol_neg * weight_multiplier
                    g_coeff_s[i] += dv_coeff * np.array([0.0, 1.0, 2.0 * (j * h / 2.0), 3.0 * (j * h / 2.0)**2, 4.0 * (j * h / 2.0)**3, 5.0 * (j * h / 2.0)**4])
                    g_dT_direct[i] += 0.5 * self.limit_weight * v_viol_neg**2 * (omg_w / self.K)

                # Angular velocity w
                w_j = w_val[i, j]
                w_viol = np.abs(w_j) - self.max_w
                if w_viol > 0.0:
                    cost += 0.5 * self.limit_weight * w_viol**2 * weight_multiplier
                    dw_coeff = self.limit_weight * w_viol * np.sign(w_j) * weight_multiplier
                    g_coeff_theta[i] += dw_coeff * np.array([0.0, 1.0, 2.0 * (j * h / 2.0), 3.0 * (j * h / 2.0)**2, 4.0 * (j * h / 2.0)**3, 5.0 * (j * h / 2.0)**4])
                    g_dT_direct[i] += 0.5 * self.limit_weight * w_viol**2 * (omg_w / self.K)

                # Linear acceleration a
                a_j = a_val[i, j]
                a_viol = np.abs(a_j) - self.max_a
                if a_viol > 0.0:
                    cost += 0.5 * self.limit_weight * a_viol**2 * weight_multiplier
                    da_coeff = self.limit_weight * a_viol * np.sign(a_j) * weight_multiplier
                    g_coeff_s[i] += da_coeff * np.array([0.0, 0.0, 2.0, 6.0 * (j * h / 2.0), 12.0 * (j * h / 2.0)**2, 20.0 * (j * h / 2.0)**3])
                    g_dT_direct[i] += 0.5 * self.limit_weight * a_viol**2 * (omg_w / self.K)

                # Angular acceleration alpha
                alpha_j = alpha_val[i, j]
                alpha_viol = np.abs(alpha_j) - self.max_dw
                if alpha_viol > 0.0:
                    cost += 0.5 * self.limit_weight * alpha_viol**2 * weight_multiplier
                    dalpha_coeff = self.limit_weight * alpha_viol * np.sign(alpha_j) * weight_multiplier
                    g_coeff_theta[i] += dalpha_coeff * np.array([0.0, 0.0, 2.0, 6.0 * (j * h / 2.0), 12.0 * (j * h / 2.0)**2, 20.0 * (j * h / 2.0)**3])
                    g_dT_direct[i] += 0.5 * self.limit_weight * alpha_viol**2 * (omg_w / self.K)

        # (D) Smoothness Cost (Integrated Jerk Squared for both yaw and arc-length)
        for i in range(M):
            T = T_vars[i]
            h = T / K
            half_h = h / 2.0
            coeff = h / 6.0

            for j in range(2 * K + 1):
                omg = self.simpson_w[j]
                t = j * half_h
                beta3 = np.array([0.0, 0.0, 0.0, 6.0, 24.0 * t, 60.0 * t**2])
                beta4 = np.array([0.0, 0.0, 0.0, 0.0, 24.0, 120.0 * t])

                # Yaw jerk
                jk_th = np.dot(coeffs_theta[i], beta3)
                th_jerk_cost = self.smooth_weight * jk_th**2 * omg * coeff
                cost += th_jerk_cost

                # Arc jerk
                jk_s = np.dot(coeffs_s[i], beta3)
                s_jerk_cost = self.smooth_weight * jk_s**2 * omg * coeff
                cost += s_jerk_cost

                # Gradients wrt coefficients
                g_coeff_theta[i] += 2.0 * self.smooth_weight * jk_th * beta3 * omg * coeff
                g_coeff_s[i] += 2.0 * self.smooth_weight * jk_s * beta3 * omg * coeff

                # Gradients wrt T_vars[i] due to Simpson scaling and point shifting
                alpha_t = j / (2.0 * K)
                djk_th_dt = np.dot(coeffs_theta[i], beta4)
                djk_s_dt = np.dot(coeffs_s[i], beta4)

                g_dT_direct[i] += (th_jerk_cost + s_jerk_cost) / T
                g_dT_direct[i] += 2.0 * self.smooth_weight * jk_th * djk_th_dt * alpha_t * omg * coeff
                g_dT_direct[i] += 2.0 * self.smooth_weight * jk_s * djk_s_dt * alpha_t * omg * coeff

        # (E) Time duration cost
        cost += self.time_weight * np.sum(T_vars)
        g_dT_direct += self.time_weight

        # 5. Backpropagate Position Gradients (g_pos) backwards through cumulative steps
        G_pos = np.zeros((M, 2 * K + 1, 2))
        suffix_sum = np.zeros(2)
        for i in range(M - 1, -1, -1):
            for k in range(K, -1, -1):
                j = 2 * k
                suffix_sum += g_pos[i, j]
                G_pos[i, j] = np.copy(suffix_sum)

        # 6. Accumulate gradients of cost wrt velocities and headings (f^x, f^y)
        g_vx = np.zeros((M, 2 * K + 1))
        g_vy = np.zeros((M, 2 * K + 1))

        for i in range(M):
            T = T_vars[i]
            coeff = (T / K) / 6.0

            for k in range(K):
                G_step = G_pos[i, 2 * k + 2]
                # End point 2k+2
                g_vx[i, 2 * k + 2] += coeff * G_step[0]
                g_vy[i, 2 * k + 2] += coeff * G_step[1]
                # Start point 2k
                g_vx[i, 2 * k] += coeff * G_step[0]
                g_vy[i, 2 * k] += coeff * G_step[1]
                # Mid point 2k+1
                g_vx[i, 2 * k + 1] += 4.0 * coeff * G_step[0]
                g_vy[i, 2 * k + 1] += 4.0 * coeff * G_step[1]

            # Derivative of coeff wrt T_vars[i] to g_dT_direct:
            for k in range(K):
                G_step = G_pos[i, 2 * k + 2]
                vx_step = vx[i, 2 * k] + 4.0 * vx[i, 2 * k + 1] + vx[i, 2 * k + 2]
                vy_step = vy[i, 2 * k] + 4.0 * vy[i, 2 * k + 1] + vy[i, 2 * k + 2]
                g_dT_direct[i] += (G_step[0] * vx_step + G_step[1] * vy_step) * (1.0 / (6.0 * K))

        # 7. Propagate gradients from (vx, vy) to (theta, v)
        for i in range(M):
            g_theta_eval[i] += g_vx[i] * (-v_val[i] * np.sin(theta_val[i])) + g_vy[i] * (v_val[i] * np.cos(theta_val[i]))
            g_v_eval_segment = g_vx[i] * np.cos(theta_val[i]) + g_vy[i] * np.sin(theta_val[i])

            # Propagate to coefficients
            T = T_vars[i]
            half_h = (T / self.K) / 2.0

            for j in range(2 * K + 1):
                t = j * half_h
                beta0 = np.array([1.0, t, t**2, t**3, t**4, t**5])
                beta1 = np.array([0.0, 1.0, 2.0 * t, 3.0 * t**2, 4.0 * t**3, 5.0 * t**4])
                beta2 = np.array([0.0, 0.0, 2.0, 6.0 * t, 12.0 * t**2, 20.0 * t**3])

                g_coeff_theta[i] += g_theta_eval[i, j] * beta0
                g_coeff_s[i] += g_v_eval_segment[j] * beta1

                # Gradients wrt T due to shift of point t_j = j * T / (2K)
                alpha_t = j / (2.0 * K)
                dtheta_dt = np.dot(coeffs_theta[i], beta1)
                dv_dt = np.dot(coeffs_s[i], beta2)

                g_dT_direct[i] += g_theta_eval[i, j] * dtheta_dt * alpha_t
                g_dT_direct[i] += g_v_eval_segment[j] * dv_dt * alpha_t

        # 9. Propagate gradients from coefficients back to boundary variables
        grad_T = np.copy(g_dT_direct)
        grad_delta_s = np.zeros(M)
        grad_theta = np.zeros(M + 1)
        grad_w = np.zeros(M + 1)
        grad_alpha = np.zeros(M + 1)
        grad_s = np.zeros(M + 1)
        grad_v = np.zeros(M + 1)
        grad_a = np.zeros(M + 1)

        for i in range(M):
            # Backpropagate theta segment
            g_c_th = g_coeff_theta[i]
            j_th = jac_theta_list[i]

            grad_theta[i] += np.dot(g_c_th, j_th[:, 0])
            grad_w[i] += np.dot(g_c_th, j_th[:, 1])
            grad_alpha[i] += np.dot(g_c_th, j_th[:, 2])

            grad_theta[i+1] += np.dot(g_c_th, j_th[:, 3])
            grad_w[i+1] += np.dot(g_c_th, j_th[:, 4])
            grad_alpha[i+1] += np.dot(g_c_th, j_th[:, 5])

            grad_T[i] += np.dot(g_c_th, j_th[:, 6])

            # Backpropagate s segment
            g_c_s = g_coeff_s[i]
            j_s = jac_s_list[i]

            grad_s[i] += np.dot(g_c_s, j_s[:, 0])
            grad_v[i] += np.dot(g_c_s, j_s[:, 1])
            grad_a[i] += np.dot(g_c_s, j_s[:, 2])

            grad_s[i+1] += np.dot(g_c_s, j_s[:, 3])
            grad_v[i+1] += np.dot(g_c_s, j_s[:, 4])
            grad_a[i+1] += np.dot(g_c_s, j_s[:, 5])

            grad_T[i] += np.dot(g_c_s, j_s[:, 6])

        # Map cumulative s gradients back to delta_s
        suffix_s = 0.0
        for i in range(M, 0, -1):
            suffix_s += grad_s[i]
            grad_delta_s[i-1] = suffix_s

        # 10. Assemble complete gradient vector
        grad_vars = np.zeros_like(boundary_vars)
        grad_vars[:M] = grad_T
        grad_vars[M:2*M] = grad_delta_s

        wps_grad = grad_vars[2*M:].reshape(M - 1, 5)
        wps_grad[:, 0] = grad_theta[1:M]
        wps_grad[:, 1] = grad_w[1:M]
        wps_grad[:, 2] = grad_alpha[1:M]
        wps_grad[:, 3] = grad_v[1:M]
        wps_grad[:, 4] = grad_a[1:M]

        return cost, grad_vars

    def optimize(self, start_pose, goal_pose, initial_guess=None):
        """Run trajectory optimization using Augmented Lagrangian Method (ALM) outer loop
        and Scipy L-BFGS-B inner loop.

        Args:
            start_pose (np.ndarray): Start pose (x, y, theta, v, w, a, alpha).
            goal_pose (np.ndarray): Goal pose (x, y, theta, v, w, a, alpha).
            initial_guess (np.ndarray): Optional initial guess.

        Returns:
            success (bool): True if ALM converged successfully.
            trajectory_data (dict): Dict of optimized trajectory profiles.
        """
        M = self.M
        total_vars = 2 * M + 5 * (M - 1)

        if initial_guess is None:
            # Setup naive straight-line initial guess if not provided
            T_guess = np.ones(M) * 1.5
            dist = np.linalg.norm(goal_pose[:2] - start_pose[:2])
            delta_s_guess = np.ones(M) * (dist / M)

            wps_guess = np.zeros((M - 1, 5))
            for i in range(M - 1):
                fraction = (i + 1) / M
                wps_guess[i, 0] = start_pose[2] + fraction * (goal_pose[2] - start_pose[2])
                wps_guess[i, 3] = 0.5 * (start_pose[3] + goal_pose[3])

            initial_guess = np.zeros(total_vars)
            initial_guess[:M] = T_guess
            initial_guess[M:2*M] = delta_s_guess
            initial_guess[2*M:] = wps_guess.flatten()

        bounds = []
        for _ in range(M):
            bounds.append((0.05, 10.0))  # T
        for _ in range(M):
            bounds.append((-10.0, 10.0))  # delta_s (allow negative to support reversal driving)
        for _ in range(M - 1):
            bounds.extend([
                (-2.0 * np.pi, 2.0 * np.pi),  # theta
                (-self.max_w, self.max_w),    # w
                (-self.max_dw, self.max_dw),  # alpha
                (0.0, self.max_v),            # v
                (-self.max_a, self.max_a)     # a
            ])

        # Initialize ALM Multipliers & Penalties exactly matching TopAY paper/code
        self.alm_lambda = np.zeros(2)
        self.alm_rho = np.ones(2) * self.goal_pos_weight  # initial rho_x, rho_y

        success = False
        res = None
        opt_vars = np.copy(initial_guess)

        # Outer ALM loop (typically 5 iterations are enough to satisfy goal to millimeter precision)
        for alm_iter in range(5):
            iteration_count = [0]

            def objective(x):
                cost, grad = self._evaluate_trajectory(x, start_pose[:2], start_pose[2:], goal_pose)
                iteration_count[0] += 1
                print(f"    [L-BFGS Iter {iteration_count[0]}] Cost = {cost:.4f}, Grad Norm = {np.linalg.norm(grad):.4f}")
                return cost, grad

            res = minimize(
                objective,
                opt_vars,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-3}
            )

            opt_vars = res.x

            # Check constraint violation
            final_xy = np.copy(self.last_final_xy)
            goal_err = final_xy - goal_pose[:2]
            err_norm = np.linalg.norm(goal_err)

            print(f"  [ALM Outer Loop {alm_iter + 1}] Position Error: {err_norm:.4f} m, Lambda: {self.alm_lambda}, Rho: {self.alm_rho}")

            # TopAY convergence criterion: tolerance is typically 0.05m
            if err_norm < 0.05:
                success = True
                break

            # Update Multipliers & Penalties (Powell-Hestenes-Rockafellar update)
            self.alm_lambda += self.alm_rho * goal_err
            # Rho scaling gamma is typically 1.5
            self.alm_rho = np.minimum(1.5 * self.alm_rho, 100000.0)

        # Reconstruct final trajectory profiles
        T_opt = opt_vars[:M]
        delta_s_opt = opt_vars[M:2*M]
        wps_opt = opt_vars[2*M:].reshape(M - 1, 5)

        theta = np.zeros(M + 1)
        w = np.zeros(M + 1)
        alpha_acc = np.zeros(M + 1)
        s = np.zeros(M + 1)
        v = np.zeros(M + 1)
        a = np.zeros(M + 1)

        theta[0] = start_pose[2]
        w[0] = start_pose[4]
        alpha_acc[0] = start_pose[6]
        s[0] = 0.0
        v[0] = start_pose[3]
        a[0] = start_pose[5]

        theta[1:M] = wps_opt[:, 0]
        w[1:M] = wps_opt[:, 1]
        alpha_acc[1:M] = wps_opt[:, 2]
        v[1:M] = wps_opt[:, 3]
        a[1:M] = wps_opt[:, 4]

        s[1:] = np.cumsum(delta_s_opt)

        theta[M] = goal_pose[2]
        w[M] = goal_pose[4]
        alpha_acc[M] = goal_pose[6]
        v[M] = goal_pose[3]
        a[M] = goal_pose[5]

        # Generate dense trajectory
        dense_times = []
        dense_x = []
        dense_y = []
        dense_theta = []
        dense_v = []
        dense_w = []
        dense_a = []
        dense_alpha = []

        current_xy = np.array(start_pose[:2])
        current_time = 0.0

        for i in range(M):
            c_th, _ = self._boundary_to_coeff(
                theta[i], w[i], alpha_acc[i],
                theta[i+1], w[i+1], alpha_acc[i+1], T_opt[i]
            )
            c_s, _ = self._boundary_to_coeff(
                s[i], v[i], a[i],
                s[i+1], v[i+1], a[i+1], T_opt[i]
            )

            eval_steps = 50
            ts = np.linspace(0.0, T_opt[i], eval_steps)

            h = T_opt[i] / eval_steps
            for idx, t in enumerate(ts):
                beta0 = np.array([1.0, t, t**2, t**3, t**4, t**5])
                beta1 = np.array([0.0, 1.0, 2.0 * t, 3.0 * t**2, 4.0 * t**3, 5.0 * t**4])
                beta2 = np.array([0.0, 0.0, 2.0, 6.0 * t, 12.0 * t**2, 20.0 * t**3])

                th = np.dot(c_th, beta0)
                wd = np.dot(c_th, beta1)
                al = np.dot(c_th, beta2)

                vel = np.dot(c_s, beta1)
                acc = np.dot(c_s, beta2)

                if idx > 0:
                    prev_t = ts[idx-1]
                    prev_beta0 = np.array([1.0, prev_t, prev_t**2, prev_t**3, prev_t**4, prev_t**5])
                    prev_beta1 = np.array([0.0, 1.0, 2.0 * prev_t, 3.0 * prev_t**2, 4.0 * prev_t**3, 5.0 * prev_t**4])
                    prev_th = np.dot(c_th, prev_beta0)
                    prev_vel = np.dot(c_s, prev_beta1)

                    dx_step = 0.5 * h * (prev_vel * np.cos(prev_th) + vel * np.cos(th))
                    dy_step = 0.5 * h * (prev_vel * np.sin(prev_th) + vel * np.sin(th))
                    current_xy += np.array([dx_step, dy_step])

                dense_times.append(current_time + t)
                dense_x.append(current_xy[0])
                dense_y.append(current_xy[1])
                dense_theta.append(th)
                dense_v.append(vel)
                dense_w.append(wd)
                dense_a.append(acc)
                dense_alpha.append(al)

            current_time += T_opt[i]

        trajectory_data = {
            'time': np.array(dense_times),
            'x': np.array(dense_x),
            'y': np.array(dense_y),
            'theta': np.array(dense_theta),
            'v': np.array(dense_v),
            'w': np.array(dense_w),
            'a': np.array(dense_a),
            'alpha': np.array(dense_alpha),
            'cost': res.fun,
            'message': res.message,
            'final_xy': self.last_final_xy
        }

        # success is defined by formal ALM convergence
        return success, trajectory_data
