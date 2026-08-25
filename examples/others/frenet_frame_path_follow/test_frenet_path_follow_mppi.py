import json
import math
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

from robotics_algorithm.env.continuous_2d.diff_drive_2d_control import DiffDrive2DControl
from robotics_algorithm.utils import math_utils
from robotics_algorithm.control.optimal_control.mppi import MPPI


class FrenetMPPI:
    """This class implements an MPPI controller that uses Frenet frame and Stanley-style critics for trajectory scoring.
    """

    def __init__(
        self,
        env: DiffDrive2DControl,
        ref_path: np.ndarray,
        num_traj_samples: int = 150,
        sample_traj_len: int = 10,  # 10 steps of action_dt (0.05s) = 0.5s horizon
        param_lambda: float = 0.5,
        filter_window_size: int = 3,
        action_mean: float | np.ndarray = 0.0,
        action_std: float | np.ndarray = 1.0,
        lateral_cost_weight: float = 15.0,
        heading_cost_max_weight: float = 2.0,
        heading_cost_min_weight: float = 0.1,
        heading_cost_lateral_sigma: float = 0.5,
        longitudinal_progress_weight: float = 1.0,
        w_theta: float = 0.2,
        spatial_resolution: float = 0.02,
        rotation_shim_threshold: float = 0.1,
        lookahead_dist: float = 0.5,
    ) -> None:
        assert isinstance(env, DiffDrive2DControl), 'env must be a DiffDrive2DControl'

        self.env = env
        self.ref_path = np.asarray(ref_path)
        self.num_samples = num_traj_samples
        self.sample_traj_len = sample_traj_len
        self.param_lambda = param_lambda
        self.filter_window_size = filter_window_size
        self.uniform_control_bias = 0.1

        # DWA / MPPI Action spaces
        self.action_shape = np.array(env.random_action()).shape
        self.action_mean = np.array([0.25, 0.0]) if isinstance(action_mean, float) and action_mean == 0.0 else np.array(action_mean)
        self.action_std = np.array([0.2, math.radians(45)]) if isinstance(action_std, float) and action_std == 1.0 else np.array(action_std)

        self.prev_actions = np.tile(self.action_mean, (self.sample_traj_len, 1))
        self.nominal_action = np.tile(self.action_mean, (self.sample_traj_len, 1))

        self._min_ang_vel = env.action_space.space[0][1]
        self._max_ang_vel = env.action_space.space[1][1]
        self._sim_steps = sample_traj_len

        # Scoring weights
        self.lateral_cost_weight = lateral_cost_weight
        self.heading_cost_min_weight = heading_cost_min_weight
        self.heading_cost_max_weight = heading_cost_max_weight
        self.heading_cost_lateral_sigma = heading_cost_lateral_sigma
        self.longitudinal_progress_weight = longitudinal_progress_weight
        self.w_theta = w_theta
        self.spatial_resolution = spatial_resolution
        self.rotation_shim_threshold = rotation_shim_threshold
        self.lookahead_dist = lookahead_dist

        # Keep reference progress entirely inside controller.
        self.cur_pos_idx = 0
        self._lookahead_cursor = 0
        self._ref_search_window = 200

        # Precompute path progression directions to detect cusps robustly
        n = len(self.ref_path)
        self.progression_directions = np.ones(n - 1)
        for i in range(n - 1):
            dp = self.ref_path[i + 1, :2] - self.ref_path[i, :2]
            if np.linalg.norm(dp) > 1e-4:
                heading = np.array([np.cos(self.ref_path[i, 2]), np.sin(self.ref_path[i, 2])])
                self.progression_directions[i] = np.sign(np.dot(dp, heading))
            else:
                self.progression_directions[i] = 0.0

    def _resample_states_by_spatial_resolution(self, states: np.ndarray, resolution: float) -> np.ndarray:
        if states.shape[0] == 0:
            return states
        if states.shape[0] == 1:
            return states

        se2_path_len = math_utils.calc_se2_path_length(states, self.w_theta)
        if se2_path_len <= 1e-9:
            return states[:1]

        target_arc = np.arange(0, se2_path_len + 1e-6, resolution)
        if len(target_arc) == 0:
            return states[:1]

        cumulative_dist = math_utils.calc_se2_cumulative_distances(states, self.w_theta)
        x_rs = np.interp(target_arc, cumulative_dist, states[:, 0])
        y_rs = np.interp(target_arc, cumulative_dist, states[:, 1])
        yaw_rs = np.interp(target_arc, cumulative_dist, np.unwrap(states[:, 2]))
        return np.column_stack((x_rs, y_rs, yaw_rs))

    def _build_reference_segment(self, start_idx: int, lookahead_dist: float) -> np.ndarray:
        n = len(self.ref_path)
        start_idx = max(0, min(start_idx, n - 1))
        if start_idx >= n - 1:
            self._lookahead_cursor = start_idx
            return self.ref_path[start_idx : start_idx + 1]

        next_cusp = self._get_next_cusp_idx(start_idx)

        segment_indices = [start_idx]
        accumulated_dist = 0.0

        for i in range(start_idx, n - 1):
            if i >= next_cusp:
                break

            p1 = self.ref_path[i]
            p2 = self.ref_path[i + 1]
            dist = math_utils.se2_distance(p1, p2, w_theta=self.w_theta)

            accumulated_dist += dist
            segment_indices.append(i + 1)

            if accumulated_dist >= lookahead_dist:
                break

        self._lookahead_cursor = segment_indices[-1]
        return self.ref_path[segment_indices]

    def _get_next_cusp_idx(self, start_idx: int) -> int:
        n = len(self.ref_path)
        if start_idx >= n - 2:
            return n - 1

        current_dir = self.progression_directions[start_idx]
        for i in range(start_idx + 1, n - 1):
            if self.progression_directions[i] != current_dir:
                return i
        return n - 1

    def _get_reference_start_idx(self, state: np.ndarray) -> int:
        start_idx = self.cur_pos_idx
        next_cusp = self._get_next_cusp_idx(start_idx)
        end_idx = min(next_cusp + 1, len(self.ref_path))

        if end_idx <= start_idx:
            return len(self.ref_path) - 1

        dx = self.ref_path[start_idx:end_idx, 0] - state[0]
        dy = self.ref_path[start_idx:end_idx, 1] - state[1]
        dyaw = self.ref_path[start_idx:end_idx, 2] - state[2]
        dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi

        dists = np.sqrt(dx**2 + dy**2 + (self.w_theta * dyaw)**2)
        nearest_idx = start_idx + int(np.argmin(dists))

        if nearest_idx >= next_cusp and next_cusp < len(self.ref_path) - 1:
            cusp_state = self.ref_path[next_cusp]
            prev_state = self.ref_path[max(0, next_cusp - 1)]

            u = cusp_state[:2] - prev_state[:2]
            v = cusp_state[:2] - state[:2]

            if np.linalg.norm(u) > 1e-4:
                has_crossed = np.dot(v, u) <= 0
                dist_to_cusp = np.linalg.norm(v)
                if not has_crossed and dist_to_cusp > 0.05:
                    nearest_idx = max(start_idx, next_cusp - 1)
            else:
                heading_err_cusp = math_utils.normalize_angle(cusp_state[2] - state[2])
                if abs(heading_err_cusp) > self.rotation_shim_threshold:
                    nearest_idx = max(start_idx, next_cusp - 1)

        self.cur_pos_idx = nearest_idx
        return nearest_idx

    def _lateral_heading_cost(self, sampled_traj: np.ndarray, ref_traj: np.ndarray, is_reverse: bool = False) -> float:
        min_len = min(len(ref_traj), len(sampled_traj))
        if min_len < 1:
            return 0.0

        total_cost = 0.0
        k_lateral = 5.0

        for k in range(min_len):
            x, y, yaw = sampled_traj[k]
            ref_x, ref_y, ref_yaw = ref_traj[k]
            normal = np.array([-np.sin(ref_yaw), np.cos(ref_yaw)])
            pos_err = np.array([x - ref_x, y - ref_y])

            lateral_err_signed = np.dot(pos_err, normal).item()
            lateral_err = abs(lateral_err_signed)

            if is_reverse:
                crosstrack_heading = np.arctan(k_lateral * lateral_err_signed)
            else:
                crosstrack_heading = np.arctan(-k_lateral * lateral_err_signed)
            yaw_desired = ref_yaw + crosstrack_heading

            heading_err = abs(math_utils.normalize_angle(yaw - yaw_desired))

            total_cost += self.lateral_cost_weight * lateral_err + self.heading_cost_max_weight * heading_err

        return total_cost / min_len

    def _longitudinal_cost(self, sampled_traj: np.ndarray, ref_traj: np.ndarray, goal_threshold: float = 0.3) -> float:
        traj_end = sampled_traj[-1]

        dists = np.array([math_utils.se2_distance(pt, traj_end, w_theta=self.w_theta) for pt in ref_traj])
        if dists.size == 0:
            closest_idx = len(ref_traj) - 1
        else:
            closest_idx = int(np.argmin(dists))

        ref_start = ref_traj[0]
        ref_goal = ref_traj[-1]
        closest_ref = ref_traj[closest_idx]

        dist_to_goal = math_utils.se2_distance(traj_end, ref_goal, w_theta=self.w_theta)
        progress_dist = math_utils.se2_distance(closest_ref, ref_start, w_theta=self.w_theta)

        heading_err = math_utils.normalize_angle(traj_end[2] - closest_ref[2])
        alignment_factor = np.cos(heading_err)
        if alignment_factor > 0:
            progress_reward = -self.longitudinal_progress_weight * progress_dist * alignment_factor
        else:
            progress_reward = 0.0

        progress_fade = np.clip(dist_to_goal / 1.0, 0.0, 1.0)
        goal_fade = 1.0 - progress_fade

        cost = progress_reward * progress_fade
        goal_attraction_weight = 2.0
        cost += goal_attraction_weight * dist_to_goal * goal_fade

        return cost

    def run(self, state: np.ndarray) -> np.ndarray:
        # 1. Exact Stop Condition
        goal_state = self.ref_path[-1]
        dist_to_goal = math_utils.se2_distance(state, goal_state, w_theta=self.w_theta)
        if dist_to_goal < 0.05:
            self.best_traj = [state]
            return np.array([0.0, 0.0])

        # Record old progression direction to detect cusp crossings
        old_dir = self.progression_directions[min(self.cur_pos_idx, len(self.progression_directions) - 1)]

        # 2. Get current reference start index and build reference segment
        ref_start_idx = self._get_reference_start_idx(state)
        ref_segment = self._build_reference_segment(ref_start_idx, self.lookahead_dist)

        new_dir = self.progression_directions[min(ref_start_idx, len(self.progression_directions) - 1)]

        # Dynamically align nominal action with the current path segment direction
        current_mean = self.action_mean.copy()
        if new_dir == -1.0:
            current_mean[0] = -abs(current_mean[0])
        else:
            current_mean[0] = abs(current_mean[0])
        self.nominal_action = np.tile(current_mean, (self.sample_traj_len, 1))

        # If we just crossed a cusp (direction of progression changed), clear/reset prev_actions
        if new_dir != old_dir:
            print(f"Cusp crossed! Clearing prev_actions from direction {old_dir} to {new_dir}")
            self.prev_actions = np.tile(current_mean, (self.sample_traj_len, 1))

        # 3. Unified Rotation Shim Triggering
        current_ref_state = self.ref_path[ref_start_idx]
        current_heading_err = math_utils.normalize_angle(current_ref_state[2] - state[2])

        if len(ref_segment) > 1:
            translation_dist = np.linalg.norm(ref_segment[-1, :2] - ref_segment[0, :2])
        else:
            translation_dist = 0.0

        lookahead_state = self.ref_path[self._lookahead_cursor]
        lookahead_heading_err = math_utils.normalize_angle(lookahead_state[2] - state[2])

        trigger_shim = False
        target_heading_err = 0.0

        if abs(current_heading_err) > self.rotation_shim_threshold:
            trigger_shim = True
            target_heading_err = current_heading_err
        elif translation_dist < 0.05:
            if abs(lookahead_heading_err) > self.rotation_shim_threshold:
                trigger_shim = True
                target_heading_err = lookahead_heading_err
            else:
                print(f"In-place rotation complete! Directly jumping cursor from {self.cur_pos_idx} to end of rotation segment {self._lookahead_cursor}")
                self.cur_pos_idx = self._lookahead_cursor
                ref_start_idx = self._lookahead_cursor
                ref_segment = self._build_reference_segment(ref_start_idx, self.lookahead_dist)

                # Update direction and nominal bias after jump
                new_dir = self.progression_directions[min(ref_start_idx, len(self.progression_directions) - 1)]
                current_mean = self.action_mean.copy()
                if new_dir == -1.0:
                    current_mean[0] = -abs(current_mean[0])
                else:
                    current_mean[0] = abs(current_mean[0])
                self.nominal_action = np.tile(current_mean, (self.sample_traj_len, 1))
                print(f"Jumping cursor completed! Clearing prev_actions to new direction {new_dir}")
                self.prev_actions = np.tile(current_mean, (self.sample_traj_len, 1))

        # Execute Rotation Shim if triggered
        if trigger_shim:
            print("Rotation Shim Mode!!!!!")

            kp_omega = 2.0
            omega = np.clip(kp_omega * target_heading_err, self._min_ang_vel, self._max_ang_vel)
            traj = [state]
            dt = self.env.action_dt
            cur_yaw = state[2]
            for _ in range(self._sim_steps):
                cur_yaw += omega * dt
                traj.append(np.array([state[0], state[1], cur_yaw]))
            self.best_traj = np.array(traj).tolist()
            return np.array([0.0, omega])

        # 4. Standard MPPI Rollout and Scoring
        all_costs = []
        all_noises = []
        all_trajs = []

        # We will reuse the pre-allocated action buffer
        for k in range(self.num_samples):
            cur_state = state
            total_cost = 0.0
            noises = np.random.randn(self.sample_traj_len, *self.action_shape) * self.action_std

            sampled_traj = [cur_state]
            for t in range(self.sample_traj_len):
                noise = noises[t]
                if k < self.uniform_control_bias * self.num_samples:
                    sampled_action = self.nominal_action[t] + noise
                else:
                    sampled_action = self.prev_actions[t] + noise

                sampled_action = np.clip(sampled_action, self.env.action_space.space[0], self.env.action_space.space[1])

                # Simulate state transition
                new_state, _, _, _, _ = self.env.sample_state_transition(cur_state, sampled_action)
                cur_state = new_state
                sampled_traj.append(cur_state)

            sampled_traj = np.array(sampled_traj)

            # Resample both reference and trajectory using the fixed spatial resolution
            ref_traj_rs = self._resample_states_by_spatial_resolution(ref_segment, self.spatial_resolution)
            sampled_traj_rs = self._resample_states_by_spatial_resolution(sampled_traj, self.spatial_resolution)

            # Evaluate cost using Frenet/Stanley/Continuous Longitudinal critics
            total_cost = self._lateral_heading_cost(sampled_traj_rs, ref_traj_rs, is_reverse=(new_dir == -1.0))
            total_cost += self._longitudinal_cost(sampled_traj_rs, ref_traj_rs)

            all_costs.append(total_cost)
            all_noises.append(noises)
            all_trajs.append(sampled_traj)

        # Rank all trajectories according to its cost
        all_costs = np.array(all_costs)
        all_noises = np.array(all_noises)
        weights = self._compute_weights(all_costs)

        # Final trajectory is the weighted average
        final_noise = np.zeros_like(all_noises[0])
        for k in range(self.num_samples):
            final_noise += weights[k] * all_noises[k]

        # Apply moving average filter
        w_epsilon = self._moving_average_filter(final_noise, self.filter_window_size)

        # Construct optimal actions
        actions = self.prev_actions + w_epsilon
        actions = np.clip(actions, self.env.action_space.space[0], self.env.action_space.space[1])

        # Visualize local plan
        best_traj_idx = np.argmin(all_costs)
        self.best_traj = all_trajs[best_traj_idx].tolist()

        # Update previous actions
        self.prev_actions[:-1] = actions[1:]
        self.prev_actions[-1] = actions[-1]

        return actions[0]

    def _compute_weights(self, traj_costs: np.ndarray) -> np.ndarray:
        w = np.zeros(self.num_samples)
        rho = traj_costs.min()
        eta = 0.0
        for k in range(self.num_samples):
            eta += np.exp((-1.0 / self.param_lambda) * (traj_costs[k] - rho))
        for k in range(self.num_samples):
            w[k] = (1.0 / eta) * np.exp((-1.0 / self.param_lambda) * (traj_costs[k] - rho))
        return w

    def _moving_average_filter(self, xx: np.ndarray, window_size: int) -> np.ndarray:
        b = np.ones(window_size) / window_size
        dim = xx.shape[1]
        xx_mean = np.zeros(xx.shape)
        for d in range(dim):
            xx_mean[:, d] = np.convolve(xx[:, d], b, mode='same')
        return xx_mean


# Main test block
if __name__ == '__main__':
    CUR_DIR = osp.dirname(osp.abspath(__file__))
    path_file = osp.join(CUR_DIR, 'test_path_se2.json')

    with open(path_file, 'r') as f:
        ref_path = json.load(f)

    # Initialize environment and reference path
    env = DiffDrive2DControl()
    env.reset(ref_path, empty=True)
    # ! Override start state to test path merging
    env.start_state = np.array([5.5, 5.0, 0.0])
    env.cur_state = np.array([5.5, 5.0, 0.0])

    controller = FrenetMPPI(env, ref_path)

    # Debug visualization
    env.interactive_viz = True

    state = env.start_state
    path = [state]
    step = 0
    while True:
        action = controller.run(state)

        # Visualize current best local trajectory
        local_plan = controller.best_traj
        env.set_local_plan(local_plan)

        next_state, reward, term, trunc, _ = env.step(action)
        env.cur_carrot_pose_index = controller._lookahead_cursor
        # print(state, action, next_state, reward, term, trunc)

        # Get new progression direction for debug
        cur_pos_idx = controller.cur_pos_idx
        cur_dir = controller.progression_directions[min(cur_pos_idx, len(controller.progression_directions) - 1)]

        print(f"Step {step} | State: {state} | Action: {action} | Dir: {cur_dir} | PathIdx: {cur_pos_idx}")
        env.render()

        path.append(next_state)
        state = next_state

        if term or trunc:
            break

        step += 1
