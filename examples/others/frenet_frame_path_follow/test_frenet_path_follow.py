import json
import os.path as osp
import numpy as np

from robotics_algorithm.env.continuous_2d.diff_drive_2d_control import DiffDrive2DControl
from robotics_algorithm.utils import math_utils


class FrenetDWB:
    """This class implements a DWB-style controller that uses Frenet frame for trajectory scoring.

    The key differences include:
        1. The trajectory sampled is represented cartesian space instead of velocity space.
        2. The trajectory scoring is done in Frenet frame instead of cartesian frame.

    """

    def __init__(
        self,
        env: DiffDrive2DControl,
        ref_path: np.ndarray,
        min_lin_vel: float = -0.5,
        max_lin_vel: float = 0.5,
        lin_vel_samples: int = 10,
        min_ang_vel: float = -1.0,
        max_ang_vel: float = 1.0,
        ang_vel_samples: int = 20,
        simulate_time: float = 0.5,
        lateral_cost_weight: float = 15.0,
        heading_cost_min_weight: float = 0.001,
        heading_cost_max_weight: float = 1.0,
        heading_cost_lateral_sigma: float = 0.5,
        longitudinal_progress_weight: float = 1.0,
        w_theta: float = 0.2,
        spatial_resolution: float = 0.02,
        rotation_shim_threshold: float = 0.25,
        lookahead_dist: float = 0.5,
    ) -> None:
        """
        Constructor

        Args:
            env (DiffDrive2DControl): The environment.
            ref_path (np.ndarray): Reference trajectory with [x, y, yaw].
            min_lin_vel (float): Minimum linear velocity.
            max_lin_vel (float): Maximum linear velocity.
            lin_vel_samples (int): Number of linear velocity samples.
            min_ang_vel (float): Minimum angular velocity.
            max_ang_vel (float): Maximum angular velocity.
            ang_vel_samples (int): Number of angular velocity samples.
            simulate_time (float): Simulation time.
            lateral_cost_weight (float): Weight for lateral tracking error.
            heading_cost_min_weight (float): Minimum heading-error weight.
            heading_cost_max_weight (float): Maximum heading-error weight.
            heading_cost_lateral_sigma (float): Sigma for heading weight scheduling.
            longitudinal_progress_weight (float): Weight for longitudinal progress reward.
            se2_lambda (float): Scaling factor [m/rad] for angular contribution to arc-length.
                Allows SE(2) distance to handle in-place rotations. Defaults to 0.2.
            spatial_resolution (float): Fixed distance spacing [m] for path/trajectory resampling.
            rotation_shim_threshold (float): Heading error threshold [rad] to trigger the rotation shim.
        """
        assert isinstance(env, DiffDrive2DControl), 'env must be a DiffDrive2DControl'

        self.env = env
        self.ref_path = np.asarray(ref_path)
        self._min_lin_vel = min_lin_vel
        self._max_lin_vel = max_lin_vel
        self._lin_vel_steps = (max_lin_vel - min_lin_vel) / lin_vel_samples
        self._min_ang_vel = min_ang_vel
        self._max_ang_vel = max_ang_vel
        self._ang_vel_steps = (max_ang_vel - min_ang_vel) / ang_vel_samples
        self._sim_steps = int(simulate_time / env.action_dt)

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
        self._ref_cursor = 0
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
                # 0.0 represents pure in-place rotation (no translation progression)
                # This naturally detects both the start and end of rotation segments as cusps!
                self.progression_directions[i] = 0.0

        # Precompute actions and local frame trajectories together.
        self.sampled_action_seq, self._local_trajs = self._precompute_local_trajectories()

    def _precompute_local_trajectories(self) -> tuple:
        """Sample all (v, omega) actions and analytically compute their local-frame trajectories.

        For constant (v, omega), diff-drive kinematics gives an exact arc:
            x(t) = (v/omega)*sin(omega*t)
            y(t) = (v/omega)*(1 - cos(omega*t))
            yaw(t) = omega*t
        For omega~0 this reduces to a straight line.

        Returns:
            actions: list of [v, omega] pairs.
            trajs: list of (N+1, 3) local-frame state arrays.
        """
        dt = self.env.action_dt
        actions = []
        trajs = []
        for v in np.arange(self._min_lin_vel, self._max_lin_vel, self._lin_vel_steps):
            for omega in np.arange(self._min_ang_vel, self._max_ang_vel, self._ang_vel_steps):
                actions.append([v, omega])
                states = np.zeros((self._sim_steps + 1, 3))
                for k in range(1, self._sim_steps + 1):
                    t = k * dt
                    if abs(omega) > 1e-6:
                        states[k, 0] = (v / omega) * np.sin(omega * t)
                        states[k, 1] = (v / omega) * (1.0 - np.cos(omega * t))
                    else:
                        states[k, 0] = v * t
                        states[k, 1] = 0.0
                    states[k, 2] = omega * t
                trajs.append(states)

        return actions, trajs

    def _transform_to_world(self, local_traj: np.ndarray, robot_state: np.ndarray) -> np.ndarray:
        """Apply SE(2) transform: local frame → world frame."""
        x_r, y_r, yaw_r = robot_state
        cos_r, sin_r = np.cos(yaw_r), np.sin(yaw_r)
        x_w = x_r + local_traj[:, 0] * cos_r - local_traj[:, 1] * sin_r
        y_w = y_r + local_traj[:, 0] * sin_r + local_traj[:, 1] * cos_r
        yaw_w = yaw_r + local_traj[:, 2]
        return np.column_stack((x_w, y_w, yaw_w))

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
        """Build reference segment starting from start_idx up to a constant physical lookahead distance,
        stopping strictly if we encounter a cusp.
        """
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
        """Find nearest reference index with a forward-only local search capped at the next cusp using SE(2) distance."""
        start_idx = self._ref_cursor

        # Find next cusp to restrict our search window and prevent jumping across overlaps
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

        # Only advance to or past the cusp if the robot has physically crossed it or is extremely close
        if nearest_idx >= next_cusp and next_cusp < len(self.ref_path) - 1:
            cusp_state = self.ref_path[next_cusp]
            prev_state = self.ref_path[max(0, next_cusp - 1)]

            # u = cusp_state[:2] - prev_state[:2]
            v = cusp_state[:2] - state[:2]

            dist_to_cusp = np.linalg.norm(v)
            if dist_to_cusp > 0.05:
                nearest_idx = max(start_idx, next_cusp - 1)
            else:
                # If it's a pure rotation cusp, check angular error against the same shim threshold
                heading_err_cusp = math_utils.normalize_angle(cusp_state[2] - state[2])
                if abs(heading_err_cusp) > self.rotation_shim_threshold:
                    nearest_idx = max(start_idx, next_cusp - 1)

        self._ref_cursor = nearest_idx
        return nearest_idx

    def _lateral_heading_cost(self, sampled_traj: np.ndarray, ref_traj: np.ndarray) -> float:
        min_len = min(len(ref_traj), len(sampled_traj))
        if min_len < 1:
            return 0.0

        total_cost = 0.0
        # Stanley parameters
        k_lateral = 5.0  # Proportional gain for crosstrack steering

        for k in range(min_len):
            x, y, yaw = sampled_traj[k]
            ref_x, ref_y, ref_yaw = ref_traj[k]
            normal = np.array([-np.sin(ref_yaw), np.cos(ref_yaw)])
            pos_err = np.array([x - ref_x, y - ref_y])

            # Signed lateral error (positive means robot is to the left of the path)
            lateral_err_signed = np.dot(pos_err, normal).item()
            lateral_err = abs(lateral_err_signed)

            # Speed-independent crosstrack heading correction (Kinematic Stanley):
            # Uses arctan as a smooth, infinitely differentiable saturation function
            crosstrack_heading = np.arctan(-k_lateral * lateral_err_signed)
            yaw_desired = ref_yaw + crosstrack_heading

            # Evaluate heading error relative to this desired corrective heading
            heading_err = abs(math_utils.normalize_angle(yaw - yaw_desired))

            # Constant weights! No more exponential hacks.
            total_cost += self.lateral_cost_weight * lateral_err + self.heading_cost_max_weight * heading_err

        return total_cost / min_len

    def _longitudinal_cost(self, sampled_traj: np.ndarray, ref_traj: np.ndarray, goal_threshold: float = 0.3) -> float:
        """Terminal progress and goal attraction cost.

        Always continuous to prevent local minima or reverse-stuck behavior near the goal.
        """
        traj_end = sampled_traj[-1]

        # Calculate closest index on ref_traj (which is ref_traj_rs) using math_utils.se2_distance
        dists = np.array([math_utils.se2_distance(pt, traj_end, w_theta=self.w_theta) for pt in ref_traj])

        if dists.size == 0:
            closest_idx = len(ref_traj) - 1
        else:
            closest_idx = int(np.argmin(dists))

        ref_start = ref_traj[0]
        ref_goal = ref_traj[-1]
        closest_ref = ref_traj[closest_idx]

        # SE(2) distance from trajectory end to reference goal
        dist_to_goal = math_utils.se2_distance(traj_end, ref_goal, w_theta=self.w_theta)

        # SE(2) progress distance along the current segment
        progress_dist = math_utils.se2_distance(closest_ref, ref_start, w_theta=self.w_theta)

        # Scale progress by heading alignment to force steering/correction when misaligned (exact kinematic projection)
        heading_err = math_utils.normalize_angle(traj_end[2] - closest_ref[2])
        alignment_factor = np.cos(heading_err)
        if alignment_factor > 0:
            # We scale progress by alignment_factor (1st power) to represent actual kinematic progression along the path coordinate
            progress_reward = -self.longitudinal_progress_weight * progress_dist * alignment_factor
        else:
            progress_reward = 0.0

        # Smoothly blend progress reward and goal attraction
        # Near the goal, we fade out progress reward to prioritize exact stopping
        progress_fade = np.clip(dist_to_goal / 1.0, 0.0, 1.0)

        cost = progress_reward * progress_fade

        # Add a continuous goal attraction cost
        goal_attraction_weight = 2.0
        cost += goal_attraction_weight * dist_to_goal

        return cost

    def run(self, state: np.ndarray) -> np.ndarray:
        """Compute the current action given current state

        This function uses Dynamic Window Approach to select the best action given the current state.

        Args:
            state (np.ndarray): current state

        Returns:
            np.ndarray: current action
        """
        # 1. Exact Stop Condition
        goal_state = self.ref_path[-1]
        dist_to_goal = math_utils.se2_distance(state, goal_state, w_theta=self.w_theta)
        if dist_to_goal < 0.05:
            self.best_traj = [state]
            return np.array([0.0, 0.0])

        # 2. Get current reference start index and build reference segment
        ref_start_idx = self._get_reference_start_idx(state)
        ref_segment = self._build_reference_segment(ref_start_idx, self.lookahead_dist)

        # 3. Unified Rotation Shim Triggering
        current_ref_state = self.ref_path[ref_start_idx]
        current_heading_err = math_utils.normalize_angle(current_ref_state[2] - state[2])

        # Check for in-place segment (extremely small translation over lookahead)
        if len(ref_segment) > 1:
            translation_dist = np.linalg.norm(ref_segment[-1, :2] - ref_segment[0, :2])
        else:
            translation_dist = 0.0

        lookahead_state = self.ref_path[self._lookahead_cursor]
        lookahead_heading_err = math_utils.normalize_angle(lookahead_state[2] - state[2])

        trigger_shim = False
        target_heading_err = 0.0

        if abs(current_heading_err) > self.rotation_shim_threshold:
            # Case A: Severe misalignment with current path pose (initialization or extreme drift)
            trigger_shim = True
            target_heading_err = current_heading_err
        elif translation_dist < 0.05:
            # Case B: On an in-place rotation segment (pure rotation cusp)
            if abs(lookahead_heading_err) > self.rotation_shim_threshold:
                # Still rotating: keep shim active
                trigger_shim = True
                target_heading_err = lookahead_heading_err
            else:
                # Heading error is now small enough to exit!
                # Directly jump the cursor and the start index/segment to the end of the rotation so the carrot moves ahead immediately in this frame
                print(f"In-place rotation complete! Directly jumping cursor from {self._ref_cursor} to end of rotation segment {self._lookahead_cursor}")
                self._ref_cursor = self._lookahead_cursor
                ref_start_idx = self._lookahead_cursor
                ref_segment = self._build_reference_segment(ref_start_idx, self.lookahead_dist)

        # Execute Rotation Shim if triggered
        if trigger_shim:
            print("Rotation Shim Mode!!!!!")
            # Proportional control for pure rotation
            kp_omega = 2.0
            omega = np.clip(kp_omega * target_heading_err, self._min_ang_vel, self._max_ang_vel)
            # Create a mock trajectory of pure rotation for visualization
            traj = [state]
            dt = self.env.action_dt
            cur_yaw = state[2]
            for _ in range(self._sim_steps):
                cur_yaw += omega * dt
                traj.append(np.array([state[0], state[1], cur_yaw]))
            self.best_traj = np.array(traj).tolist()
            return np.array([0.0, omega])

        # 4. Standard DWB
        best_cost = np.inf
        best_action = np.array([0.0, 0.0])
        self.best_traj = [state]

        # Transform all precomputed local trajectories to world frame — no env stepping needed.
        all_trajs = []
        for i, (lin_vel, ang_vel) in enumerate(self.sampled_action_seq):
            action = np.array([lin_vel, ang_vel])
            traj = self._transform_to_world(self._local_trajs[i], state)
            all_trajs.append((action, traj))

        # Iterate over all possible trajectories
        for action, sampled_traj in all_trajs:
            if sampled_traj.shape[0] < 2:
                continue

            # Resample both reference and trajectory using the fixed spatial resolution (reuse ref_segment)
            ref_traj_rs = self._resample_states_by_spatial_resolution(ref_segment, self.spatial_resolution)
            sampled_traj_rs = self._resample_states_by_spatial_resolution(sampled_traj, self.spatial_resolution)

            # Sum lateral and heading costs over all comparable resampled points (inside helper)
            total_cost = self._lateral_heading_cost(sampled_traj_rs, ref_traj_rs)

            # Add terminal progress cost based on trajectory endpoint (inside helper)
            total_cost += self._longitudinal_cost(sampled_traj_rs, ref_traj_rs)

            # Update the best cost and trajectory if the current one is better
            if total_cost < best_cost:
                best_cost = total_cost
                self.best_traj = sampled_traj.tolist()
                best_action = action

        # Return the best action
        return best_action


CUR_DIR = osp.join(osp.dirname(osp.abspath(__file__)))
path_file = osp.join(CUR_DIR, 'test_path_se2_easy.json')

with open(path_file, 'r') as f:
    ref_path = json.load(f)

# Initialize environment and reference path.
env = DiffDrive2DControl()
env.reset(ref_path, empty=True)

controller = FrenetDWB(env, ref_path)

# Debug visualization.
env.interactive_viz = True

state = env.start_state
path = [state]
while True:
    action = controller.run(state)

    # Visualize current best local trajectory.
    local_plan = controller.best_traj
    env.set_local_plan(local_plan)

    next_state, reward, term, trunc, _ = env.step(action)
    # Synchronize the environment's carrot index with the controller's lookahead cursor for accurate plotting
    env.cur_carrot_pose_index = controller._lookahead_cursor
    print(state, action, next_state, reward, term, trunc)

    env.render()

    path.append(next_state)
    state = next_state

    if term or trunc:
        break
