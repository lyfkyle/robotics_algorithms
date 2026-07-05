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
        lateral_cost_weight: float = 2.0,
        heading_cost_min_weight: float = 0.1,
        heading_cost_max_weight: float = 0.5,
        heading_cost_lateral_sigma: float = 0.5,
        longitudinal_progress_weight: float = 1.0,
        se2_lambda: float = 0.2,
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
        self.se2_lambda = se2_lambda

        # Keep reference progress entirely inside controller.
        self._ref_cursor = 0
        self._ref_search_window = 200

        self._eval_points = self._sim_steps + 1

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

    def _compute_arc_lengths(self, states: np.ndarray) -> np.ndarray:
        return math_utils.se2_arc_lengths(states, self.se2_lambda)

    def _resample_states_by_arc_length(self, states: np.ndarray, num_points: int) -> np.ndarray:
        if states.shape[0] == 0:
            return states
        if states.shape[0] == 1 or num_points <= 1:
            return np.repeat(states[:1], max(num_points, 1), axis=0)

        arc = self._compute_arc_lengths(states)
        total = arc[-1]
        if total <= 1e-9:
            return np.repeat(states[:1], num_points, axis=0)

        target_arc = np.linspace(0.0, total, num_points)
        x_rs = np.interp(target_arc, arc, states[:, 0])
        y_rs = np.interp(target_arc, arc, states[:, 1])
        yaw_rs = np.interp(target_arc, arc, np.unwrap(states[:, 2]))
        return np.column_stack((x_rs, y_rs, yaw_rs))

    def _build_reference_segment(self, start_idx: int, end_idx: int) -> np.ndarray:
        start_idx = max(0, start_idx)
        end_idx = min(end_idx, len(self.ref_path) - 1)
        if end_idx <= start_idx:
            return self.ref_path[start_idx : start_idx + 1]
        return self.ref_path[start_idx : end_idx + 1]

    def _get_reference_start_idx(self, state: np.ndarray) -> int:
        """Find nearest reference index with a forward-only local search."""
        start_idx = self._ref_cursor
        end_idx = min(start_idx + self._ref_search_window, len(self.ref_path))
        if end_idx <= start_idx:
            return len(self.ref_path) - 1

        xy = np.array([state[0], state[1]])
        dists = np.linalg.norm(self.ref_path[start_idx:end_idx, :2] - xy, axis=1)
        nearest_idx = start_idx + int(np.argmin(dists))
        self._ref_cursor = nearest_idx
        return nearest_idx

    def _lateral_heading_cost(self, state: np.ndarray, ref_state: np.ndarray) -> float:
        x, y, yaw = state
        ref_x, ref_y, ref_yaw = ref_state
        normal = np.array([-np.sin(ref_yaw), np.cos(ref_yaw)])
        pos_err = np.array([x - ref_x, y - ref_y])
        lateral_err = np.dot(pos_err, normal).item()
        heading_err = math_utils.normalize_angle(yaw - ref_yaw)

        sigma = max(self.heading_cost_lateral_sigma, 1e-6)
        heading_weight = self.heading_cost_min_weight + (
            self.heading_cost_max_weight - self.heading_cost_min_weight
        ) * np.exp(-(lateral_err**2) / (sigma**2))

        return self.lateral_cost_weight * lateral_err**2 + heading_weight * heading_err**2

    def _get_closest_reference_idx(self, traj_end: np.ndarray, start_idx: int) -> int:
        """Find closest reference index to trajectory endpoint."""
        xy = np.array([traj_end[0], traj_end[1]])
        dists = np.linalg.norm(self.ref_path[start_idx:, :2] - xy, axis=1)
        if dists.size == 0:
            return len(self.ref_path) - 1
        return start_idx + int(np.argmin(dists))

    def _longitudinal_cost(self, traj_end: np.ndarray, start_idx: int, goal_threshold: float = 0.3) -> float:
        """Terminal progress cost based on where trajectory reaches on reference path.

        If trajectory endpoint is close to reference goal, penalize not stopping.
        Otherwise, reward reaching far along the reference path.
        """
        closest_idx = self._get_closest_reference_idx(traj_end, start_idx)
        ref_start = self.ref_path[start_idx]
        ref_goal = self.ref_path[-1]
        closest_ref = self.ref_path[closest_idx]

        # Distance from trajectory end to reference goal
        dist_to_goal = np.linalg.norm(traj_end[:2] - ref_goal[:2])

        # If close to goal, penalize not being exactly at goal
        if dist_to_goal < goal_threshold:
            return dist_to_goal  # Penalty to stop at goal

        # Otherwise, reward progress: distance from start to matched reference point
        progress_dist = np.linalg.norm(closest_ref[:2] - ref_start[:2])
        return -self.longitudinal_progress_weight * progress_dist  # Negative = reward

    def run(self, state: np.ndarray) -> np.ndarray:
        """Compute the current action given current state

        This function uses Dynamic Window Approach to select the best action given the current state.

        Args:
            state (np.ndarray): current state

        Returns:
            np.ndarray: current action
        """
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
        ref_start_idx = self._get_reference_start_idx(state)
        for action, traj_arr in all_trajs:
            if traj_arr.shape[0] < 2:
                continue

            ref_end_idx = ref_start_idx + traj_arr.shape[0] - 1
            ref_segment = self._build_reference_segment(ref_start_idx, ref_end_idx)
            ref_resampled = self._resample_states_by_arc_length(ref_segment, self._eval_points)
            traj_resampled = self._resample_states_by_arc_length(traj_arr, self._eval_points)

            total_cost = 0

            # Sum lateral and heading costs over all resampled points
            for k in range(self._eval_points):
                total_cost += self._lateral_heading_cost(traj_resampled[k], ref_resampled[k])

            # Add terminal progress cost based on trajectory endpoint
            total_cost += self._longitudinal_cost(traj_resampled[-1], ref_start_idx)

            # Update the best cost and trajectory if the current one is better
            if total_cost < best_cost:
                best_cost = total_cost
                self.best_traj = traj_arr.tolist()
                best_action = action

        # Return the best action
        return best_action


CUR_DIR = osp.join(osp.dirname(osp.abspath(__file__)))
path_file = osp.join(CUR_DIR, 'test_path_se2.json')

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
    print(state, action, next_state, reward, term, trunc)

    env.render()

    path.append(next_state)
    state = next_state

    if term or trunc:
        break
