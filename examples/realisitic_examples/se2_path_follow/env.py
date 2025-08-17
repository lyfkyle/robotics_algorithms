from typing_extensions import override
from robotics_algorithm.utils import math_utils
import numpy as np

from robotics_algorithm.utils import math_utils
from robotics_algorithm.env.continuous_2d.diff_drive_2d_control import DiffDrive2DControl


class DiffDriveSE2PathFollow(DiffDrive2DControl):
    """This class implements a differential drive robot environment that follows a reference path in SE(2) space. The path may consist of in-place rotations or reverse.

    NOTE: we borrow DiffDrive2DControl base class. Some charaacteristics of the base class is not used such as quadratic reward function.
    Args:
        DiffDrive2DControl (_type_): _description_

    Returns:
        _type_: _description_
    """

    @override
    def step(self, action):
        res = super().step(action)

        # update current reference path index if reference path is present
        if self.ref_path is not None and self.lookahead_index > 0:
            self.cur_ref_path_idx = self.get_nearest_waypoint_to_state_se2(res[0], self.ref_path)
            self.cur_carrot_pose_index = min(self.cur_ref_path_idx + self.lookahead_index, len(self.ref_path) - 1)

        return res

    @override
    def reward_func(self, state, action=None, new_state=None):
        return 0

    def traj_cost_func(self, sampled_traj, action, w_prox=0.1, w_lateral=100.0, w_prog=0.02):
        """Calculate the total reward for a trajectory."""

        proximity_cost = self.proximity_cost(sampled_traj)
        progress_cost = self.progress_cost(sampled_traj)
        lateral_cost = self.lateral_cost(sampled_traj)
        print(action, proximity_cost, lateral_cost, progress_cost)
        total_cost = w_prox * proximity_cost + w_lateral * lateral_cost + w_prog * progress_cost

        return total_cost

    def discretize_se2_trajectory(self, trajectory, resolution=-1, num_points=-1):
        """
        Discretize a continuous SE(2) trajectory into discrete waypoints.

        Args:
            trajectory (np.ndarray): Continuous trajectory as an array of shape (N, 3),
                                    where each row is [x, y, theta].
            resolution (float): Distance between consecutive waypoints in the discretized trajectory.

        Returns:
            np.ndarray: Discretized trajectory as an array of shape (M, 3).
        """
        if resolution <= 0 and num_points < 0:
            raise ValueError('Either resolution or num_points must be specified.')

        discretized_trajectory = []
        for i in range(len(trajectory) - 1):
            start = trajectory[i]
            end = trajectory[i + 1]
            distance = math_utils.se2_distance(
                start, end, w_theta=0.5
            )  # Set w_theta to a large value to emphasize more angle difference

            # Number of intermediate points to add
            if num_points < 0:
                num_points = int(np.ceil(distance / resolution))
            diff = math_utils.se2_diff(start, end)
            step_size = diff / num_points

            for j in range(num_points):
                interpolated_point = start + step_size * j
                interpolated_point[2] = math_utils.normalize_angle(interpolated_point[2])
                discretized_trajectory.append(interpolated_point)

        discretized_trajectory.append(trajectory[-1])  # Ensure the last point is included
        return np.array(discretized_trajectory)

    def proximity_cost(self, sampled_traj):
        """Calculate the proximity cost for the sampled trajectory wrt to reference trajectory."""
        discretized_ref_path = self.discretize_se2_trajectory(
            self.ref_path[self.cur_ref_path_idx : self.cur_carrot_pose_index + 1], resolution=0.005
        )

        reached_pose_idx = self.get_nearest_waypoint_to_state_se2(sampled_traj[-1], discretized_ref_path)

        discretized_sampled_path = self.discretize_se2_trajectory(sampled_traj, resolution=0.005)
        discretized_ref_path = discretized_ref_path[: reached_pose_idx + 1]
        # print(discretized_sampled_path, discretized_ref_path)
        # Calculate absolute trajectory error.
        sampled_traj_len = len(discretized_sampled_path)
        ref_traj_len = len(discretized_ref_path)
        N = max(sampled_traj_len, ref_traj_len)

        total_cost = 0.0
        for i in range(N):
            if i >= sampled_traj_len:
                total_cost += math_utils.se2_distance(
                    discretized_sampled_path[-1], discretized_ref_path[i], w_theta=0.5
                )
                # rel_pose = math_utils.transform_to_frame(discretized_sampled_path[-1], discretized_ref_path[i])
            elif i >= ref_traj_len:
                total_cost += math_utils.se2_distance(
                    discretized_sampled_path[i], discretized_ref_path[-1], w_theta=0.5
                )
                # rel_pose = math_utils.transform_to_frame(discretized_sampled_path[i], discretized_ref_path[-1])
            else:
                total_cost += math_utils.se2_distance(discretized_sampled_path[i], discretized_ref_path[i], w_theta=0.5)
                # rel_pose = math_utils.transform_to_frame(discretized_sampled_path[i], discretized_ref_path[i])
            # total_cost += np.sqrt(rel_pose[1] ** 2 + 0.1 * rel_pose[2] ** 2)
        return total_cost

    def lateral_cost(self, sampled_traj):
        """Calculate the proximity cost for the sampled trajectory wrt to reference trajectory."""
        discretized_ref_path = self.discretize_se2_trajectory(
            self.ref_path[self.cur_ref_path_idx : self.cur_carrot_pose_index + 1], resolution=0.005
        )

        reached_pose_idx = self.get_nearest_waypoint_to_state_se2(sampled_traj[-1], discretized_ref_path)

        discretized_sampled_path = self.discretize_se2_trajectory(sampled_traj, resolution=0.005)
        discretized_ref_path = discretized_ref_path[: reached_pose_idx + 1]
        # print(discretized_sampled_path, discretized_ref_path)
        # Calculate absolute trajectory error.
        sampled_traj_len = len(discretized_sampled_path)
        ref_traj_len = len(discretized_ref_path)
        N = max(sampled_traj_len, ref_traj_len)

        total_cost = 0.0
        for i in range(N):
            if i >= sampled_traj_len:
                # total_cost += math_utils.se2_distance(
                #     discretized_sampled_path[-1], discretized_ref_path[i], w_theta=0.5
                # )
                rel_pose = math_utils.transform_to_frame(discretized_sampled_path[-1], discretized_ref_path[i])
            elif i >= ref_traj_len:
                # total_cost += math_utils.se2_distance(
                #     discretized_sampled_path[i], discretized_ref_path[-1], w_theta=0.5
                # )
                rel_pose = math_utils.transform_to_frame(discretized_sampled_path[i], discretized_ref_path[-1])
            else:
                # total_cost += math_utils.se2_distance(discretized_sampled_path[i], discretized_ref_path[i], w_theta=0.5)
                rel_pose = math_utils.transform_to_frame(discretized_sampled_path[i], discretized_ref_path[i])
            total_cost += np.sqrt(rel_pose[1] ** 2)
        return total_cost / N

    def progress_cost(self, sampled_path):
        # discretize both sampled trajectory and reference path
        discretized_sampled_path = self.discretize_se2_trajectory(sampled_path, resolution=0.005)
        discretized_ref_path = self.discretize_se2_trajectory(
            self.ref_path[self.cur_ref_path_idx : self.cur_carrot_pose_index + 1], resolution=0.005
        )

        reached_pose_idx = self.get_nearest_waypoint_to_state_se2(discretized_sampled_path[-1], discretized_ref_path)

        # return math_utils.se2_distance(discretized_ref_path[reached_pose_idx], discretized_ref_path[-1])
        return len(discretized_ref_path) - reached_pose_idx

    def get_nearest_waypoint_to_state_se2(self, state: np.ndarray, ref_path):
        best_dist = np.inf
        nearest_idx = -1
        for i, waypoint in enumerate(ref_path):
            dist = math_utils.se2_distance(state, waypoint, w_theta=0.5)  # Emphasize angle difference
            if dist < best_dist:
                best_dist = dist
                nearest_idx = i

        return nearest_idx
