"""Standalone differential drive kinematics model with a rectangular footprint.

Self-contained (no dependency on the `robotics_algorithm` package) so that this demo
lives entirely outside of it.
"""

import numpy as np


def normalize_angle(angle: float) -> float:
    """Normalize an angle to [-pi, pi]."""
    angle = (angle + 6 * np.pi) % (2 * np.pi)
    if angle > np.pi:
        angle -= 2 * np.pi
    return angle


class DiffDriveRect:
    """Differential drive robot with a rectangular footprint.

    State: [x, y, theta] (theta = 0 means the robot's length axis points along +x).
    Action: [lin_vel, ang_vel].
    """

    def __init__(self, wheel_radius: float, wheel_dist: float, length: float, width: float, dt: float = 0.1):
        self.wheel_radius = wheel_radius
        self.wheel_dist = wheel_dist
        self.length = length  # footprint extent along the heading axis
        self.width = width  # footprint extent perpendicular to the heading axis
        self.dt = dt

        # Corners in the body frame, fixed order: front-left, front-right, rear-right, rear-left.
        half_l, half_w = length / 2.0, width / 2.0
        self._body_corners = np.array(
            [
                [half_l, half_w],
                [half_l, -half_w],
                [-half_l, -half_w],
                [-half_l, half_w],
            ]
        )

    def control(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Advance the unicycle kinematics by one dt given [lin_vel, ang_vel].

        Args:
            state (np.ndarray): [x, y, theta]
            action (np.ndarray): [lin_vel, ang_vel]

        Returns:
            new state [x, y, theta]
        """
        x, y, theta = state
        lin_vel, ang_vel = action

        x_new = x + lin_vel * np.cos(theta) * self.dt
        y_new = y + lin_vel * np.sin(theta) * self.dt
        theta_new = normalize_angle(theta + ang_vel * self.dt)

        return np.array([x_new, y_new, theta_new])

    def control_wheel_speed(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """Advance the robot given left/right wheel angular velocities.

        Args:
            state (np.ndarray): [x, y, theta]
            control (np.ndarray): [v_l, v_r] left and right wheel angular velocities (rad/s)

        Returns:
            new state [x, y, theta]
        """
        v_l, v_r = control
        lin_vel = self.wheel_radius * (v_r + v_l) / 2.0
        ang_vel = self.wheel_radius * (v_r - v_l) / self.wheel_dist
        return self.control(state, np.array([lin_vel, ang_vel]))

    def get_corner_positions(self, state: np.ndarray) -> np.ndarray:
        """Return the world-frame positions of the 4 footprint corners.

        Order: front-left, front-right, rear-right, rear-left.

        Args:
            state (np.ndarray): [x, y, theta]

        Returns:
            np.ndarray of shape (4, 2)
        """
        x, y, theta = state
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        world_corners = self._body_corners @ rot.T + np.array([x, y])
        return world_corners
