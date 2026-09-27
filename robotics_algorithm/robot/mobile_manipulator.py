import numpy as np

from robotics_algorithm.robot.robot import Robot


class MobileManipulator2Link(Robot):
    """A mobile manipulator composed of a differential-drive circular base and a planar 2-link arm
    mounted at the base center.

    State: [x, y, theta, q1, q2]
        - x, y, theta: base pose (theta is the base heading)
        - q1, q2: arm joint angles, measured relative to the base heading

    Action: [v, omega, q1_dot, q2_dot]
        - v, omega: base linear and angular velocity (unicycle/differential-drive kinematics)
        - q1_dot, q2_dot: arm joint velocities
    """

    def __init__(self, link1_length: float = 0.3, link2_length: float = 0.2, dt: float = 0.1):
        super().__init__(dt)

        self.l1 = link1_length
        self.l2 = link2_length

    def control(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Integrate the mobile manipulator kinematics forward by one time step.

        Args:
            state (np.ndarray): [x, y, theta, q1, q2]
            action (np.ndarray): [v, omega, q1_dot, q2_dot]

        Returns:
            new state [x, y, theta, q1, q2]
        """
        x, y, theta, q1, q2 = state
        v, omega, q1_dot, q2_dot = action

        x_new = x + v * np.cos(theta) * self.dt
        y_new = y + v * np.sin(theta) * self.dt
        theta_new = theta + omega * self.dt
        q1_new = q1 + q1_dot * self.dt
        q2_new = q2 + q2_dot * self.dt

        return np.array([x_new, y_new, theta_new, q1_new, q2_new])

    def fk(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward kinematics of the arm.

        Args:
            state (np.ndarray): [x, y, theta, q1, q2]

        Returns:
            base_pos (np.ndarray): [x, y] base/shoulder position
            elbow_pos (np.ndarray): [x, y] position of the joint between link1 and link2
            eef_pos (np.ndarray): [x, y] end-effector position
        """
        x, y, theta, q1, q2 = state

        base_pos = np.array([x, y])
        elbow_pos = base_pos + self.l1 * np.array([np.cos(theta + q1), np.sin(theta + q1)])
        eef_pos = elbow_pos + self.l2 * np.array([np.cos(theta + q1 + q2), np.sin(theta + q1 + q2)])

        return base_pos, elbow_pos, eef_pos

    def eef_pos(self, state: np.ndarray) -> np.ndarray:
        """End-effector position for a given state."""
        _, _, eef_pos = self.fk(state)
        return eef_pos
