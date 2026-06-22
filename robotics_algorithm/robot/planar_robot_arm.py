import matplotlib.pyplot as plt
import numpy as np
from typing_extensions import override

from robotics_algorithm.robot.robot import Robot
from robotics_algorithm.utils.transformation import transformation_matrix_2d


class PlanarRobotArm(Robot):
    def __init__(self, dt=0.01):
        """A planar 3 link robot arm with fixed base.

        Link 1 and 3 are revolute joints, link 2 is a prismatic joint.

        State: [theta1, theta1_dot, p2, p2_dot, theta3, theta3_dot]
        Action: []

        """
        super().__init__(dt)

        self.l1 = 0.1  # link1 length
        self.l2 = 0.1  # link2 length
        self.l3 = 0.1  # link3 length

    def forward_kinematics(self, q):
        theta1, p2, theta3 = q

        # ! Given a SE2 pose (x, y, theta), transformation matrix is constructed as T = p @ R, where p is translation and R is rotation. The order matters.
        # ! Therefore here we first apply rotation and then translation for each joint, and multiply them in the correct order to get the final transformation matrix.
        R_01 = transformation_matrix_2d(0, 0, theta1)
        p_01 = transformation_matrix_2d(self.l1, 0, 0)
        T_01 = R_01 @ p_01

        R_12 = transformation_matrix_2d(0, 0, 0)  # no rotation for prismatic joint
        p_12 = transformation_matrix_2d(p2, 0, 0)  # only translate along the x axis of the current frame
        T_12 = R_12 @ p_12

        R_23 = transformation_matrix_2d(0, 0, theta3)
        p_23 = transformation_matrix_2d(self.l3, 0, 0)
        T_23 = R_23 @ p_23

        T_03 = T_01 @ T_12 @ T_23
        return T_03[0:2, 2]  # return the end effector position (x, y)

    def inverse_kinematics(eef_pos):
        pass

    @override
    def control(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        pass

    @override
    def state_transition_jacobian(self, state, action):
        pass

    def get_joint_positions(self, q):
        """Compute positions of all joints given configuration q.

        Returns list of (x, y) positions: [base, joint1, joint2, end_effector]
        """
        theta1, p2, theta3 = q

        base = np.array([0.0, 0.0])

        # Joint 1: end of link 1
        joint1 = base + np.array([self.l1 * np.cos(theta1), self.l1 * np.sin(theta1)])

        # Joint 2: end of link 2 (prismatic)
        joint2 = joint1 + np.array([p2 * np.cos(theta1), p2 * np.sin(theta1)])

        # End effector: end of link 3
        eef = joint2 + np.array([self.l3 * np.cos(theta1 + theta3), self.l3 * np.sin(theta1 + theta3)])

        return [base, joint1, joint2, eef]

    def plot_configuration(self, q, ax=None, title='Robot Arm Configuration'):
        """Plot the robot arm at configuration q.

        Args:
            q: Joint configuration [theta1, p2, theta3]
            ax: Matplotlib axis (creates new figure if None)
            title: Plot title
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        positions = self.get_joint_positions(q)
        positions = np.array(positions)

        # Plot links
        ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=3, label='Links (get_joint_positions)')

        # Plot joints
        ax.plot(positions[:-1, 0], positions[:-1, 1], 'ro', markersize=8, label='Joints')

        # Plot end effector from get_joint_positions
        ax.plot(positions[-1, 0], positions[-1, 1], 'g*', markersize=20, label='End Effector (get_joint_positions)')

        # Plot forward kinematics result
        fk_pos = self.forward_kinematics(q)
        ax.plot(fk_pos[0], fk_pos[1], 'c+', markersize=20, markeredgewidth=3, label='End Effector (forward_kinematics)')

        # Plot base
        ax.plot(0, 0, 'ks', markersize=10, label='Base')

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title(title)

        # Set reasonable limits
        max_reach = self.l1 + self.l2 + self.l3
        ax.set_xlim(-max_reach * 1.2, max_reach * 1.2)
        ax.set_ylim(-max_reach * 1.2, max_reach * 1.2)

        # Verify they match
        error = np.linalg.norm(positions[-1] - fk_pos)
        if error > 1e-10:
            ax.text(
                0.05,
                0.95,
                f'ERROR: FK mismatch = {error:.2e}',
                transform=ax.transAxes,
                color='red',
                fontweight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
            )
        else:
            ax.text(
                0.05,
                0.95,
                f'✓ FK verified (error = {error:.2e})',
                transform=ax.transAxes,
                color='green',
                fontweight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
            )

        return ax


if __name__ == '__main__':
    arm = PlanarRobotArm()

    # Plot several random configurations
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(4):
        # Random configuration: [theta1, p2, theta3]
        theta1 = np.random.uniform(-np.pi, np.pi)
        p2 = np.random.uniform(0.0, 0.2)  # prismatic joint extension
        theta3 = np.random.uniform(-np.pi, np.pi)
        q = np.array([theta1, p2, theta3])

        arm.plot_configuration(q, ax=axes[i], title=f'Config {i + 1}: θ₁={theta1:.2f}, p₂={p2:.3f}, θ₃={theta3:.2f}')

    plt.tight_layout()
    plt.show()
