import matplotlib.pyplot as plt
import numpy as np
from typing_extensions import override
from scipy.linalg import expm, logm

from robotics_algorithm.robot.robot import Robot
import robotics_algorithm.utils.transformation as tf_utils
import robotics_algorithm.utils.math_utils as math_utils


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

        self.T_base2eef = None

    def forward_kinematics(self, q, method='transformation'):
        theta1, p2, theta3 = q

        if method == 'geometric':
            base = np.array([0.0, 0.0])

            # Joint 1: end of link 1
            joint1 = base + np.array([self.l1 * np.cos(theta1), self.l1 * np.sin(theta1)])

            # Joint 2: end of link 2 (prismatic)
            joint2 = joint1 + np.array([p2 * np.cos(theta1), p2 * np.sin(theta1)])

            # End effector: end of link 3
            eef = joint2 + np.array([self.l3 * np.cos(theta1 + theta3), self.l3 * np.sin(theta1 + theta3)])

            eef_theta = math_utils.normalize_angle(theta1 + theta3)

            self.T_base2eef = tf_utils.get_T([eef[0], eef[1], 0], [0, 0, eef_theta])

            return np.array([eef[0], eef[1], eef_theta])

        elif method == 'transformation':
            # get_T(pos, rotvec) builds T = [R, t; 0, 1] with t = pos taken as-is (not rotated). Each joint needs
            # to rotate first and then translate along the *rotated* link offset, so we must compose Rot(theta) @
            # Trans(l, 0, 0) as two separate transforms rather than passing the offset directly into get_T.
            T_01 = tf_utils.get_T([0, 0, 0], [0, 0, theta1]) @ tf_utils.get_T([self.l1, 0, 0], [0, 0, 0])
            T_12 = tf_utils.get_T([p2, 0, 0], [0, 0, 0])  # no rotation for prismatic joint
            T_23 = tf_utils.get_T([0, 0, 0], [0, 0, theta3]) @ tf_utils.get_T([self.l3, 0, 0], [0, 0, 0])

            T_03 = T_01 @ T_12 @ T_23

            self.T_base2eef = T_03
            return tf_utils.se3_to_planar_pose(T_03)  # return the end effector pose (x, y, theta)

        elif method == 'exponential':
            screw_1 = np.array([0, 0, 1, 0, 0, 0])  # screw axis for revolute joint 1 at the base
            screw_2 = np.array(
                [0, 0, 0, 1, 0, 0]
            )  # screw axis for prismatic joint 2 at the base, assuming joint1 is at zero position
            screw_3 = np.array(
                [0, 0, 1, 0, -self.l1, 0]
            )  # screw axis for revolute joint 3 at the base, assuming joint1 and joint2 are at its zero position

            T_01 = expm(tf_utils.get_twist_matrix(screw_1) * theta1)
            T_12 = expm(tf_utils.get_twist_matrix(screw_2) * p2)
            T_23 = expm(tf_utils.get_twist_matrix(screw_3) * theta3)

            # end-effector pose at the home configuration (all joints zero), as a 4x4 SE(3) matrix
            M = np.eye(4)
            M[0, 3] = self.l1 + self.l3

            # Product of exponential formula for forward kinematics
            T_03 = T_01 @ T_12 @ T_23 @ M
            self.T_base2eef = T_03

            return tf_utils.se3_to_planar_pose(T_03)  # return the end effector pose (x, y, theta)

    def inverse_kinematics_analytic(self, eef_pose_se2):
        """Analytically solve for joint configuration given a desired end-effector pose.

        Args:
            eef_pose_se2: Desired end-effector pose (x, y, phi), where phi = theta1 + theta3.

        Returns:
            q: Joint configuration [theta1, p2, theta3]
        """
        x, y, phi = eef_pose_se2

        # This is the link2 end position/joint3 position
        wx = x - self.l3 * np.cos(phi)
        wy = y - self.l3 * np.sin(phi)

        # Since joint2 is prismatic, we can just use arc to get theta1
        theta1 = np.arctan2(wy, wx)

        # Compute prismatic joint extension p2 based on the distance from joint1 to joint2
        p2 = np.sqrt(wx**2 + wy**2) - self.l1

        # Finally get theta3
        theta3 = phi - theta1

        return np.array([theta1, p2, theta3])

    def analytical_jacobian(self, state) -> np.ndarray:
        """Compute the analytical Jacobian of the forward kinematics, as a full 6x3 SE(3) twist Jacobian
        [wx, wy, wz, vx, vy, vz] (in world-aligned axes, i.e. LOCAL_WORLD_ALIGNED convention).

        Args:
            state (np.ndarray): joint configuration [theta1, p2, theta3]

        Returns:
            np.ndarray: 6x3 Jacobian matrix relating joint velocities to end-effector twist
        """
        # Just manually differentiating the fk_simple equations to get the Jacobian
        theta1, p2, theta3 = state
        s1 = np.sin(theta1)
        c1 = np.cos(theta1)
        s3 = np.sin(theta3)
        c3 = np.cos(theta3)

        dx_dtheta1 = -s1 * c3 * self.l3 - c1 * s3 * self.l3 - s1 * (p2 + self.l1)
        dx_dp2 = c1
        dx_dtheta3 = -c1 * s3 * self.l3 - s1 * c3 * self.l3
        dy_theta1 = c1 * c3 * self.l3 - s1 * s3 * self.l3 + c1 * (p2 + self.l1)
        dy_dp2 = s1
        dy_dtheta3 = -s1 * s3 * self.l3 + c1 * c3 * self.l3

        # This arm only ever rotates about z and translates in the xy-plane, so wx, wy, vz are always zero
        J = np.zeros((6, 3))
        J[2, :] = [1, 0, 1]  # wz row: dtheta/dq
        J[3, :] = [dx_dtheta1, dx_dp2, dx_dtheta3]  # vx row
        J[4, :] = [dy_theta1, dy_dp2, dy_dtheta3]  # vy row
        return J

    def body_jacobian(self, state):
        """Compute the body Jacobian of the forward kinematics.

        Args:
            state (np.ndarray): joint configuration [theta1, p2, theta3]

        Returns:
            np.ndarray: 6x3 body Jacobian matrix relating joint velocities to the end-effector twist,
            expressed in the end-effector's own (body) frame.
        """
        J = self.analytical_jacobian(state)

        # Analytical jacobian is in LOCAL_WORLD_ALIGNED frame, so velocity component already represents the
        # end-effector velocity in the world frame. To get the body Jacobian, rotate the angular and linear
        # parts into the body frame (no translation coupling needed, since both track the same eef origin).
        # Literally, w_b = R.T @ w_w, v_b = R.T @ v_w
        _, _, theta = self.forward_kinematics(state)
        R = tf_utils.get_R([0, 0, theta])
        R_block = np.zeros((6, 6))
        R_block[0:3, 0:3] = R.T
        R_block[3:6, 3:6] = R.T
        return R_block @ J

    def inverse_kinematics_numeric(self, eef_pose_se2, q_init=None, max_iters=100, tol=1e-6):
        """Numerically solve for joint configuration via Newton's method, using the body Jacobian.

        Args:
            eef_pose_se2: Desired end-effector pose (x, y, theta).
            q_init: Initial guess for joint configuration [theta1, p2, theta3]. Defaults to zeros.
            max_iters: Maximum number of Newton iterations.
            tol: Convergence tolerance on the pose error norm.

        Returns:
            q: Joint configuration [theta1, p2, theta3]
        """
        q = np.zeros(3) if q_init is None else np.array(q_init, dtype=float)

        T_des = tf_utils.get_T([eef_pose_se2[0], eef_pose_se2[1], 0], [0, 0, eef_pose_se2[2]])

        # Newton-Raphson iteration for inverse kinematics
        success = False
        for _ in range(max_iters):
            # Compute the desired end-effector pose in the current end-effector frame
            self.forward_kinematics(q)  # To get current T_base2eef
            T_cur_des = np.linalg.inv(self.T_base2eef) @ T_des

            # matrix logarithm to get the 6D twist representation [wx, wy, wz, vx, vy, vz] of the pose error
            error = tf_utils.twist_matrix_to_twist(logm(T_cur_des))

            if np.linalg.norm(error) < tol:
                success = True
                break

            jacobian = self.body_jacobian(q)
            q = q + np.linalg.pinv(jacobian) @ error

        if not success:
            print('Warning: Inverse kinematics did not converge')
        return success, q

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
        theta1, p2, theta3 = q

        # Plot links
        ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=3)

        # Plot joints, annotated with their joint values
        ax.plot(positions[:-1, 0], positions[:-1, 1], 'ro', markersize=8)
        ax.annotate(
            f'  θ₁={theta1:.2f} rad',
            positions[1],
            fontsize=8,
            color='r',
        )
        ax.annotate(
            f'  p₂={p2:.3f} m',
            positions[2],
            fontsize=8,
            color='r',
        )

        # Plot end effector, annotated with its SE2 pose
        _, _, fk_theta = self.forward_kinematics(q)
        fk_x, fk_y = positions[-1]
        ax.plot(fk_x, fk_y, 'go', markersize=6, label='End Effector')
        arrow_len = self.l3 * 0.3
        ax.arrow(
            fk_x,
            fk_y,
            arrow_len * np.cos(fk_theta),
            arrow_len * np.sin(fk_theta),
            head_width=0.5 * arrow_len,
            head_length=0.5 * arrow_len,
            fc='g',
            ec='g',
            length_includes_head=True,
        )
        ax.annotate(
            f'  θ₃={theta3:.2f} rad\n  ({fk_x:.3f}, {fk_y:.3f}, {np.degrees(fk_theta):.1f}°)',
            (fk_x, fk_y),
            fontsize=8,
            color='g',
        )

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

        return ax


if __name__ == '__main__':
    arm = PlanarRobotArm()

    # Verify the three forward kinematics methods agree
    def poses_close(pose_a, pose_b, atol=1e-5):
        pos_close = np.allclose(pose_a[:2], pose_b[:2], atol=atol)
        theta_diff = np.arctan2(np.sin(pose_a[2] - pose_b[2]), np.cos(pose_a[2] - pose_b[2]))  # wrap to [-pi, pi]
        return pos_close and np.isclose(theta_diff, 0.0, atol=atol)

    all_passed = True
    for _ in range(100):
        q = np.array([np.random.uniform(-np.pi, np.pi), np.random.uniform(0.0, 0.2), np.random.uniform(-np.pi, np.pi)])
        fk_geometric = arm.forward_kinematics(q, method='geometric')
        fk_transformation = arm.forward_kinematics(q, method='transformation')
        fk_exponential = arm.forward_kinematics(q, method='exponential')

        if not (poses_close(fk_geometric, fk_transformation) and poses_close(fk_geometric, fk_exponential)):
            all_passed = False
            print(
                f'FAILED for q={q}: geometric={fk_geometric}, transformation={fk_transformation}, exponential={fk_exponential}'
            )

    print('Forward kinematics methods agree: PASSED' if all_passed else 'Forward kinematics methods agree: FAILED')

    # Verify both inverse kinematics methods recover a target end-effector pose
    ik_all_passed = True
    for _ in range(100):
        q_target = np.array(
            [np.random.uniform(-np.pi, np.pi), np.random.uniform(0.0, 0.2), np.random.uniform(-np.pi, np.pi)]
        )
        target_pose = arm.forward_kinematics(q_target, method='transformation')

        q_analytic = arm.inverse_kinematics_analytic(target_pose)
        pose_analytic = arm.forward_kinematics(q_analytic, method='transformation')

        q_init = q_target + np.random.uniform(-0.1, 0.1, size=3)
        success, q_numeric = arm.inverse_kinematics_numeric(target_pose, q_init=q_init)
        pose_numeric = arm.forward_kinematics(q_numeric, method='transformation')

        if not poses_close(target_pose, pose_analytic):
            ik_all_passed = False
            print(f'ANALYTIC IK FAILED for target={target_pose}: recovered pose={pose_analytic}')
        if not success or not poses_close(target_pose, pose_numeric):
            ik_all_passed = False
            print(f'NUMERIC IK FAILED for target={target_pose}: recovered pose={pose_numeric}, success={success}')

    print('Inverse kinematics methods agree: PASSED' if ik_all_passed else 'Inverse kinematics methods agree: FAILED')

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
