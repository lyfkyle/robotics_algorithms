import numpy as np


class Arm(object):
    def __init__(self, num_links=2, link_len=0.2):
        self.num_links = num_links
        self.link_len = link_len

    def forward_kinematics(self, joint_state: list) -> list:
        pass

    def inverse_kinematics(self, eef_state: list) -> list:
        pass

    def forward_vel_kinematics(self, joint_vel: list) -> list:
        pass

    def inverse_vel_kinematics(self, eef_vel: list) -> list:
        pass


class ThreeLinkArm(Arm):
    """
    https://www.researchgate.net/figure/Configuration-of-spherical-wrist_fig3_228412878
    """

    def __init__(self, link_len=0.2):
        super().__init__(num_links=3, link_len=link_len)

    def forward_kinematics(self, joint_state: list, return_joint_pose=False) -> list:
        theta1, theta2, theta3 = joint_state
        c1 = np.cos(theta1)
        s1 = np.sin(theta1)
        T_01 = np.array([[s1, 0, c1, 0], [-c1, 0, s1, 0], [0, -1, 0, self.link_len], [0, 0, 0, 1]])

        c2 = np.cos(theta2)
        s2 = np.sin(theta2)
        T_12 = np.array(
            [[0, -s2, -c2, -c2 * self.link_len], [0, c2, -s2, -s2 * self.link_len], [1, 0, 0, 0], [0, 0, 0, 1]]
        )

        c3 = np.cos(theta3)
        s3 = np.sin(theta3)
        T_23 = np.array([[s3, c3, 0, 0], [-c3, s3, 0, 0], [0, 0, 1, self.link_len], [0, 0, 0, 1]])

        T_02 = T_01 @ T_12
        T_03 = T_02 @ T_23

        if not return_joint_pose:
            return T_03
        else:
            return [np.eye(4), T_01, T_02, T_03]


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    # Create figure for plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Define lengths of the links
    L1 = 1  # Length from base to first joint
    L2 = 1  # Length from first joint to second joint
    L3 = 1  # Length of wrist (end-effector)
    gripper_length = 0.2  # Length of the gripper fingers
    gripper_width = 0.1  # Distance between the gripper fingers

    # Function to add a gripper at the end-effector position
    def draw_gripper(end_effector_T):
        eef_pos = end_effector_T[:3, 3].tolist()

        finger1_start_T = end_effector_T @ np.array(
            [[1, 0, 0, 0], [0, 1, 0, gripper_width], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        finger1_end_T = finger1_start_T @ np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, gripper_length], [0, 0, 0, 1]]
        )

        finger2_start_T = end_effector_T @ np.array(
            [[1, 0, 0, 0], [0, 1, 0, -gripper_width], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        finger2_end_T = finger2_start_T @ np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, gripper_length], [0, 0, 0, 1]]
        )

        finger1_start = finger1_start_T[:3, 3].tolist()
        finger1_end = finger1_end_T[:3, 3].tolist()
        finger2_start = finger2_start_T[:3, 3].tolist()
        finger2_end = finger2_end_T[:3, 3].tolist()

        # Plot gripper fingers
        ax.plot(
            [eef_pos[0], finger1_start[0]],
            [eef_pos[1], finger1_start[1]],
            [eef_pos[2], finger1_start[2]],
            "r-",
            lw=3,
        )
        ax.plot(
            [finger1_start[0], finger1_end[0]],
            [finger1_start[1], finger1_end[1]],
            [finger1_start[2], finger1_end[2]],
            "r-",
            lw=3,
        )
        ax.plot(
            [eef_pos[0], finger2_start[0]],
            [eef_pos[1], finger2_start[1]],
            [eef_pos[2], finger2_start[2]],
            "r-",
            lw=3,
        )
        ax.plot(
            [finger2_start[0], finger2_end[0]],
            [finger2_start[1], finger2_end[1]],
            [finger2_start[2], finger2_end[2]],
            "r-",
            lw=3,
        )

    robot = ThreeLinkArm(link_len=L1)

    # Plotting the robot configuration
    def update_robot(theta1, theta2, theta3):
        ax.cla()  # Clear the previous plot

        # Set the axis limits
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        ax.set_zlim([0, 3])

        # Get the joint positions
        # points = forward_kinematics(theta1, theta2, theta3)

        joint_poses = robot.forward_kinematics([theta1, theta2, theta3], return_joint_pose=True)
        # print(points[-1], eef_pos)
        joint_pos = [p[:3, 3].tolist() for p in joint_poses]
        print(joint_pos)

        # Extract the x, y, z coordinates of each joint
        x_vals = [p[0] for p in joint_pos]
        y_vals = [p[1] for p in joint_pos]
        z_vals = [p[2] for p in joint_pos]

        # Plot the robot as a line connecting joints
        ax.plot(x_vals, y_vals, z_vals, "-o", markersize=8, lw=2, color="b")

        # draw the gripper
        draw_gripper(joint_poses[-1])

        # Set labels and title
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(
            f"Spherical Wrist Robot\nTheta1={np.degrees(theta1):.1f}, Theta2={np.degrees(theta2):.1f}, Theta3={np.degrees(theta3):.1f}"
        )

    # Create an animation of the spherical wrist robot
    def animate(i):
        theta1 = np.radians(i)
        # theta1 = 0
        theta2 = np.radians(i / 2)
        # theta2 = 0
        theta3 = np.radians(i / 3)
        # theta3 = 0
        update_robot(theta1, theta2, theta3)

    # Create animation
    ani = FuncAnimation(fig, animate, frames=np.arange(0, 360, 2), interval=50)

    # Display the plot
    plt.show()
