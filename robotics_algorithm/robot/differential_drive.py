import numpy as np
import matplotlib.pyplot as plt

from robotics_algorithm.utils import math_utils


class DiffDrive:
    def __init__(self, wheel_radius: float, wheel_dist: float):
        # Robot parameters
        self.wheel_radius = 0.05  # meters
        self.wheel_dist = 0.2  # meters
        self.time_res = 0.05

    def control_wheel_speed(self, state: list, control: list, dt: float) -> list:
        """_summary_

        Args:
            state (list): [x, y, theta] robot's current state
            control (list): [v_l, v_r] left and right wheel velocities in radians
            dt (float): time step

        Returns:
            new state [x, y, theta]
        """
        v_l, v_r = control
        lin_vel = self.wheel_radius * (v_r + v_l) / 2.0
        ang_vel = self.wheel_radius * (v_r - v_l) / self.wheel_dist

        return self.control_velocity(state, lin_vel, ang_vel, dt)

    def control_velocity(self, state: list, lin_vel: float, ang_vel: float, dt: float) -> list:
        """
        Update the robot state based on the differential drive kinematics.

        Args:
            state (list): [x, y, theta] robot's current state.
            lin_vel (float): linear velocity
            ang_vel (float): angular velocity
            dt (float): time step

        Returns:
            new state [x, y, theta]
        """
        x, y, theta = state

        t = 0
        while t < dt:
            # Update the state
            x_new = x + lin_vel * np.cos(theta) * self.time_res
            y_new = y + lin_vel * np.sin(theta) * self.time_res
            theta_new = theta + ang_vel * self.time_res
            theta_new = math_utils.normalize_angle(theta_new)

            t += self.time_res
            x, y, theta = x_new, y_new, theta_new

        return [x_new.item(), y_new.item(), theta_new]

    def linearize_dynamics(self, state):
        # linearize dynamics around state -> x_new = Ax + Bu
        x, y, theta = state
        A = np.eye(3)
        B = np.array([[np.cos(theta) * self.time_res, 0], [np.sin(theta) * self.time_res, 0], [0, self.time_res]])

        return A, B

if __name__ == "__main__":
    # Simulation parameters
    dt = 1.0  # time step
    total_time = 10.0  # total simulation time
    num_steps = int(total_time / dt)

    # Initial state [x, y, theta]
    state = [0.0, 0.0, 0.0]

    # Control inputs (left and right wheel velocities)
    control_inputs = [0.2, 0.5]  # constant velocities

    # Record state history for plotting
    state_history = [state]

    diff_drive_system = DiffDrive(wheel_radius=0.05, wheel_dist=0.2)

    # Run simulation
    for _ in range(num_steps):
        state1 = diff_drive_system.control_wheel_speed(state, control_inputs, dt)

        num_sub_steps = int(dt / diff_drive_system.time_res)
        for _ in range(num_sub_steps):
            state = diff_drive_system.control_wheel_speed(state, control_inputs, dt=diff_drive_system.time_res)
            state_history.append(state)

        assert np.allclose(np.array(state1), np.array(state))

    # Convert state history to numpy array for easier indexing
    state_history = np.array(state_history)
    print(state_history)

    # Plot results
    plt.figure()
    plt.plot(state_history[:, 0], state_history[:, 1], label="Robot Path")
    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.title("Differential Drive Robot Path")
    plt.legend()
    plt.axis("equal")
    plt.grid()
    plt.show()
