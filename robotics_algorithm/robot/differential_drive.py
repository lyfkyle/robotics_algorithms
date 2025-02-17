import matplotlib.pyplot as plt
import numpy as np
from typing_extensions import override

from robotics_algorithm.robot.robot import Robot
from robotics_algorithm.utils import math_utils


class DiffDrive(Robot):
    def __init__(self, wheel_radius: float, wheel_dist: float, dt=0.05):
        super().__init__(dt)

        # Robot parameters
        self.wheel_radius = 0.05  # meters
        self.wheel_dist = 0.2  # meters

    def control_wheel_speed(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        """_summary_

        Args:
            state (np.ndarray): [x, y, theta] robot's current state
            control (np.ndarray): [v_l, v_r] left and right wheel velocities in radians
            dt (float): time step

        Returns:
            new state [x, y, theta]
        """
        v_l, v_r = control
        lin_vel = self.wheel_radius * (v_r + v_l) / 2.0
        ang_vel = self.wheel_radius * (v_r - v_l) / self.wheel_dist

        return self.control(state, lin_vel, ang_vel, dt)

    @override
    def control(self, state: np.ndarray, action: np.ndarray, dt: float) -> np.ndarray:
        """
        Update the robot state based on the differential drive kinematics.

        Args:
            state (np.ndarray): [x, y, theta] robot's current state.
            action (np.ndarray): [lin_vel, ang_vel]
            dt (float): time step

        Returns:
            new state [x, y, theta]
        """
        x, y, theta = state
        lin_vel, ang_vel = action

        t = 0
        while t < dt:
            # Update the state
            x_new = x + lin_vel * np.cos(theta) * self.dt
            y_new = y + lin_vel * np.sin(theta) * self.dt
            theta_new = theta + ang_vel * self.dt
            theta_new = math_utils.normalize_angle(theta_new)

            t += self.dt
            x, y, theta = x_new, y_new, theta_new

        return np.array([x_new, y_new, theta_new])

    @override
    def linearize_state_transition(self, state, action):
        # linearize dynamics around state in discrete time -> x_new = Ax + Bu

        x, y, theta = state
        lin_vel, ang_vel = action
        A = np.array(
            [
                [1, 0, -lin_vel * np.sin(theta) * self.dt],
                [0, 1, lin_vel * np.cos(theta) * self.dt],
                [0, 0, 1]
            ]
        )
        B = np.array([[np.cos(theta) * self.dt, 0], [np.sin(theta) * self.dt, 0], [0, self.dt]])

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

        num_sub_steps = int(dt / diff_drive_system.dt)
        for _ in range(num_sub_steps):
            state = diff_drive_system.control_wheel_speed(state, control_inputs, dt=diff_drive_system.dt)
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
