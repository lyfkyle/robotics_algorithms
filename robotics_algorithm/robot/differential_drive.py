import numpy as np
import matplotlib.pyplot as plt


class DiffDrive:
    def __init__(self, wheel_radius: float, wheel_dist: float):
        # Robot parameters
        self.wheel_radius = 0.05  # meters
        self.wheel_dist = 0.2  # meters

    def control_wheel_speed(self, state: np.ndarray, control: list, dt: float) -> np.ndarray:
        """_summary_

        Args:
            state (np.ndarray): [x, y, theta] robot's current state
            control (list): [v_l, v_r] left and right wheel velocities in radians
            dt (float): time step

        Returns:
            new state [x, y, theta]
        """
        v_l, v_r = control
        lin_vel = self.wheel_radius * (v_r + v_l) / 2.0
        ang_vel = self.wheel_radius * (v_r - v_l) / self.wheel_dist

        return self.control_velocity(state, lin_vel, ang_vel, dt)

    def control_velocity(self, state: np.ndarray, lin_vel: float, ang_vel: float, dt: float) -> np.ndarray:
        """
        Update the robot state based on the differential drive kinematics.

        Args:
            state (np.ndarray): [x, y, theta] robot's current state.
            lin_vel (float): linear velocity
            ang_vel (float): angular velocity
            dt (float): time step

        Returns:
            new state [x, y, theta]
        """
        x, y, theta = state

        # Update the state
        x_new = x + lin_vel * np.cos(theta) * dt
        y_new = y + lin_vel * np.sin(theta) * dt
        theta_new = theta + ang_vel * dt

        return np.array([x_new, y_new, theta_new])


if __name__ == "__main__":
    # Simulation parameters
    dt = 0.1  # time step
    total_time = 10.0  # total simulation time
    num_steps = int(total_time / dt)

    # Initial state [x, y, theta]
    state = np.array([0.0, 0.0, 0.0])

    # Control inputs (left and right wheel velocities)
    control_inputs = [0.2, 0.5]  # constant velocities

    # Record state history for plotting
    state_history = [state]

    diff_drive_system = DiffDrive(wheel_radius=0.05, wheel_dist=0.2)

    # Run simulation
    for _ in range(num_steps):
        state = diff_drive_system.control_wheel_speed(state, control_inputs, dt)
        state_history.append(state)

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
