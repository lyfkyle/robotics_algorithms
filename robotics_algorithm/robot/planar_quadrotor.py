import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from typing_extensions import override

from robotics_algorithm.robot.robot import Robot


class PlanarQuadrotor(Robot):
    def __init__(self, dt=0.01):
        """Planar Quadrotor.

        https://arxiv.org/pdf/2106.15134

        State: [x, z, theta, x_dot, z_dot, theta_dot]
        Action: [net_thrust, net_moment, g]  gravity acceleration needs to be appended to simplify state space equation

        """
        super().__init__(dt)

        # Constants
        self.g = 9.81
        self.m = 1.0
        self.L = 0.4   # length of the , in m
        self.I = self.m * self.L * self.L / 12.0  # Using moment of inertia of a thin rod

    @override
    def control(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        x, z, theta, x_vel, z_vel, theta_vel = state

        left_thrust, right_thrust = action
        net_thrust = left_thrust + right_thrust
        net_moment = self.L / 2.0 * (right_thrust - left_thrust)

        x_accel = -net_thrust * math.sin(theta) / self.m
        z_accel = net_thrust * math.cos(theta) / self.m - self.g
        theta_accel = net_moment / self.I

        # Forward euler simulation
        new_x = x + x_vel * self.dt
        new_y = z + z_vel * self.dt
        new_theta = theta + theta_vel * self.dt
        new_x_vel = x_vel + x_accel * self.dt
        new_z_vel = z_vel + z_accel * self.dt
        new_theta_vel = theta_vel + theta_accel * self.dt

        return np.array([new_x, new_y, new_theta, new_x_vel, new_z_vel, new_theta_vel])

    # def control_thrust(self, state, left_thrust, right_thrust):
        # net_thrust = left_thrust + right_thrust
        # net_moment = self.L / 2.0 * (right_thrust - left_thrust)

        # return self.control(state, np.array([net_thrust, net_moment]))

    @override
    def linearize_state_transition(self, state, action):
        # linearize dynamics around state in discrete time -> x_new = Ax + Bu

        x, z, theta, x_vel, z_vel, theta_vel = state
        u1, u2 = action

        # discrete-time case: x_t+1 = Ax + Bu
        A = np.array([
            [1, 0, 0, self.dt, 0, 0],
            [0, 1, 0, 0, self.dt, 0],
            [0, 0, 1, 0, 0, self.dt],
            [0, 0, -u1 * math.cos(theta) / self.m * self.dt, 1, 0, 0],
            [0, 0, u1 * math.sin(theta) / self.m * self.dt, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        B = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [-math.sin(theta) / self.m * self.dt, -math.sin(theta) / self.m * self.dt],
            [math.cos(theta) / self.m * self.dt, math.cos(theta) / self.m * self.dt],
            [-self.L / (2 * self.I), self.L / (2 * self.I)],
        ])
        return A, B

if __name__ == "__main__":
    env = PlanarQuadrotor()

    total_time = 10

    # Initial state [x, z, theta, x_dot, z_dot, theta_dot]
    state = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    # Control inputs: constant thrusts
    u1 = 5.0              # Thrust from left rotor (N)
    u2 = 5.0              # Thrust from right rotor (N)

    # Simulation variables
    trajectory_x = []
    trajectory_z = []

    # Update function for integration
    def update_state(state, u1, u2, dt):
        """Update the state using numerical integration (Euler method)."""
        new_state = env.control(state, np.array([u1, u2]))
        return new_state

    # Set up the figure and animation
    fig, ax = plt.subplots(figsize=(8, 6))

    # Limits for animation
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1, 3)

    # Quadrotor body and thrusters
    body, = ax.plot([], [], 'o-', lw=4, markersize=10, label='Quadrotor Body')  # Quadrotor body
    trajectory, = ax.plot([], [], 'b--', lw=1, label="Trajectory")             # Trajectory of the center of mass

    # Text for simulation time
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    # Initialize the quadrotor drawing components
    def init():
        body.set_data([], [])
        trajectory.set_data([], [])
        time_text.set_text('')
        return body, trajectory, time_text

    # Animation function
    def animate(i):
        global state, trajectory_x, trajectory_z

        # Update the state using Euler integration
        state = update_state(state, u1, u2, env.dt)

        # Extract positions and angles
        x, z, theta, _, _, _ = state

        # Compute quadrotor geometry
        rotor1_x = x - env.L * np.cos(theta)
        rotor1_z = z - env.L * np.sin(theta)
        rotor2_x = x + env.L * np.cos(theta)
        rotor2_z = z + env.L * np.sin(theta)

        # Update trajectory
        trajectory_x.append(x)
        trajectory_z.append(z)

        # Update the quadrotor body and trajectory data
        body.set_data([rotor1_x, rotor2_x], [rotor1_z, rotor2_z])
        trajectory.set_data(trajectory_x, trajectory_z)

        # Update simulation time
        time_text.set_text(f"Time: {i*env.dt:.2f}s")

        return body, trajectory, time_text

    # Run the animation
    ani = FuncAnimation(fig, animate, frames=int(total_time / env.dt), interval=env.dt * 1000, init_func=init, blit=True)
    plt.legend()
    plt.xlabel("X Position (m)")
    plt.ylabel("Z Position (m)")
    plt.title("Planar Quadrotor Simulation")
    plt.grid()
    plt.show()