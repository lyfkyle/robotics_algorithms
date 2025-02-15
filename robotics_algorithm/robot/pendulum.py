import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


class Pendulum():
    def __init__(self, dt=0.01):
        # Constants
        self.g = 9.81  # acceleration due to gravity, in m/s^2
        self.L = 1.0   # length of the pendulum, in m
        self.dt = dt

    def control(self, state: np.ndarray, action: np.ndarray):
        theta, theta_dot = state
        theta += np.pi

        theta_dot_dot = -self.g / self.L * math.sin(theta) + action.item() / self.L / self.L

        # Forward euler
        new_theta = theta + self.dt * theta_dot - np.pi
        new_theta_dot = theta_dot + self.dt * theta_dot_dot

        return np.array([new_theta, new_theta_dot])

    def linearized_dynamics(self, state, action):
        # discrete-time case: x_t+1 = Ax + Bu
        theta = state[0]
        theta += np.pi
        A = np.array([[1, self.dt], [-self.g / self.L * math.cos(theta) * self.dt, 1]])
        B = np.array([0, 1 / (self.L ** 2) * self.dt]).reshape(2, 1)
        return A, B


if __name__ == "__main__":
    env = Pendulum()

    # Time vector
    t = np.arange(0, 10, env.dt)  # simulate for 10 seconds

    # State vector [theta, omega]
    x = np.zeros((len(t), 2))
    x[0] = [0, 0]

    # Control input (constant or time-varying)
    u = np.ones(len(t)) * 20 # example: no control input
    # u = np.sin(t)  # example: sinusoidal control input

    for i in range(1, len(t)):
        x[i] = env.control(x[i-1], u[i-1])

    # Extract theta and omega for plotting
    theta = x[:, 0]
    omega = x[:, 1]

    # Create an animation of the pendulum
    fig, ax = plt.subplots()
    ax.set_xlim(-env.L-0.1, env.L+0.1)
    ax.set_ylim(-env.L-0.1, env.L+0.1)
    ax.set_aspect('equal')
    line, = ax.plot([], [], 'o-', lw=2)

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        x_pendulum = env.L * np.sin(theta[frame] + np.pi)
        y_pendulum = -env.L * np.cos(theta[frame] + np.pi)
        line.set_data([0, x_pendulum], [0, y_pendulum])
        return line,

    ani = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=env.dt*1000)
    plt.show()