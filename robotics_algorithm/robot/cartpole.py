import numpy as np
from typing_extensions import override

from robotics_algorithm.robot.robot import Robot


class Cartpole(Robot):
    def __init__(self, dt=0.02):
        super().__init__(dt)

        self.kinematics_integrator = "euler"
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0

    @override
    def control(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        action = action.item()

        x, x_dot, theta, theta_dot = state

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (action + self.polemass_length * np.square(theta_dot) * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * np.square(costheta) / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.dt * x_dot
            x_dot = x_dot + self.dt * xacc
            theta = theta + self.dt * theta_dot
            theta_dot = theta_dot + self.dt * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.dt * xacc
            x = x + self.dt * x_dot
            theta_dot = theta_dot + self.dt * thetaacc
            theta = theta + self.dt * theta_dot

        return np.array([x, x_dot, theta, theta_dot])
