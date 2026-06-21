import numpy as np
from typing_extensions import override

from robotics_algorithm.robot.robot import Robot


class Cartpole(Robot):
    def __init__(self, dt=0.01):
        super().__init__(dt)

        # self.kinematics_integrator = 'euler'
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
        effective_pole_inertia = self.length * (4.0 / 3.0 - self.masspole * np.square(costheta) / self.total_mass)
        thetaacc = (self.gravity * sintheta - costheta * temp) / effective_pole_inertia
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Forward Euler integration
        x = x + self.dt * x_dot
        x_dot = x_dot + self.dt * xacc
        theta = theta + self.dt * theta_dot
        theta_dot = theta_dot + self.dt * thetaacc

        return np.array([x, x_dot, theta, theta_dot])

    @override
    def state_transition_jacobian(self, state, action):
        # linearize discrete-time dynamics around state -> x_new = A x + B u
        action = action.item()
        x, x_dot, theta, theta_dot = state

        # Get some constant first
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (action + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        effective_pole_inertia = self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        thetaacc = (self.gravity * sintheta - costheta * temp) / effective_pole_inertia

        # partial derivatives of temp
        dtemp_dtheta = self.polemass_length * theta_dot**2 * costheta / self.total_mass
        dtemp_dthetadot = 2.0 * self.polemass_length * theta_dot * sintheta / self.total_mass
        dtemp_du = 1.0 / self.total_mass

        # partial derivative of effective_pole_inertia
        dinertia_dtheta = self.length * (2.0 * self.masspole * costheta * sintheta / self.total_mass)

        # partials of theta acceleration
        # quotient rule
        thetaacc_dtheta = (
            (self.gravity * costheta + sintheta * temp - costheta * dtemp_dtheta) * effective_pole_inertia
            - (self.gravity * sintheta - costheta * temp) * dinertia_dtheta
        ) / (effective_pole_inertia**2)
        thetaacc_dthetadot = -costheta * dtemp_dthetadot / effective_pole_inertia
        thetaacc_du = -costheta * dtemp_du / effective_pole_inertia

        # partials of x acceleration
        xacc_dtheta = (
            dtemp_dtheta - self.polemass_length * (-sintheta * thetaacc + costheta * thetaacc_dtheta) / self.total_mass
        )
        xacc_dthetadot = dtemp_dthetadot - self.polemass_length * costheta * thetaacc_dthetadot / self.total_mass
        xacc_du = dtemp_du - self.polemass_length * costheta * thetaacc_du / self.total_mass

        # forward euler integration
        A = np.array(
            [
                [1.0, self.dt, 0.0, 0.0],
                [0.0, 1.0, self.dt * xacc_dtheta, self.dt * xacc_dthetadot],
                [0.0, 0.0, 1.0, self.dt],
                [0.0, 0.0, self.dt * thetaacc_dtheta, 1.0 + self.dt * thetaacc_dthetadot],
            ]
        )
        B = np.array([0.0, self.dt * xacc_du, 0.0, self.dt * thetaacc_du]).reshape(4, 1)

        return A, B
