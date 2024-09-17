from typing_extensions import override

import matplotlib.pyplot as plt
import random
import numpy as np
import math
import numpy as np
from numpy.random import randn

from robotics_algorithm.robot.double_integrator import DoubleIntegrator
from robotics_algorithm.env.base_env import DeterministicEnv


class DoubleIntegratorEnv(DeterministicEnv):
    """A DoubleIntegrator robot is tasked to stop at the goal state which is at zero position  with zero velocity.

    State: [pos, vel]
    Action: [acceleration]

    Continuous state space.
    Continuous action space.
    Deterministic transition.
    Fully observable.

    Linear state transition function.
    Quadratic cost.
    """

    def __init__(self, size=10, continuous_time=True):
        super().__init__()

        self.size = size

        self.robot_model = DoubleIntegrator(continuous_time=continuous_time)
        self.state_space = [[-self.size / 2, float("-inf")], [self.size / 2, float("inf")]]  # pos and vel
        self.action_space = [float("-inf"), float("inf")]  # accel

        # declare linear state transition
        # x_dot = Ax + Bu
        self.state_transition_type = "linear"
        self.A = self.robot_model.A
        self.B = self.robot_model.B

        # declare quadratic cost
        # L = x.T @ Q @ x + u.T @ R @ u
        self.reward_func_type = "quadratic"
        self.Q = np.array([[1, 0], [0, 1]])
        self.R = np.array([[1]])

        self.path = None

    def reset(self):
        self.start_state = [random.uniform(self.state_space[0][0], self.state_space[1][0]), 0]
        self.goal_state = [0, 0]  # fixed
        self.cur_state = self.start_state

    @override
    def state_transition_func(self, state: list, action: list) -> tuple[list, float, bool, bool, dict]:
        new_state = self.robot_model.control(state, action, dt=0.01)

        # Check bounds
        if (
            new_state[0] <= self.state_space[0][0]
            or new_state[0] >= self.state_space[1][0]
            or new_state[1] <= self.state_space[0][1]
            or new_state[1] >= self.state_space[1][1]
        ):
            return state, -100, True, False, {"success": False}

        # Check goal state reached for termination
        term = False
        if np.allclose(np.array(new_state), np.array(self.goal_state), atol=1e-4):
            term = True

        self.cur_state = new_state

        return new_state, self.reward_func(state, action), term, False, {}

    @override
    def reward_func(self, state: list, action: list | None = None) -> float:
        # quadratic reward func
        state = np.array(state)
        action = np.array(action)

        cost = state.T @ self.Q @ state + action.T @ self.R @ action
        return -cost

    def render(self):
        fig, ax = plt.subplots(2, 1, dpi=100)
        ax[0].plot(0, self.start_state[0], "o")
        ax[1].plot(0, self.start_state[1], "o")

        if self.path:
            for i, state in enumerate(self.path):
                ax[0].plot(i, state[0], "o")
                ax[1].plot(i, state[1], "o")

        ax[0].set_ylim(-self.size / 2, self.size / 2)
        plt.show()

    def add_path(self, path):
        self.path = path


# TODO
class OneDLocalizationContinuous(DeterministicEnv):
    def __init__(self, initial_position=0, initial_velocity=1, measurement_var=0.1, process_var=0.1, dt=1):
        """
        measurement_variance - variance in measurement m^2
        process_variance - variance in process (m/s)^2
        """
        self.position = initial_position
        self.velocity = initial_velocity
        self.dt = dt
        self.measurement_noise = math.sqrt(measurement_var)
        self.process_noise = math.sqrt(process_var)

        self.state = np.array([[self.position], [self.velocity]])
        self.R = np.array([[process_var, 0], [0, process_var]], dtype=np.float32)

        self.H = np.array([[1, 0]], dtype=np.float32)
        self.Q = np.array([[measurement_var]], dtype=np.float32)

    def control(self, accel):
        """
        Compute new position and velocity of agent assume accel is applied for dt time
        xt+1 = Axt + But
        """
        # compute new position based on acceleration. Add in some process noise
        new_state = (
            self.A @ self.state
            + np.dot(self.B, accel)
            + np.array([[randn() * self.process_noise], [randn() * self.process_noise]])
        )
        self.state = new_state

    def sense(self):
        """
        Measure agent's position only.
        """

        # simulate measuring the position with noise
        meas = self.H @ self.state + np.array([[randn() * self.measurement_noise]])
        return meas

    def control_and_sense(self, accel):
        self.control(accel)
        return self.sense()
