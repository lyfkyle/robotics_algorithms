import math

import matplotlib.pyplot as plt
import numpy as np
from typing_extensions import override

from robotics_algorithm.env.base_env import ContinuousSpace, DeterministicEnv, FullyObservableEnv, FunctionType
from robotics_algorithm.robot.planar_quadrotor import PlanarQuadrotor


class PlanarQuadrotorHoverEnv(DeterministicEnv, FullyObservableEnv):
    """Inverted pendulum environment.

    State: [theta, theta_dot]
    Action: [torque]

    """

    def __init__(self, hover_height=1.0, quadratic_reward=True, term_if_constraints_violated=False):
        super().__init__()

        self.state_transition_func_type = FunctionType.LINEAR.value

        # state and action space
        self.state_space = ContinuousSpace(low=np.full(6, -np.inf), high=np.full(6, np.inf))
        self.action_space = ContinuousSpace(low=[0, 0], high=[10, 10])  # torque

        # reward
        self.quadratic_reward = quadratic_reward
        if quadratic_reward:
            self.reward_func_type = FunctionType.QUADRATIC.value
            self.Q = np.diag([100, 100, 100, 1, 1, 1])
            self.R = np.diag([1, 1])

        # robot model
        self.robot_model = PlanarQuadrotor()
        self.action_dt = self.robot_model.dt
        self._ref_state = np.array([0, hover_height, 0, 0, 0, 0])  # hover at 1 meter height

        self.theta_threshold_radians = math.pi / 2
        self.max_steps = 1000
        self.term_if_constraints_violated = term_if_constraints_violated

        # others
        self._fig_created = False

    @override
    def reset(self):
        self.cur_state = np.zeros(6)  # On the ground stationary
        self.goal_state = self._ref_state  # hover at 1 meter
        # ! In order to hover, net thrust must equal to gravitational force.
        self.goal_action = np.array([0.5 * self.robot_model.g * self.robot_model.m, 0.5 * self.robot_model.g * self.robot_model.m])

        self.step_cnt = 0

        return self.sample_observation(self.cur_state), {}

    @override
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        new_state, reward, term, trunc, info = super().step(action)

        self.step_cnt += 1
        trunc = self.step_cnt > self.max_steps  # set truncated flag

        return new_state, reward, term, trunc, info

    @override
    def state_transition_func(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        # compute next state
        new_state = self.robot_model.control(state, action)
        return new_state

    @override
    def reward_func(self, state: np.ndarray, action: np.ndarray = None, new_state: np.ndarray = None) -> float:
        _, z, theta, _, _, _ = new_state

        terminated = False
        # quadcopter is too tilted
        if theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians:
            terminated = True
        # falls below floor
        if z < -1:
            terminated = True

        if not terminated:
            state_error = new_state - self.goal_state
            action_error = action - self.goal_action
            if self.quadratic_reward:
                reward = -(state_error.T @ self.Q @ state_error + action_error.T @ self.R @ action_error)
            else:
                reward = 1.0
        else:
            reward = -1.0

        return reward

    @override
    def get_state_transition_info(self, state, action, new_state):
        _, z, theta, _, _, _ = new_state

        terminated = False
        # quadcopter is too tilted
        if theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians:
            terminated = True
        # falls below floor
        if z < -1:
            terminated = True

        if not self._is_action_valid(action):
            print(f'action {action} is not within valid range!')
            if self.term_if_constraints_violated:
                terminated = True

        return terminated, False, {}

    @override
    def linearize_state_transition(self, state, action):
        return self.robot_model.linearize_state_transition(state, action)

    @override
    def render(self):
        if not self._fig_created:
            # Create an animation of the pendulum
            plt.ion()
            plt.figure()

            self._fig_created = True

        plt.clf()
        # Extract positions and angles
        x, z, theta, _, _, _ = self.cur_state

        # Compute quadrotor geometry
        rotor1_x = x - self.robot_model.L * np.cos(theta)
        rotor1_z = z - self.robot_model.L * np.sin(theta)
        rotor2_x = x + self.robot_model.L * np.cos(theta)
        rotor2_z = z + self.robot_model.L * np.sin(theta)

        # Quadrotor body and thrusters
        (body,) = plt.plot(
            [rotor1_x, rotor2_x], [rotor1_z, rotor2_z], 'o-', lw=4, markersize=10, label='Quadrotor Body'
        )  # Quadrotor body

        # Update simulation time
        plt.text(0.02, 0.95, f'Time: {self.step_cnt * self.action_dt}s')

        # Run the animation
        plt.legend()
        plt.xlabel('X Position (m)')
        plt.ylabel('Z Position (m)')
        plt.title('Planar Quadrotor Simulation')
        # Limits for animation
        plt.xlim(-2, 2)
        plt.ylim(-1, 3)
        plt.grid()
        plt.pause(0.01)
