import math

import matplotlib.pyplot as plt
import numpy as np
from typing_extensions import override

from robotics_algorithm.env.base_env import ContinuousSpace, DeterministicEnv, FullyObservableEnv, FunctionType
from robotics_algorithm.robot.planar_quadrotor import PlanarQuadrotor


class PlanarQuadrotorHoverEnv(DeterministicEnv, FullyObservableEnv):
    """A planar quadrotor hover must hover at a given height

    State: [x, z, theta, x_dot, z_dot, theta_dot]
    Action: [left thrust, right thrust]

    """

    def __init__(self, hover_pos=0.5, hover_height=1.0, quadratic_reward=True, term_if_constraints_violated=True):
        """Constructor

        Args:
            hover_pos (float, optional): desired hover position. Defaults to 1.0.
            hover_height (float, optional): desired hover height. Defaults to 1.0.
            quadratic_reward (bool, optional): whether to use quadratic reward. Defaults to True.
            term_if_constraints_violated (bool, optional): whether the environment should terminate if constrains are violated. Defaults to True.
        """
        super().__init__()

        self.state_transition_func_type = FunctionType.LINEAR.value

        # state and action space
        self.state_space = ContinuousSpace(low=np.full(6, -np.inf), high=np.full(6, np.inf))
        self.action_space = ContinuousSpace(low=[0, 0], high=[10, 10])  # torque

        # reward
        self.quadratic_reward = quadratic_reward
        if quadratic_reward:
            self.reward_func_type = FunctionType.QUADRATIC.value
            self.Q = np.diag([1, 1, 1, 1, 1, 1])
            self.R = np.diag([1, 1])

        # robot model
        self.robot_model = PlanarQuadrotor()
        self.action_dt = self.robot_model.dt
        self._ref_state = np.array([hover_pos, hover_height, 0, 0, 0, 0])  # hover at 1 meter height

        self.theta_threshold_radians = math.pi / 2
        self.max_steps = 500
        self.term_if_constraints_violated = term_if_constraints_violated

        # others
        self._fig_created = False

    @override
    def reset(self):
        self.cur_state = np.zeros(6)  # On the ground stationary
        self.goal_state = self._ref_state  # hover at 1 meter
        # ! In order to hover, net thrust must equal to gravitational force.
        self.goal_action = np.array(
            [0.5 * self.robot_model.g * self.robot_model.m, 0.5 * self.robot_model.g * self.robot_model.m]
        )
        self.step_cnt = 0

        return self.sample_observation(self.cur_state), {}

    @override
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        new_state, reward, term, trunc, info = super().step(action)

        if not self._is_action_valid(action):
            print(f'action {action} is not within valid range!')
            if self.term_if_constraints_violated:
                term = True

        return new_state, reward, term, trunc, info

    @override
    def state_transition_func(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        # compute next state
        new_state = self.robot_model.control(state, action)
        return new_state

    @override
    def reward_func(self, state: np.ndarray, action: np.ndarray = None, new_state: np.ndarray = None) -> float:
        terminated = self.is_state_terminal(new_state)

        if not self._is_action_valid(action):
            if self.term_if_constraints_violated:
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
    def is_state_terminal(self, state):
        _, z, theta, _, _, _ = state

        terminated = False
        # quadcopter is too tilted
        if theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians:
            terminated = True
        # falls below floor
        if z < -1:
            terminated = True

        return terminated

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
