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
        self.action_space = ContinuousSpace(low=np.array([0, 0]), high=np.array([10, 10]))  # torque

        # reward
        self._quadratic_reward = quadratic_reward
        if quadratic_reward:
            self.reward_func_type = FunctionType.QUADRATIC.value
            # x, z, theta, xdot, zdot, thetadot
            self.Q = np.diag([20, 50, 100, 5, 20, 10])
            # left thrust, right thrust
            self.R = np.diag([0.5, 0.5])

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
        self.cur_action = action
        new_state, reward, term, trunc, info = super().step(action)

        info['constraint_violated'] = False
        if not self._is_action_valid(action):
            print(f'action {action} is not within valid range!')
            if self.term_if_constraints_violated:
                term = True
            info['constraint_violated'] = True

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

        state_error = new_state - self.goal_state
        action_error = action - self.goal_action
        if self._quadratic_reward:
            reward = -(state_error.T @ self.Q @ state_error + action_error.T @ self.R @ action_error)
        else:
            if not terminated:
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

        if np.allclose(state, self.goal_state, atol=1e-2):
            terminated = True

        return terminated

    @override
    def get_state_info(self, state):
        info = {'success': False}
        if np.allclose(state, self.goal_state, atol=1e-2):
            info = {'success': True}

        return info

    @override
    def reward_jacobian(self, state, action):
        if self._quadratic_reward:
            # Cost is -(state_error^T Q state_error + action_error^T R action_error)
            # where state_error = state - goal_state, action_error = action - goal_action
            # Jacobian w.r.t. state: d/d_state[-(err^T Q err)] = -2 * Q @ (state - goal_state)
            state_error = state - self.goal_state
            action_error = action - self.goal_action
            self.l_x = -2 * self.Q @ state_error
            self.l_u = -2 * self.R @ action_error
            return self.l_x, self.l_u
        else:
            raise NotImplementedError('First order approximation of reward is not implemented for non quadratic reward')

    @override
    def reward_hessian(self, state, action):
        if self._quadratic_reward:
            # Hessian is constant for quadratic cost in the state error
            self.l_xx = -2 * self.Q
            self.l_uu = -2 * self.R
            return self.l_xx, self.l_uu
        else:
            raise NotImplementedError(
                'Second order approximation of reward is not implemented for non quadratic reward'
            )

    @override
    def render(self):
        if not self._fig_created:
            # Create an animation of the pendulum
            plt.ion()
            plt.subplots(figsize=(10, 10), dpi=100)

            self._fig_created = True

        plt.clf()
        # Extract positions and angles
        x, z, theta, _, _, _ = self.cur_state
        goal_x, goal_z = self.goal_state[0], self.goal_state[1]

        # Compute quadrotor geometry
        rotor1_x = x - self.robot_model.L * np.cos(theta)
        rotor1_z = z - self.robot_model.L * np.sin(theta)
        rotor2_x = x + self.robot_model.L * np.cos(theta)
        rotor2_z = z + self.robot_model.L * np.sin(theta)

        # Quadrotor body and thrusters
        plt.plot(
            [rotor1_x, rotor2_x],
            [rotor1_z, rotor2_z],
            'o-',
            lw=4,
            markersize=10,
            color='tab:blue',
            label='Quadrotor Body',
        )
        # Draw center of the quadrotor
        plt.scatter([x], [z], c='tab:red', s=40, zorder=3, label='Quadrotor Center')

        # Draw thrust arrows
        if hasattr(self, 'cur_action') and self.cur_action is not None:
            left_thrust, right_thrust = self.cur_action
        else:
            left_thrust, right_thrust = self.goal_action

        arrow_scale = 0.02
        thrust_color = 'tab:purple'

        # Left thrust arrow
        arrow_dx = arrow_scale * left_thrust * np.sin(theta)
        arrow_dz = arrow_scale * left_thrust * np.cos(theta)
        plt.arrow(
            rotor1_x,
            rotor1_z,
            arrow_dx,
            arrow_dz,
            head_width=0.05,
            head_length=0.08,
            fc=thrust_color,
            ec=thrust_color,
            length_includes_head=True,
            zorder=4,
        )
        plt.text(
            rotor1_x + arrow_dx * 1.1,
            rotor1_z + arrow_dz * 1.1,
            f'{left_thrust:.2f}',
            color=thrust_color,
            fontsize=10,
            va='bottom',
            ha='center',
        )

        # Right thrust arrow
        arrow_dx = arrow_scale * right_thrust * np.sin(theta)
        arrow_dz = arrow_scale * right_thrust * np.cos(theta)
        plt.arrow(
            rotor2_x,
            rotor2_z,
            arrow_dx,
            arrow_dz,
            head_width=0.05,
            head_length=0.08,
            fc=thrust_color,
            ec=thrust_color,
            length_includes_head=True,
            zorder=4,
        )
        plt.text(
            rotor2_x + arrow_dx * 1.1,
            rotor2_z + arrow_dz * 1.1,
            f'{right_thrust:.2f}',
            color=thrust_color,
            fontsize=10,
            va='bottom',
            ha='center',
        )

        # Draw goal hover position
        plt.scatter([goal_x], [goal_z], c='tab:green', s=80, marker='X', zorder=2, label='Goal Hover Target')
        plt.plot([goal_x - 0.1, goal_x + 0.1], [goal_z, goal_z], c='tab:green', lw=1)
        plt.plot([goal_x, goal_x], [goal_z - 0.1, goal_z + 0.1], c='tab:green', lw=1)

        # Update simulation time
        plt.text(0.02, -0.2, f'Time: {self.step_cnt * self.action_dt:.2f}s')

        plt.legend(loc='upper right')
        plt.xlabel('X Position (m)')
        plt.ylabel('Z Position (m)')
        plt.title('Planar Quadrotor Simulation')
        # Limits for animation

        plt.xlim(-2, 2)
        plt.ylim(-1, 3)
        plt.grid()
        plt.pause(0.01)
