"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
from typing_extensions import override

import numpy as np
import pygame
from pygame import gfxdraw

from robotics_algorithm.env.base_env import ContinuousSpace, DeterministicEnv, FullyObservableEnv
from robotics_algorithm.robot.cartpole import Cartpole


class CartPoleEnv(DeterministicEnv, FullyObservableEnv):
    """
    ## Description

    This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
    ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
    The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
     in the left and right direction on the cart.

    ## Action Space

    The action is a list with shape `(1,)` which can take values `{-10, 10.0}` indicating the force of the push.

    **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
     the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

    ## Observation Space

    The observation is a `list` with shape `(4,)` with the values corresponding to the following positions and velocities:

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |

    **Note:** While the ranges above denote the possible values for observation space of each element,
        it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
    -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
       if the cart leaves the `(-2.4, 2.4)` range.
    -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
       if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

    ## Rewards
    Since the goal is to keep the pole upright for as long as possible, by default, a reward of `+1` is given for every
    step taken, including the termination step. The default reward threshold is 500 for v1 and 200 for v0 due to the
    time limit on the environment.

    If `sutton_barto_reward=True`, then a reward of `0` is awarded for every non-terminating step and `-1` for the
    terminating step. As a result, the reward threshold is 0 for v0 and v1.

    ## Starting State
    All observations are assigned a uniformly random value in `(-0.05, 0.05)`

    ## Episode End
    The episode ends if any one of the following occurs:

    1. Termination: Pole Angle is greater than ±12°
    2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Truncation: Episode length is greater than 500 (200 for v0)


    | Parameter               | Type       | Default                 | Description                                                                                   |
    |-------------------------|------------|-------------------------|-----------------------------------------------------------------------------------------------|
    | `sutton_barto_reward`   | **bool**   | `False`                 | If `True` the reward function matches the original sutton barto implementation                |
    """

    metadata = {
        "render_fps": 50,
    }

    def __init__(self, sutton_barto_reward: bool = False):
        super().__init__()

        self.robot_model = Cartpole()

        self._sutton_barto_reward = sutton_barto_reward

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        self.max_steps = 1000

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.inf,
                self.theta_threshold_radians * 2,
                np.inf,
            ],
            dtype=np.float32,
        )

        self.action_space = ContinuousSpace(low=[-10], high=[10])
        self.state_space = ContinuousSpace(low=-high, high=high)

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.cur_state: np.ndarray | None = None

        self.steps_beyond_terminated = None

    @override
    def reset(self):
        self.cur_state = (np.random.randn(4) * 0.05).tolist()  # near-upright position
        self.goal_state = [0, 0, 0, 0]  # upright position

        self.steps_beyond_terminated = None
        self.step_cnt = 0

        return self.cur_state, {}

    @override
    def step(self, action: list) -> tuple[list, float, bool, bool, dict]:
        new_state, reward, term, trunc, info = super().step(action)

        self.step_cnt += 1
        trunc = self.step_cnt > self.max_steps  # set truncated flag

        return new_state, reward, term, trunc, info

    @override
    def state_transition_func(self, state: list, action: list):
        assert self.cur_state is not None, "Call reset before using step method."

        new_state = self.robot_model.control(state, action)
        return new_state

    @override
    def reward_func(self, state: list, action: list = None, new_state: list = None) -> float:
        x = new_state[0]
        theta = new_state[2]

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            reward = 0.0 if self._sutton_barto_reward else 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0

            reward = -1.0 if self._sutton_barto_reward else 1.0
        else:
            if self.steps_beyond_terminated == 0:
                print(
                    "You are calling 'step()' even though this environment has already returned terminated = True. "
                    "You should always call 'reset()' once you receive 'terminated = True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1

            reward = -1.0 if self._sutton_barto_reward else 0.0

        return reward

    @override
    def get_state_info(self, state):
        x = state[0]
        theta = state[2]

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        return terminated, False, {}

    def render(self):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.robot_model.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.cur_state is None:
            return None

        x = self.cur_state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))

        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


if __name__ == "__main__":
    env = CartPoleEnv()  # For online tree search, dense reward needs to be enabled.
    # env = CliffWalking(dense_reward=True)  # For online tree search, dense reward needs to be enabled.
    state = env.reset()
    env.render()

    # Execute
    while True:
        # Call tree search online
        # action = env.sample_action()
        action = 0.0
        new_state, reward, term, trunc, info = env.step(action)

        print(state)
        print(action)
        print(reward)
        print(new_state)

        env.render()
        state = new_state

        if term or trunc:
            break
