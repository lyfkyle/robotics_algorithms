import numpy as np

from robotics_algorithm.env.base_env import BaseEnv, SpaceType


class PID:
    """Implements Linear-quadratic regulator.

    # Given a system with linear state transition function and quadratic cost function, the optimal control strategy
    # can be computed analytically and has the form: u = -Kx
    """

    def __init__(self, env: BaseEnv, goal_state=None, Kp: float = 1.0, Ki: float = 0.0, Kd: float = 0.0):
        """
        State transition:
        """
        assert env.state_space.type == SpaceType.CONTINUOUS.value
        assert env.action_space.type == SpaceType.CONTINUOUS.value

        self.env = env
        if goal_state is None:
            self.goal_state = np.array(env.goal_state, dtype=np.float32)
        else:
            self.goal_state = np.array(goal_state, dtype=np.float32)
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self._prev_error = np.zeros_like(self.goal_state, dtype=np.float32)
        self._accum_error = np.zeros_like(self.goal_state, dtype=np.float32)

    def run(self, state: np.ndarray) -> np.ndarray:
        """Compute the current action based on the current state.

        Args:
            state (np.ndarray): current state.

        Returns:
            np.ndarray: current action
        """
        state = np.array(state)

        error = self.goal_state - state
        delta_error = error - self._prev_error
        self._accum_error += error

        control = self.Kp * error + self.Kd * delta_error + self.Ki * self._accum_error

        self._prev_error = error

        return control
