import numpy as np

from robotics_algorithm.env.base_env import BaseEnv


class PID(object):
    """Implements Linear-quadratic regulator.

    # Given a system with linear state transition function and quadratic cost function, the optimal control strategy
    # can be computed analytically and has the form: u = -Kx
    """

    def __init__(self, env: BaseEnv, Kp: float = 1.0, Ki: float = 0.0, Kd: float = 0.0):
        """
        State transition:
        """
        self.env = env
        self.goal_state = np.array(env.goal_state)
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self._prev_error = np.zeros_like(self.goal_state)
        self._accum_error = np.zeros_like(self.goal_state)

    def run(self, state: list) -> list:
        """Compute the current action based on the current state.

        Args:
            state (list): current state.

        Returns:
            list: current action
        """
        state = np.array(state)

        error = state - self.goal_state
        delta_error = error - self._prev_error
        self._accum_error += error

        control = self.Kp @ error + self.Kd @ delta_error + self.Ki @ self._accum_error

        self._prev_error = error

        return control.tolist()
