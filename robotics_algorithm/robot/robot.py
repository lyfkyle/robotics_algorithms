import numpy as np


class Robot:
    def __init__(self, dt):
        self.dt = dt

    def control(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Simulate robot kinematics/dynamics forward in time to get the next state

        Args:
            state (np.ndarray): previous state
            action (np.ndarray): action to apply

        Returns:
            np.ndarray: next state
        """
        raise NotImplementedError()

    def linearize_state_transition(self, state: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Linearize dynamics in discrete time so that x_new = A @ x + B @ u

        Args:
            state (np.ndarray): previous state
            action (np.ndarray): action to apply

        Returns:
            tuple[np.ndarray, np.ndarray]: A and B
        """
        raise NotImplementedError
