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

    def state_transition_jacobian(self, state: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Linearize dynamics in discrete time so that x_new = A @ x + B @ u

        Args:
            state (np.ndarray): previous state
            action (np.ndarray): action to apply

        Returns:
            tuple[np.ndarray, np.ndarray]: A and B
        """
        raise NotImplementedError

    def state_transition_hessian(self, state: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Quadratic approximation of the state transition function around current state

        Args:
            state (np.ndarray): previous state
            action (np.ndarray): action to apply

        Returns:
            f_xx, f_ux, f_uu such that new_state = first order terms + 0.5 * state.T @ f_xx @ state + action.T @ f_ux @ state + 0.5 * action.T @ f_uu @ action.
        """
        raise NotImplementedError
