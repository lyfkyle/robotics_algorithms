import numpy as np
import matplotlib.pyplot as plt


class DoubleIntegrator:
    def __init__(self, use_discrete_time_model=True, dt=0.01):
        # Robot parameters

        # x = [q, q_dot]
        # x_dot = [q_dot, q_dot_dot] = Ax + Bu
        if not use_discrete_time_model:
            self.A = np.array([[0, 1], [0, 0]])
            self.B = np.array([[0, 1]]).T
        else:
            self.A = np.array([[1, dt], [0, 1]])
            self.B = np.array([[0.5 * dt * dt], [dt]])

        self.use_discrete_time_model = use_discrete_time_model

    def control(self, state: list, control: list, dt: float) -> list:
        """Compute the end state given then current state and control.

        Args:
            state (list): [q, q_dot] robot's current state
            control (list): left and right wheel velocities in radians
            dt (float): time step

        Returns:
            new state [q, q_dot]
        """
        state = np.array(state).reshape(-1, 1)
        control = np.array(control).reshape(-1, 1)

        if not self.use_discrete_time_model:
            # discretize using zero order hold, assuming dt is small enough
            x_dot = self.A @ state + self.B @ control
            new_state = state + x_dot * dt
        else:
            # if discrete_time model is used, transition func directly gives new state
            new_state = self.A @ state + self.B @ control

        return new_state.reshape(-1).tolist()


if __name__ == "__main__":
    # Simulation parameters
    dt = 0.01  # time step
    total_time = 10.0  # total simulation time
    num_steps = int(total_time / dt)

    # Initial state [q, q_dot]
    state = np.array([0.0, 0.0])

    # Control inputs (accelerations)
    control_inputs = [0.1]

    # Record state history for plotting
    state_history = [state]

    system = DoubleIntegrator()

    # Run simulation
    time_history = [0]
    for time_step in range(num_steps):
        state = system.control(state, control_inputs, dt)
        state_history.append(state)
        time_history.append(dt + time_step * dt)

    # Convert state history to numpy array for easier indexing
    state_history = np.array(state_history)
    print(state_history)

    # Plot results
    plt.figure()
    plt.plot(time_history, state_history[:, 0], label="Robot Path")
    plt.xlabel("time (s)")
    plt.ylabel("position (m)")
    plt.title("Double integrator Robot Path")
    plt.legend()
    plt.grid()
    plt.show()
