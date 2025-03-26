import numpy as np

from robotics_algorithm.env.continuous_2d.diff_drive_2d_control import DiffDrive2DControl


class DWA:
    """Implements Dynamic Window Approach for path follow.

    paper https://www.ri.cmu.edu/pub_files/pub1/fox_dieter_1997_1/fox_dieter_1997_1.pdf

    The basic idea of the Dynamic Window Approach (DWA) algorithm is as follows:

    - Discretely sample in the robot's control space
    - For each sampled velocity, perform forward simulation from the robot's current state to predict what would happen if the sampled velocity were applied for some (short) period of time.
    - Evaluate (score) each trajectory resulting from the forward simulation.
    - Pick the highest-scoring trajectory and send the associated velocity to the mobile base.
    - Rinse and repeat.
    """

    def __init__(
        self,
        env: DiffDrive2DControl,
        min_lin_vel: float = 0.0,
        max_lin_vel: float = 0.5,
        lin_vel_samples: int = 10,
        min_ang_vel: float = -1.0,
        max_ang_vel: float = 1.0,
        ang_vel_samples: int = 20,
        simulate_time: float = 0.5,
    ) -> None:
        """
        Constructor

        Args:
            env (DiffDrive2DControl): The environment.
            min_lin_vel (float): Minimum linear velocity.
            max_lin_vel (float): Maximum linear velocity.
            lin_vel_samples (int): Number of linear velocity samples.
            min_ang_vel (float): Minimum angular velocity.
            max_ang_vel (float): Maximum angular velocity.
            ang_vel_samples (int): Number of angular velocity samples.
            simulate_time (float): Simulation time.
        """
        assert isinstance(env, DiffDrive2DControl), 'env must be a DiffDrive2DControl'

        self.env = env
        self._min_lin_vel = min_lin_vel
        self._max_lin_vel = max_lin_vel
        self._lin_vel_steps = (max_lin_vel - min_lin_vel) / lin_vel_samples
        self._min_ang_vel = min_ang_vel
        self._max_ang_vel = max_ang_vel
        self._ang_vel_steps = (max_ang_vel - min_ang_vel) / ang_vel_samples
        self._sim_steps = int(simulate_time / env.action_dt)

        self.sampled_action_seq = []
        self._sample_action_seq()

    def _sample_action_seq(self):
        for lin_vel in np.arange(self._min_lin_vel, self._max_lin_vel, self._lin_vel_steps):
            for ang_vel in np.arange(self._min_ang_vel, self._max_ang_vel, self._ang_vel_steps):
                self.sampled_action_seq.append([lin_vel, ang_vel])

    def run(self, state: np.ndarray) -> np.ndarray:
        """Compute the current action given current state

        This function uses Dynamic Window Approach to select the best action given the current state.

        Args:
            state (np.ndarray): current state

        Returns:
            np.ndarray: current action
        """
        best_cost = np.inf
        best_action = np.array([0.0, 0.0])
        self.best_traj = [state]

        # Iterate over all possible actions
        for lin_vel, ang_vel in self.sampled_action_seq:
            cur_state = state
            total_cost = 0

            # * The action stays constant for self._sim_steps because we do not enforce acceleration limits.
            # * If acceleration limits are to be enforced, here we need to slowly accelerate/decelerate to sampled
            # * velocity (like ROS2).
            action = np.array([lin_vel, ang_vel])
            traj = [cur_state]
            for _ in range(self._sim_steps):
                # Take one step in the simulation
                new_state, reward, term, trunc, _ = self.env.sample_state_transition(cur_state, action)
                cost = -reward

                total_cost += cost
                cur_state = new_state
                traj.append(new_state)

                # If the simulation is terminated, break
                if term or trunc:
                    break

            # Update the best cost and trajectory if the current one is better
            if total_cost < best_cost:
                best_cost = total_cost
                best_action = action
                self.best_traj = traj

        # Return the best action
        return best_action
