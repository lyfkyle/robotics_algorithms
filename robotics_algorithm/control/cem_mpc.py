import numpy as np
from scipy.stats import truncnorm

from robotics_algorithm.env.base_env import BaseEnv, SpaceType


class CEMMPC(object):
    """Implements Cross-entropy method for model-predictive control.

    Assuming a Gaussion policy, the general idea is to sample a bunch of trajectories using env state transition model
    and get their respective cost. Then take the top-K best trajectories and update the policy's mean and std based
    on these trajectories. We then sample another set of trajectories based on the updated Gaussian policy. We iterate
    this process for certain number of times and return the best action in the current best trajectory.
    """

    def __init__(
        self,
        env: BaseEnv,
        num_traj_samples: int = 40,
        num_elites: int = 10,
        sample_traj_len: int = 20,
        action_mean: float | list = 0.0,
        action_std: float | list = 1.0,
        opt_iter: int = 5,
        alpha: float = 0.1,
    ):
        """Constructor

        Args:
            env (BaseEnv): the env.
            num_traj_samples (int, optional): number of rollout trajectories to sample. Defaults to 40.
            num_elites (int, optional): number of best trajectories to use for estimating the next mean and std.
                Defaults to 10.
            sample_traj_len (int, optional): the length of each rollout trajectory. Defaults to 20.
            action_mean (float | list, optional): the nominal mean of sampled action. Defaults to 0.0.
            action_std (float | list, optional): the nominal std of sampled action. Defaults to 1.0.
            opt_iter (int, optional): number of optimization iteration. Defaults to 5.
            alpha (float, optional): the low pass filter gain for updating mean and std in each optimization iteration.
                Defaults to 0.1.
        """
        assert env.state_space.type == SpaceType.CONTINUOUS.value
        assert env.action_space.type == SpaceType.CONTINUOUS.value

        self.env = env
        self.num_traj_samples = num_traj_samples
        self.num_elites = num_elites
        self.sample_traj_len = sample_traj_len
        self.opt_iter = opt_iter
        self.alpha = alpha

        self.nominal_action_mean = np.array(action_mean)
        self.nominal_action_std = np.array(action_std)
        self.action_seq_mean = np.tile(self.nominal_action_mean, (self.sample_traj_len, 1))
        self.action_seq_std = np.tile(self.nominal_action_std, (self.sample_traj_len, 1))

    def run(self, state: list) -> list:
        """Compute the current action given current state

        Args:
            state (list): current state

        Returns:
            list: current action
        """
        for _ in range(self.opt_iter):
            # Sample trajectories
            all_costs = np.zeros((self.num_traj_samples))
            all_traj_rollouts = np.zeros((self.num_traj_samples, *self.action_seq_mean.shape))

            # Sample actions in a batch [K, N, Action_dim]
            sampled_action_seq = self._sample_action_seq()

            # Perform rollout
            for k in range(self.num_traj_samples):
                cur_state = state
                total_cost = 0
                for t in range(self.sample_traj_len):
                    action = sampled_action_seq[k, t].tolist()
                    new_state, reward, term, trunc, _ = self.env.sample_state_transition(cur_state, action)
                    cost = -reward

                    total_cost += cost
                    cur_state = new_state
                    all_traj_rollouts[k, t] = sampled_action_seq[k, t]

                    if term or trunc:
                        break

                all_costs[k] = total_cost

            # Update policy distribution based on elite trajectories
            self._update_distribution(all_traj_rollouts, all_costs)

        # After enough optimization iterations,
        # Get the best trajectory
        best_traj_idx = np.argmin(all_costs)
        self.best_traj = all_traj_rollouts[best_traj_idx]  # For debug purpose

        # update previous control sequence, shift 1 step to the left
        action_mean = np.copy(self.action_seq_mean)
        self.action_seq_mean[:-1] = action_mean[1:]
        self.action_seq_mean[-1] = action_mean[-1]

        # reset std to nominal
        self.action_seq_std = np.copy(self.nominal_action_std)

        # Return the best traj's first action
        return self.best_traj[0].tolist()

    def _sample_action_seq(self):
        m = self.action_seq_mean[None, :]
        s = self.action_seq_std[None, :]
        samples = truncnorm.rvs(
            self.env.action_space.space[0],
            self.env.action_space.space[1],
            loc=m,
            scale=s,
            size=(self.num_traj_samples, *self.action_seq_mean.shape),
        )
        # NOTE: truncnorm may still violate the bounds a little bit.
        samples = np.clip(samples, self.env.action_space.space[0], self.env.action_space.space[1])
        return samples  # shape: [K, N, Action_dim]

    def _update_distribution(self, all_traj_rollouts: np.ndarray, all_costs: np.ndarray):
        # Get elite parameters
        elite_idxs = np.array(all_costs).argsort()[: self.num_elites]
        elite_sequences = all_traj_rollouts[elite_idxs]

        # fit around mean of elites
        new_mean = elite_sequences.mean(axis=0)
        new_std = elite_sequences.std(axis=0)

        # Low pass filter
        self.action_seq_mean = (1 - self.alpha) * new_mean + self.alpha * self.action_seq_mean  # [h,d]
        self.action_seq_std = (1 - self.alpha) * new_std + self.alpha * self.action_seq_std
