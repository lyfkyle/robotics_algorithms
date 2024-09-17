import math
import numpy as np

from robotics_algorithm.env.base_env import BaseEnv, SpaceType


class MPPI(object):
    """
    Model Predictive Path Integral controller.

    modified from https://github.com/MizuhoAOKI/python_simple_mppi/blob/master/scripts/mppi_pathtracking.py.
    Original Paper: https://arxiv.org/pdf/1707.02342s

    The general idea is to sample a bunch of trajectories using env state transition model and get their respective cost.
    The final optimal trajectory is computed as the weighted average of all sampled trajectories with weight being
    exp(-cost). This has the effect of minimizing KL divergence between optimal trajectory and sampled trajectory.
    """

    def __init__(
        self,
        env: BaseEnv,
        num_traj_samples: int = 250,
        sample_traj_len: int = 20,
        action_mean: float | list = 0.0,
        action_std: float | list = 1.0,
        param_lambda: float = 0.8,
        filter_window_size: int = 5,
    ):
        """Constructor

        Args:
            env (BaseEnv): The env
            num_traj_samples (int, optional): num of trajectories to sample. Defaults to 250.
            sample_traj_len (int, optional): the lookahead distance. Defaults to 20.
            action_mean (float | list, optional): the mean of sampled action. Also form the nominal action. Defaults to 0.0.
            action_std (float| list, optional): the std of sampled action. Defaults to 1.0.
            param_lambda (float, optional): controls how much we weigh trajectory cost. Smaller value means we favour
                low cost trajectories more. Defaults to 0.8.
            filter_window_size (int, optional): window size of moving average filter to smooth final computed control
                trajectory. Defaults to 5.
        """
        assert env.state_space.type == SpaceType.CONTINUOUS.value
        assert env.action_space.type == SpaceType.CONTINUOUS.value

        self.env = env

        self.sample_traj_len = sample_traj_len
        self.num_samples = num_traj_samples
        self.uniform_control_bias = 0.1
        self.action_mean = np.array(action_mean)
        self.action_std = np.array(action_std)
        self.param_lambda = param_lambda
        self.filter_window_size = filter_window_size

        self.action_shape = np.array(env.sample_action()).shape
        self.prev_actions = np.tile(self.action_mean, (self.sample_traj_len, 1))
        self.nominal_action = np.tile(self.action_mean, (self.sample_traj_len, 1))

    def run(self, state: list) -> list:
        """Compute the current action given current state

        Args:
            state (list): current state

        Returns:
            list: current action
        """
        # Sample trajectories
        all_costs = []
        all_noises = []
        for k in range(self.num_samples):
            # For each trajectory, sample actions by sampling noises
            cur_state = state
            total_cost = 0
            noises = np.random.randn(self.sample_traj_len, *self.action_shape) * self.action_std
            for t in range(self.sample_traj_len):
                # actual action at each timestep is nominal action + sampled noise
                noise = noises[t]
                if k < self.uniform_control_bias * self.num_samples:
                    sampled_action = self.nominal_action[t] + noise
                else:
                    sampled_action = self.prev_actions[t] + noise

                sampled_action = np.clip(sampled_action, self.env.action_space.space[0], self.env.action_space.space[1])

                # Simulate state transitions forward (the model predictive part)
                new_state, reward, term, trunc, _ = self.env.sample_state_transition(cur_state, sampled_action.tolist())
                cost = -reward

                total_cost += cost
                cur_state = new_state

                if term or trunc:
                    break

            all_costs.append(total_cost)
            all_noises.append(noises)

        # Rank all trajectories according to its cost.
        all_costs = np.array(all_costs)
        all_noises = np.array(all_noises)
        weights = self._compute_weights(np.array(all_costs))

        # Final trajectory is the weighted average of all sampled trajectories
        # - By convention, weighted average operate on noise level
        final_noise = np.zeros_like(all_noises[0])
        for k in range(self.num_samples):
            final_noise += weights[k] * all_noises[k]

        # apply moving average filter for smoothing actions
        w_epsilon = self._moving_average_filter(xx=final_noise, window_size=self.filter_window_size)

        # Construct final optimal action trajectory by adding optimal noise
        actions = self.prev_actions + w_epsilon
        actions = np.clip(actions, self.env.action_space.space[0], self.env.action_space.space[1])

        # update previous control sequence, shift 1 step to the left
        self.prev_actions[:-1] = actions[1:]
        self.prev_actions[-1] = actions[-1]

        # Execute the first one
        return actions[0].tolist()

    def _compute_weights(self, traj_costs: np.ndarray) -> np.ndarray:
        """compute weights for each sample"""
        # prepare buffer
        w = np.zeros(self.num_samples)

        # calculate rho
        rho = traj_costs.min()

        # calculate eta
        eta = 0.0
        for k in range(self.num_samples):
            eta += np.exp((-1.0 / self.param_lambda) * (traj_costs[k] - rho))

        # calculate weight
        for k in range(self.num_samples):
            w[k] = (1.0 / eta) * np.exp((-1.0 / self.param_lambda) * (traj_costs[k] - rho))

        return w

    def _moving_average_filter(self, xx: np.ndarray, window_size: int) -> np.ndarray:
        """Apply moving average filter for smoothing input sequence.

        Ref. https://zenn.dev/bluepost/articles/1b7b580ab54e95
        """
        b = np.ones(window_size) / window_size
        dim = xx.shape[1]
        xx_mean = np.zeros(xx.shape)

        for d in range(dim):
            xx_mean[:, d] = np.convolve(xx[:, d], b, mode="same")
            n_conv = math.ceil(window_size / 2)
            xx_mean[0, d] *= window_size / n_conv
            for i in range(1, n_conv):
                xx_mean[i, d] *= window_size / (i + n_conv)
                xx_mean[-i, d] *= window_size / (i + n_conv - (window_size % 2))

        return xx_mean
