from typing_extensions import override
import math
import numpy as np
from collections import defaultdict
import random

from robotics_algorithm.env.base_env import PartiallyObservableEnv
from robotics_algorithm.state_estimation.particle_filter import ParticleFilter


class AMCL(ParticleFilter):
    def __init__(self, env: PartiallyObservableEnv, min_particles= 500, max_particles: int = 10000):
        """Adaptive Monte-Carlo localisation.

        Particle filter with Augmentation and KLD resampling.
        Augmentation: Adjust number of random samples based on the measurement likelihood.
        KLD resampling: Adjust total number of particles based on KL divergence against true prior.

        Args:
            env (PartiallyObservableEnv): environment
            min_particles (int, optional): minimum number of samples. Defaults to 500.
            max_particles (int, optional): maximum number of samples. Defaults to 10000.
        """
        super().__init__(env)

        self.hist_res = 0.1
        self.hist = defaultdict(int)
        self.min_samples = min_particles
        self.max_samples = max_particles

        # For KLD sampling
        # ! The statistical bound says: if we choose N number of samples, then we can guarantee that with probability quantile,
        # ! the KL-divergence between the MLE and the true distribution is less than epsilon.
        self.quantile = 0.99
        self.epsilon = 0.05

        # For Augmentation
        self.alpha_short_term = 0.1
        self.alpha_long_term = 0.001
        self.w_short_term = 0
        self.w_long_term = 0

    @override
    def update(self, observation):
        # Add particles until desired number is reached.
        """
        @param observation, observation
        """
        # calculate particle weights
        # - get_observation_prob may return pdf, need to normalize after.
        weights = []
        w_avg = 0
        for particle in self.particles:
            weight = self.env.get_observation_prob(particle, observation) + 1e-300  # avoid round-off to zero
            weights.append(weight)
            w_avg += 1.0 / len(self.particles) * weight
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # normalize!

        # Calculate average likelihood of measurement over a large and short time periods
        # in order to determine whether the current measurement likelihood is too large
        self.w_short_term += self.alpha_short_term * (w_avg - self.w_short_term)
        self.w_long_term += self.alpha_long_term * (w_avg - self.w_long_term)

        # Augmentation + KLD resampling.
        # ! probability of choosing random sample is decided by the divergence between short-term likelihood and long-term likelihood
        # ! if short-term likelihood is worse (small), then random sample probability is higher
        random_sample_prob = max(0.0, 1.0 - self.w_short_term / self.w_long_term)
        # print(w_avg, self.w_short_term, self.w_long_term, random_sample_prob)
        particle_indices = np.arange(len(self.particles))
        new_particles = []
        while len(new_particles) < self._KLD_sample_limit():  # ! Keep adding particles until statistical bound is met
            x = random.uniform(0, 1)
            if x <= random_sample_prob:
                new_particle = self.env.random_state()
            else:
                new_particle_index = np.random.choice(particle_indices, replace=True, p=weights)  # Draw 1 sample
                new_particle = self.particles[new_particle_index]
            new_particles.append(new_particle)

            # check sample falls into which bin, and increment bin count
            particle_key = tuple(np.floor(np.array(new_particle) / self.hist_res).astype(np.int32).)
            self.hist[particle_key] += 1

        self.particles = new_particles
        self.hist.clear()  # ! Statistical bound is calculated every iteration.

    def _KLD_sample_limit(self):
        # Compute the required number of samples, given that there are k bins
        # with samples in them.  This is taken directly from Fox et al.
        k = len(self.hist)  # How many bins are filled up
        if k <= 1:
            return self.max_samples

        # ! The statistical bound says: if we choose n number of samples, then we can guarantee that with probability quantile,
        # ! the KL-divergence between the MLE and the true distribution is less than epsilon.
        a = 1
        b = 2 / (9 * (k - 1))
        c = math.sqrt(2 / (9 * (k - 1))) * self.quantile  # Upper X quantile of normal distribution
        x = a - b + c
        n = math.ceil((k - 1) / (2 * self.epsilon) * x * x * x)  # epsilon is the threshold of KL divergence.

        if n < self.min_samples:
            return self.min_samples

        if n > self.max_samples:
            return self.max_samples

        return n

    @property
    def num_of_cur_particles(self):
        return len(self.particles)