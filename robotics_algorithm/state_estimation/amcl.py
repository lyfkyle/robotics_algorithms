from typing_extensions import override
import math

from robotics_algorithm.state_estimation.particle_filter import ParticleFilter


class AMCL(ParticleFilter):
    def __init__(self):
        """Adaptive Monte-Carlo localisation.

        Particle filter with Augmentation and KLD resampling.
            Augmentation: number of random samples added are determined by the measurement confidence
            KLD resampling: Adjust total number of particles based on KL divergence against true prior.
        """
        pass

    @override
    def update(self, observation):
        # Add particles until desired number is reached.
        pass

    def _pf_sample_limit(self, k: int):
        # Compute the required number of samples, given that there are k bins
        # with samples in them.  This is taken directly from Fox et al.

        if k <= 1:
            return self.max_samples

        # The statistical bound says: if we choose n number of samples, then we can guarantee that with probability quantile,
        # the KL-divergence between the MLE and the true distribution is less than epsilon.
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
