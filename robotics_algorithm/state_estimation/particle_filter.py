import numpy as np

from robotics_algorithm.env.base_env import PartiallyObservableEnv, EnvType


class ParticleFilter:
    def __init__(
        self, env: PartiallyObservableEnv, num_of_particles: int = 10000, uniform_particle_ratio: float = 0.05
    ):
        """Particle filter

        Args:
            env (BaseEnv): _description_
            num_of_particles (int, optional): num of particles. Defaults to 10000.
            uniform_particle_ratio (float, optional): if set to non-zero, filter will add a small number of random
                particles after each iteration to tackle particle deprivation problem. Defaults to 0.05.
        """

        # sanity check
        assert env.state_transition_type == EnvType.STOCHASTIC.value
        assert env.observability == EnvType.PARTIALLY_OBSERVABLE.value

        self.env = env

        # attributes
        self.num_of_particles = num_of_particles
        self.num_of_uniform_particles = int(num_of_particles * uniform_particle_ratio)

        self.particles = []
        for _ in range(self.num_of_particles):
            particle = self.env.random_state()
            self.particles.append(particle)

    def set_initial_state(self, state: np.ndarray):
        """
        Set the initial state of filter.

        @param state, initial state
        """
        self.state = np.array(state)
        for i in range(self.num_of_uniform_particles, self.num_of_particles):
            self.particles[i] = state

    def get_state(self) -> np.ndarray:
        # simply average
        mean_state = np.mean(np.array(self.particles), axis=0)
        return mean_state

    def run(self, action: np.ndarray, observation: np.ndarray):
        """
        Run one iteration of the filter.

        @param action, control
        @param observation, observation
        """
        self.predict(action)
        self.update(observation)

    def predict(self, action):
        """
        @param action, action
        """
        # sample new particles according to state transition function
        new_particles = []
        for particle in self.particles:
            # NOTE: Unlike kalman filter which assumes a Gaussian distribution, here we sample state transition instead
            #       calling state_transition_function which returns the mean of the new state.
            new_particle = self.env.sample_state_transition(particle, action)[0]
            new_particles.append(new_particle)

        self.particles = new_particles

    def update(self, observation):
        """
        @param observation, observation
        """
        # calculate particle weights
        # - get_observation_prob may return pdf, need to normalize after.
        weights = []
        for particle in self.particles:
            weight = self.env.get_observation_prob(particle, observation) + 1e-300  # avoid round-off to zero
            weights.append(weight)
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # normalize!

        # importance sampling, sample with replacement according to weights
        particle_indices = np.arange(self.num_of_particles)
        new_particle_indices = np.random.choice(particle_indices, size=self.num_of_particles, replace=True, p=weights)
        new_particles = np.array(self.particles)[new_particle_indices].tolist()

        # Add random particles if required
        for i in range(self.num_of_uniform_particles):
            particle = self.env.random_state()
            new_particles[i] = particle

        self.particles = new_particles
