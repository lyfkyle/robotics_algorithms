import math
import numpy as np
import random

class ParticleFilter:
    def __init__(self, initial_particles, sample_transition, measurement_prob):
        '''
        @param initial_particles. A list of initial particles
        @param sample_transition, a function to sample state transition given state and control
        @param measurement_orb, a function to get the prob of a given measurement given state
        '''
        # settings
        self.enable_add_particle = False # if set to True, this filter will add a small number of random particles after each iteration to tackle particle deprivation problem

        # attributes
        self.num_of_particles = len(initial_particles)
        self.particles = initial_particles # represented by a set of particles
        self.weights = [1] * self.num_of_particles
        self._sample_transition = sample_transition
        self._measurement_prob = measurement_prob

    def predict(self, control):
        '''
        @param control, control
        '''
        # sample new particles according to state transition function
        new_particles = []
        for particle in self.particles:
           new_particle = self._sample_transition(particle, control)
           new_particles.append(new_particle)

        self.particles = new_particles

    def update(self, measurement):
        '''
        @param measurement, measurement
        '''
        # calculate particle weights
        weights = []
        for particle in self.particles:
            weight = self._measurement_prob(particle, measurement) + 1e-300 # avoid round-off to zero
            weights.append(weight)
        weights = np.array(weights)
        weights = weights / np.sum(weights) # make it sum to 1

        # importance sampling
        # particle_idx = np.arange(self.num_of_particles)
        # new_particle_idxes = np.random.choice(particle_idx, size=self.num_of_particles, replace=True, p=weights)
        # new_particles = [self.particles[idx] for idx in new_particle_idxes]
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1. # avoid round-off error
        new_particle_idxes = [np.searchsorted(cumulative_sum, random.random()) for _ in range(self.num_of_particles)]
        new_particles = [self.particles[idx] for idx in new_particle_idxes]

        # Add random particles if required
        # Note, TODO this is hard coded for now.
        if self.enable_add_particle:
            for i in range(-50, 0):
                particle = np.array([[random.uniform(0.0, 10.0)],
                                     [random.uniform(0.0, 10.0)],
                                     [random.uniform(-math.pi, math.pi)]])
                new_particles[i] = particle

        self.particles = new_particles






