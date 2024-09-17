#!/usr/bin/evn python

import sys
sys.path.append('../../environment/')
sys.path.append('../')

import numpy as np
import random
import matplotlib.pyplot as plt
import math
from scipy.stats import norm

from two_d_localization_with_feature import TwoDLocalizationWithFeature
from particle_filter import ParticleFilter

# -------- Settings ------------
NUM_OF_TIMESTAMP = 10

NUM_OF_PARTICLES = 500
NUM_OF_PARTICLES_AT_TRUE_POS = 250

# -------- Helper Functions -------------
def get_filter_state(particle_filter, method = "gaussian"):
    # Option 1: use gaussion to fit particles. Return filter state as the mean of fitted gaussion
    if method == "gaussian":
        filter_x = [particle_filter.particles[i][0, 0] for i in range(particle_filter.num_of_particles)]
        filter_y = [particle_filter.particles[i][1, 0] for i in range(particle_filter.num_of_particles)]
        filter_theta = [particle_filter.particles[i][2, 0] for i in range(particle_filter.num_of_particles)]
        mean_x, std = norm.fit(filter_x)
        mean_y, std = norm.fit(filter_y)
        mean_theta, std = norm.fit(filter_theta)
        filter_state = np.array([[mean_x], [mean_y], [mean_theta]])

    # Option 2: simply average
    elif method == "average":
        filter_state = np.mean(np.array(particle_filter.particles), axis = 0)
    return filter_state

# -------- Main Code ----------

# Initialize environment
env = TwoDLocalizationWithFeature()

# Initialize filter
# construct particles
num_of_particles = NUM_OF_PARTICLES
initial_particles = []
# NUM_OF_PARTICLES_AT_TRUE_POS of the particle is at the true initial pos
for _ in range(NUM_OF_PARTICLES_AT_TRUE_POS):
    particle = env.state
    initial_particles.append(particle)
# the rest is randomly distributed
for _ in range(NUM_OF_PARTICLES - NUM_OF_PARTICLES_AT_TRUE_POS):
    particle = np.array([[random.uniform(0.0, env.size)],
                         [random.uniform(0.0, env.size)],
                         [random.uniform(-math.pi, math.pi)]])
    initial_particles.append(particle)

my_filter = ParticleFilter(initial_particles, env.state_transition, env.measuremnt_prob)

# Add initial state
true_states = []
filter_states = []
true_states.append(env.state)
filter_states.append(get_filter_state(my_filter))

# sanity check
# meas = env.measure()
# print("meas: {}".format(meas))

# Run test
for i in range(NUM_OF_TIMESTAMP):
    print("timestamp : {}".format(i))
    control = [random.uniform(-1.0, 1.0), random.uniform(-math.pi / 2, math.pi / 2)]
    # control = [random.uniform(-1.0, 1.0), 0]

    meas = env.control_and_measure(control)
    # print("True state: {}".format(env.state))
    # print("meas: {}".format(meas))

    my_filter.predict(control)
    # print("Filter belief before meas: {}".format(get_filter_state(my_filter)))
    my_filter.update(meas)
    # print("Filter belief after meas: {}".format(get_filter_state(my_filter)))

    true_states.append(env.state)
    filter_states.append(get_filter_state(my_filter))

# plot result
NUM_OF_TIMESTAMP += 1
t = np.arange(NUM_OF_TIMESTAMP)

fig, (ax1, ax2, ax3) = plt.subplots(nrows = 3, ncols = 1)
ax1.plot(t, [true_states[i][0, 0] for i in range(NUM_OF_TIMESTAMP)], 'k')
ax1.plot(t, [filter_states[i][0, 0] for i in range(NUM_OF_TIMESTAMP)], 'bo')
ax1.set_ylabel('x')
ax2.plot(t, [true_states[i][1, 0] for i in range(NUM_OF_TIMESTAMP)], 'k')
ax2.plot(t, [filter_states[i][1, 0] for i in range(NUM_OF_TIMESTAMP)], 'bo')
ax2.set_ylabel('y')
ax3.plot(t, [true_states[i][2, 0] for i in range(NUM_OF_TIMESTAMP)], 'k')
ax3.plot(t, [filter_states[i][2, 0] for i in range(NUM_OF_TIMESTAMP)], 'bo')
ax3.set_ylabel('theta')
plt.show()