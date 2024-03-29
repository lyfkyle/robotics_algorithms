#!/usr/bin/evn python

import sys
import os.path as osp
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../../'))

import random

from env.one_d_localization_discrete import OneDLocalizationDiscrete
from state_estimation.discrete_bayes_filter import DiscreteBayesFilter

env = OneDLocalizationDiscrete()
my_filter = DiscreteBayesFilter(env.state_space_size)

print("Initial filter belief : {}".format(my_filter.state_belief))

for i in range(10):
    target_end_pos = random.randint(0, env.state_space_size)
    dist = target_end_pos - env.pos
    true_pos = env.move(dist)
    meas = env.sense()

    my_filter.predict(dist, env.transition_func)
    my_filter.update(meas, env.measurement_func)

    print("Move by {}".format(dist))
    print("True pos: {}".format(true_pos))
    print("Filter belief : {}".format(my_filter.state_belief))
