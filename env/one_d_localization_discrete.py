#!/usr/bin/evn python

import numpy as np

# !!! must be odd size with center value being the probability of perfect control
TRANSITION_PROB = [0.15, 0.7, 0.15]
# !!! must be odd size with center value being the probability of perfect measurement
MEASUREMENT_PROB = [0.15, 0.7, 0.15]


class OneDLocalizationDiscrete(object):

    def __init__(self, max_pos = 20, transition_prob=TRANSITION_PROB, meas_prob=MEASUREMENT_PROB):
        self.max_pos = max_pos
        self.state_space_size = max_pos
        self.pos = int(max_pos // 2) # at center
        self.meas = 0
        self.transition_prob = transition_prob
        self.meas_prob = meas_prob

    def move(self, distance=1):
        '''
        move in the specified direction
        with some small chance of error
        '''
        tmp = int(len(self.transition_prob) // 2)

        self.pos += distance
        array = [x for x in range(self.pos - tmp, self.pos + tmp + 1)]
        self.pos = np.random.choice(array, p=self.transition_prob)
        if (self.pos < 0):
            self.pos = 0
        if (self.pos >= self.max_pos):
            self.pos = self.max_pos - 1

        return self.pos

    def sense(self):
        '''
        Get a measurement of current position
        '''

        tmp = int(len(self.transition_prob) // 2)

        array = [x for x in range(self.pos - tmp, self.pos + tmp + 1)]
        self.meas = np.random.choice(array, p=TRANSITION_PROB)

        if (self.meas < 0):
            self.meas = 0
        if (self.meas >= self.max_pos):
            self.meas = self.max_pos - 1

        return self.meas

    def transition_func(self, pos, distance):
        '''
        output new_pos with prob if the agent is at pos and move by distance
        '''

        perfect_transition_pos = pos + distance

        tmp = int(len(self.transition_prob) // 2)

        pos_prob_list = []
        for i, prob in enumerate(self.transition_prob):
            p = i - tmp + perfect_transition_pos
            if p < 0 or p >= self.max_pos:
                continue
            pos_prob_list.append((p, prob))

        return pos_prob_list

    def measurement_func(self, pos, meas):
        '''
        output prob of getting meas at pos 
        '''

        tmp = int(len(self.transition_prob) // 2)
        for i, prob in enumerate(self.meas_prob):
            p = i - tmp + pos
            if p < 0 or p >= self.max_pos:
                continue
            if meas == p:
                return prob

        return 0
