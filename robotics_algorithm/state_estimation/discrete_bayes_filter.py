#!/usr/bin/evn python

import numpy as np

EPSILON = 1e-5

class DiscreteBayesFilter():
    def __init__(self, state_space_size):
        self._state_space_size = state_space_size
        self.state_belief = np.full(state_space_size, 1 / state_space_size)

    def predict(self, control, transition_func):
        '''
        @param control, control 
        @param transition_func, a function that takes in state and control, output next state with  prob
        '''
        new_state_belief = np.zeros(self._state_space_size)

        # In discrete case, just add them up.
        for state in range(self._state_space_size):
            for next_state, prob in transition_func(state, control):
                new_state_belief[next_state] += prob * self.state_belief[state]
                # set small number to 0 to prevent NaN
                if new_state_belief[next_state] < EPSILON:
                    new_state_belief[next_state] = 0

        new_state_belief = new_state_belief / np.sum(new_state_belief)
        self.state_belief = new_state_belief

    def update(self, measurement, measurement_func):
        '''
        @param measurement, measurement 
        @param measurement_func, a function that takes in state and measurement, output probability of getting that measurement in that state
        '''

        for state in range(self._state_space_size):
            self.state_belief[state] *= measurement_func(state, measurement)
            # set small number to 0 to prevent NaN
            if self.state_belief[state] < EPSILON:
                self.state_belief[state] = 0

        self.state_belief = self.state_belief / np.sum(self.state_belief)

