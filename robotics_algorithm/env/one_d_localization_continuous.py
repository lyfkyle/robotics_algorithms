import copy
import math
import numpy as np
from numpy.random import randn

class OneDLocalizationContinuous(object):

    def __init__(self, initial_position=0, initial_velocity=1, measurement_var=0.1, process_var=0.1, dt = 1):
        '''
        measurement_variance - variance in measurement m^2
        process_variance - variance in process (m/s)^2
        '''
        self.position = initial_position
        self.velocity = initial_velocity
        self.dt = dt
        self.measurement_noise = math.sqrt(measurement_var)
        self.process_noise = math.sqrt(process_var)

        self.state = np.array([[self.position], [self.velocity]])

        self.A = np.array([[1, dt], [0, 1]], dtype=np.float32)
        self.B = np.array([[0.5 * dt * dt], [dt]], dtype=np.float32)
        self.R = np.array([[process_var, 0], [0, process_var]], dtype=np.float32)

        self.H = np.array([[1, 0]], dtype=np.float32)
        self.Q = np.array([[measurement_var]], dtype=np.float32)

    def control(self, accel):
        '''
        Compute new position and velocity of agent assume accel is applied for dt time
        xt+1 = Axt + But
        '''
        # compute new position based on acceleration. Add in some process noise
        new_state = self.A @ self.state + np.dot(self.B, accel) + np.array([[randn() * self.process_noise], [randn() * self.process_noise]])
        self.state = new_state

    def sense(self):
        '''
        Measure agent's position only.
        '''

        # simulate measuring the position with noise
        meas = self.H @ self.state + np.array([[randn() * self.measurement_noise]])
        return meas

    def control_and_sense(self, accel):
        self.control(accel)
        return self.sense()