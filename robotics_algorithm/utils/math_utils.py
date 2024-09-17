import numpy as np
# import math


def normalize_angle(angle):
    angle = (angle + 6 * np.pi) % (2 * np.pi)  # normalize theta to [-pi, pi]
    if angle > np.pi:
        angle = angle - 2 * np.pi

    return angle