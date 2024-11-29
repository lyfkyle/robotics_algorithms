import numpy as np

# import math


def normalize_angle(angle):
    angle = (angle + 6 * np.pi) % (2 * np.pi)  # normalize theta to [-pi, pi]
    if angle > np.pi:
        angle = angle - 2 * np.pi

    return angle


def smooth(scalars: list[float], weight: float = 0.5) -> list[float]:  # Weight between 0 and 1
    """
    Smooth a list of scalars by applying exponential smoothing with a given weight.

    Args:
        scalars: List of scalars to be smoothed.
        weight: Weight between 0 and 1 for exponential smoothing.

    Return:
        List of smoothed values.
    """
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed
