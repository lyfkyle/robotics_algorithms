import numpy as np

# import math


def normalize_angle(angle):
    angle = (angle + 6 * np.pi) % (2 * np.pi)  # normalize theta to [-pi, pi]
    if angle > np.pi:
        angle = angle - 2 * np.pi

    return angle


def se2_distance(state1: np.ndarray, state2: np.ndarray, w_theta: float = 0.2) -> float:
    """Compute SE(2) distance between two [x, y, yaw] states.

    Uses d = sqrt(dx^2 + dy^2 + (w_theta * d_yaw)^2) so that in-place rotations
    contribute to the distance metric.

    Args:
        state1: Array of shape (3,) with columns [x, y, yaw].
        state2: Array of shape (3,) with columns [x, y, yaw].
        w_theta: Scaling factor [m/rad] for angular contribution. Defaults to 0.2.

    Returns:
        SE(2) distance between the two states.
    """
    dx = state1[0] - state2[0]
    dy = state1[1] - state2[1]
    dyaw = normalize_angle(state1[2] - state2[2])
    return np.sqrt(dx**2 + dy**2 + (w_theta * dyaw) ** 2)


def se2_arc_lengths(states: np.ndarray, w_theta: float = 0.2) -> np.ndarray:
    """Compute cumulative SE(2) arc-lengths along a sequence of [x, y, yaw] states.

    Uses d = sqrt(dx^2 + dy^2 + (w_theta * d_yaw)^2) so that in-place rotations
    contribute to the arc-length parameter.

    Args:
        states: Array of shape (N, 3) with columns [x, y, yaw].
        w_theta: Scaling factor [m/rad] for angular contribution. Defaults to 0.2.

    Returns:
        Array of shape (N,) with cumulative arc-lengths starting at 0.
    """
    if states.shape[0] <= 1:
        return np.zeros(states.shape[0], dtype=float)
    dxy = np.diff(states[:, :2], axis=0)
    dyaw = np.diff(np.unwrap(states[:, 2]), axis=0)
    seg_lengths = np.sqrt(np.sum(dxy**2, axis=1) + (w_theta * dyaw) ** 2)
    return np.concatenate(([0.0], np.cumsum(seg_lengths)))


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
