import numpy as np

# import math


def normalize_angle(angle):
    # Normalize angle to the range (-pi, pi]
    while angle < 0:
        angle += 2 * np.pi
    angle = angle % (2 * np.pi)  # normalize theta to [-pi, pi]
    if angle > np.pi:
        angle = angle - 2 * np.pi

    return angle


def se2_distance(s1, s2, w_theta=0.1) -> float:
    dx = s2[0] - s1[0]
    dy = s2[1] - s1[1]
    dtheta = normalize_angle(s2[2] - s1[2])

    distance = np.sqrt(dx**2 + dy**2 + w_theta * dtheta**2)
    return distance


def se2_diff(s1, s2):
    dx = s2[0] - s1[0]
    dy = s2[1] - s1[1]
    dtheta = normalize_angle(s2[2] - s1[2])

    return np.array([dx, dy, dtheta])


def transform_to_frame(pose, pose_ref):
    """
    Transform an SE(2) pose into the frame of another SE(2) pose.

    Args:
        pose (np.ndarray): Target pose [x, y, theta].
        pose_ref (np.ndarray): Reference pose [x_ref, y_ref, theta_ref].

    Returns:
        np.ndarray: Transformed pose [x_rel, y_rel, theta_rel] in the reference frame.
    """
    x, y, theta = pose
    x_ref, y_ref, theta_ref = pose_ref

    # Relative translation
    dx = x - x_ref
    dy = y - y_ref

    # Rotate by the negative of theta_ref
    x_rel = dx * np.cos(-theta_ref) - dy * np.sin(-theta_ref)
    y_rel = dx * np.sin(-theta_ref) + dy * np.cos(-theta_ref)

    # Relative orientation
    theta_rel = normalize_angle(theta - theta_ref)

    return np.array([x_rel, y_rel, theta_rel])


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
