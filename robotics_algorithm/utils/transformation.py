import numpy as np


def rotation_matrix_2d(theta):
    """Get 2D rotation matrix for angle theta."""
    return np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )


def transformation_matrix_2d(x, y, theta):
    """Get 2D transformation matrix for rotation theta and translation (x, y)."""
    T = np.zeros((3, 3))
    T[0:2, 0:2] = rotation_matrix_2d(theta)
    T[0:2, 2] = np.array([x, y])
    T[2, 2] = 1
    return T
