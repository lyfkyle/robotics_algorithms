import numpy as np
from scipy.spatial.transform import Rotation

import robotics_algorithm.utils.math_utils as math_utils


def get_R(rotvec):
    """Get a 3x3 rotation matrix from a rotation vector."""
    return Rotation.from_rotvec(rotvec).as_matrix()


def get_T(pos, rotvec):
    """Get a 4x4 SE(3) transformation matrix for translation (x, y, z) and rotation given by a rotation vector."""
    T = np.eye(4)
    T[0:3, 0:3] = Rotation.from_rotvec(rotvec).as_matrix()
    T[0:3, 3] = pos
    return T


def se3_to_planar_pose(T):
    """Extract a planar SE(2) pose (x, y, theta) from a 4x4 SE(3) matrix, assuming rotation about z only."""
    x, y = T[0, 3], T[1, 3]
    theta = math_utils.normalize_angle(Rotation.from_matrix(T[0:3, 0:3]).as_rotvec()[2])
    return np.array([x, y, theta])


def skew_symmetric(v):
    """Get the skew-symmetric matrix of a 3D vector."""
    return np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )


def get_twist_matrix(twist):
    """Get the 4x4 matrix representation of a 6D twist vector."""
    omega = twist[0:3]
    v = twist[3:6]
    twist_matrix = np.zeros((4, 4))
    twist_matrix[0:3, 0:3] = skew_symmetric(omega)
    twist_matrix[0:3, 3] = v
    return twist_matrix


def twist_matrix_to_twist(twist_matrix):
    """Extract the 6D twist vector [wx, wy, wz, vx, vy, vz] from its 4x4 se(3) matrix representation."""
    R = twist_matrix[0:3, 0:3]
    omega = np.array([R[2, 1], R[0, 2], R[1, 0]])
    v = twist_matrix[0:3, 3]
    return np.concatenate([omega, v])
