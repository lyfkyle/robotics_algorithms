"""World model: two rectangular floor platforms separated by a gap on one side, plus a
4-corner drop sensor model.

Platform A and Platform B are fully separate floor islands. Only the sides facing each other
are separated by a small crossable gap; the other 3 sides of each platform are genuine cliffs
(no floor beyond them at all).
"""

import numpy as np

# Platform A: x in [-4, 0], y in [-2, 2]
PLATFORM_A = (-4.0, -2.0, 0.0, 2.0)  # (x_min, y_min, x_max, y_max)

# Gap width between the facing edges of A and B. Must be physically crossable by tilting the
# robot: with the demo's footprint (length=0.25, width=0.45), the best achievable bridging
# clearance (see GapTraversalController's optimal-tilt calculation) is ~0.167 m, so this is kept
# comfortably below that.
GAP_WIDTH = 0.12

# Platform B sits immediately across the gap from A's east edge, same y-range.
PLATFORM_B = (PLATFORM_A[2] + GAP_WIDTH, PLATFORM_A[1], PLATFORM_A[2] + GAP_WIDTH + 4.0, PLATFORM_A[3])


def _inside_rect(x: float, y: float, rect: tuple) -> bool:
    x_min, y_min, x_max, y_max = rect
    return x_min <= x <= x_max and y_min <= y <= y_max


def has_floor(x: float, y: float) -> bool:
    """Return True if the given world point has floor beneath it."""
    return _inside_rect(x, y, PLATFORM_A) or _inside_rect(x, y, PLATFORM_B)


def read_drop_sensors(corners: np.ndarray) -> np.ndarray:
    """Read the 4 corner drop sensors.

    Args:
        corners (np.ndarray): shape (4, 2), world-frame corner positions.

    Returns:
        np.ndarray of 4 bools; True means the corner has no floor beneath it (drop triggered).
    """
    return np.array([not has_floor(cx, cy) for cx, cy in corners])
