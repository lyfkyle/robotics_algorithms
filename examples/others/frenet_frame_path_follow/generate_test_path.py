"""Generate a test reference path consisting of:
  Segment 1 — Reeds-Shepp C|C|C:  forward arc → reverse arc → forward arc
  Segment 2 — In-place rotation
  Segment 3 — Dubins forward arc

The path is saved to test_path_se2.json and visualised.
"""

import json
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt


def _arc_segment(x0: float, y0: float, yaw0: float, radius: float, arc_angle: float, n: int) -> np.ndarray:
    """Generate a forward circular arc.

    Args:
        x0, y0, yaw0: start pose.
        radius: turning radius (positive = left/CCW turn).
        arc_angle: total arc angle in radians (positive = CCW).
        n: number of waypoints including start.
    """
    # Centre of curvature is perpendicular-left of heading.
    cx = x0 - radius * np.sin(yaw0)
    cy = y0 + radius * np.cos(yaw0)

    # Angle of start position relative to centre.
    theta0 = np.arctan2(y0 - cy, x0 - cx)

    states = []
    for i in range(n):
        phi = arc_angle * i / (n - 1)
        theta = theta0 + phi
        x = cx + radius * np.cos(theta)
        y = cy + radius * np.sin(theta)
        yaw = yaw0 + phi
        states.append([x, y, yaw])
    return np.array(states)


def _reverse_arc_segment(x0: float, y0: float, yaw0: float, radius: float, arc_angle: float, n: int) -> np.ndarray:
    """Generate a reverse (backward) circular arc.

    Driving backward with left steering sweeps CCW arc_angle but yaw
    changes in the opposite sense compared to forward driving.

    Args:
        x0, y0, yaw0: start pose.
        radius: turning radius (positive = left/CCW centre from vehicle).
        arc_angle: magnitude of arc swept in radians (positive = CCW centre movement).
        n: number of waypoints including start.
    """
    # Centre is still perpendicular-left, same as forward.
    cx = x0 - radius * np.sin(yaw0)
    cy = y0 + radius * np.cos(yaw0)
    theta0 = np.arctan2(y0 - cy, x0 - cx)

    states = []
    for i in range(n):
        # Going backward: position angle decreases, yaw also decreases.
        phi = arc_angle * i / (n - 1)
        theta = theta0 - phi
        x = cx + radius * np.cos(theta)
        y = cy + radius * np.sin(theta)
        yaw = yaw0 - phi
        states.append([x, y, yaw])
    return np.array(states)


def _straight_segment(x0: float, y0: float, yaw0: float, length: float, n: int) -> np.ndarray:
    """Generate a straight line segment. Length can be negative for reverse."""
    cos_yaw = np.cos(yaw0)
    sin_yaw = np.sin(yaw0)
    states = []
    for i in range(n):
        s = length * i / (n - 1)
        x = x0 + s * cos_yaw
        y = y0 + s * sin_yaw
        states.append([x, y, yaw0])
    return np.array(states)


def _rotation_segment(x: float, y: float, yaw_start: float, yaw_end: float, n: int) -> np.ndarray:
    """Generate an in-place rotation (position fixed, yaw changes)."""
    yaws = np.linspace(yaw_start, yaw_end, n)
    return np.column_stack((np.full(n, x), np.full(n, y), yaws))


def generate_test_path() -> np.ndarray:
    """Return Nx3 array of [x, y, yaw] waypoints."""
    # 1: forward left arc from (5,5,0), R=1.5, sweep 90°
    seg1 = _arc_segment(x0=5.0, y0=5.0, yaw0=0.0, radius=1.5, arc_angle=np.pi / 2, n=25)
    
    # 2 (CUSP 1): reverse right arc (opposite steering, reverse movement), sweep 90°
    end1 = seg1[-1]
    seg2 = _reverse_arc_segment(x0=end1[0], y0=end1[1], yaw0=end1[2], radius=1.5, arc_angle=np.pi / 2, n=25)

    # 3 (ROTATION 1): in-place rotation CCW of 90°
    end2 = seg2[-1]
    seg3 = _rotation_segment(end2[0], end2[1], yaw_start=end2[2], yaw_end=end2[2] + np.pi / 2, n=20)

    # 4: forward straight line of 1.2m
    end3 = seg3[-1]
    seg4 = _straight_segment(x0=end3[0], y0=end3[1], yaw0=end3[2], length=1.2, n=20)

    # 5 (CUSP 2): reverse straight line of -1.0m
    end4 = seg4[-1]
    seg5 = _straight_segment(x0=end4[0], y0=end4[1], yaw0=end4[2], length=-1.0, n=20)

    # 6 (ROTATION 2): in-place rotation CW of 180°
    end5 = seg5[-1]
    seg6 = _rotation_segment(end5[0], end5[1], yaw_start=end5[2], yaw_end=end5[2] - np.pi, n=25)

    # 7: forward left arc of 60°
    end6 = seg6[-1]
    seg7 = _arc_segment(x0=end6[0], y0=end6[1], yaw0=end6[2], radius=1.0, arc_angle=np.pi / 3, n=20)

    # Concatenate, dropping duplicate junction points.
    path = np.vstack([seg1, seg2[1:], seg3[1:], seg4[1:], seg5[1:], seg6[1:], seg7[1:]])
    return path, len(seg1), len(seg2) - 1, len(seg3) - 1, len(seg4) - 1, len(seg5) - 1, len(seg6) - 1, len(seg7) - 1


if __name__ == '__main__':
    path = generate_test_path()[0]

    # Save to JSON for use by the controller test.
    out_file = osp.join(osp.dirname(osp.abspath(__file__)), 'test_path_se2.json')
    with open(out_file, 'w') as f:
        json.dump(path.tolist(), f)
    print(f'Saved {len(path)} waypoints to {out_file}')

    # Visualise.
    plt.figure(figsize=(7, 7))
    plt.plot(path[:, 0], path[:, 1], 'g-o', ms=3, label='Test Path')

    # Draw heading arrows at every 5th point.
    for x, y, yaw in path[::5]:
        plt.arrow(
            x, y, 0.15 * np.cos(yaw), 0.15 * np.sin(yaw), head_width=0.05, head_length=0.08, fc='k', ec='k', alpha=0.5
        )

    plt.scatter(*path[0, :2], s=100, c='yellow', zorder=5, label='Start')
    plt.scatter(*path[-1, :2], s=100, c='red', zorder=5, label='Goal')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title('Challenging Zigzag and In-Place Rotation SE(2) Path')
    plt.tight_layout()
    plt.show()
