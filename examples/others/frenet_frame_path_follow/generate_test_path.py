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


def _rotation_segment(x: float, y: float, yaw_start: float, yaw_end: float, n: int) -> np.ndarray:
    """Generate an in-place rotation (position fixed, yaw changes)."""
    yaws = np.linspace(yaw_start, yaw_end, n)
    return np.column_stack((np.full(n, x), np.full(n, y), yaws))


def generate_test_path() -> np.ndarray:
    """Return Nx3 array of [x, y, yaw] waypoints."""
    # --- Segment 1: Reeds-Shepp C|C|C ---
    # 1a: forward left arc from (5,0,0), R=1.5, sweep 60°
    rs1a = _arc_segment(x0=5.0, y0=5.0, yaw0=0.0, radius=1.5, arc_angle=np.pi / 3, n=25)
    # 1b: reverse right arc (same radius, opposite steering = right), sweep 60°
    #     right turn → centre is perpendicular-right, implemented as negative arc_angle
    end1a = rs1a[-1]
    rs1b = _reverse_arc_segment(x0=end1a[0], y0=end1a[1], yaw0=end1a[2], radius=1, arc_angle=np.pi / 2, n=25)

    # --- Segment 2: in-place rotation ---
    end_rs = rs1b[-1]
    seg2 = _rotation_segment(end_rs[0], end_rs[1], yaw_start=end_rs[2], yaw_end=end_rs[2] + np.pi / 2, n=20)

    # --- Segment 3: Dubins forward arc ---
    end2 = seg2[-1]
    seg3 = _arc_segment(x0=end2[0], y0=end2[1], yaw0=end2[2], radius=1.5, arc_angle=np.pi / 3, n=25)

    # Concatenate, dropping duplicate junction points.
    path = np.vstack([rs1a, rs1b[1:], seg2[1:], seg3[1:]])
    return path, len(rs1a), len(rs1b) - 1, 0, len(seg2) - 1, len(seg3) - 1


if __name__ == '__main__':
    path, n1a, n1b, n1c, n2, n3 = generate_test_path()

    # Save to JSON for use by the controller test.
    out_file = osp.join(osp.dirname(osp.abspath(__file__)), 'test_path_se2.json')
    with open(out_file, 'w') as f:
        json.dump(path.tolist(), f)
    print(f'Saved {len(path)} waypoints to {out_file}')

    # Visualise.
    i0, i1 = 0, n1a
    i2 = i1 + n1b
    i3 = i2 + n1c
    i4 = i3 + n2

    plt.figure(figsize=(7, 7))
    plt.plot(path[i0:i1, 0], path[i0:i1, 1], 'b-o', ms=3, label='RS 1a: fwd arc')
    plt.plot(path[i1:i2, 0], path[i1:i2, 1], 'm-o', ms=3, label='RS 1b: rev arc')
    plt.plot(path[i2:i3, 0], path[i2:i3, 1], 'c-o', ms=3, label='RS 1c: fwd arc')
    plt.plot(path[i3:i4, 0], path[i3:i4, 1], 'r-o', ms=5, label='Seg 2: in-place rotation')
    plt.plot(path[i4:, 0], path[i4:, 1], 'g-o', ms=3, label='Seg 3: Dubins fwd arc')

    # Draw heading arrows at every 10th point.
    for x, y, yaw in path[::8]:
        plt.arrow(
            x, y, 0.15 * np.cos(yaw), 0.15 * np.sin(yaw), head_width=0.05, head_length=0.08, fc='k', ec='k', alpha=0.5
        )

    plt.scatter(*path[0, :2], s=100, c='yellow', zorder=5, label='Start')
    plt.scatter(*path[-1, :2], s=100, c='red', zorder=5, label='Goal')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title('SE(2) test path: Reeds-Shepp C|C|C → in-place rotation → Dubins arc')
    plt.tight_layout()
    plt.show()
