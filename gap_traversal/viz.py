"""Live matplotlib visualization for the gap traversal demo."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

from gap_traversal.world import PLATFORM_A, PLATFORM_B


class Visualizer:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(9, 6))

    def render(self, state, corners, sensors, phase, candidates=None, crossing_idx=None):
        self.ax.clear()

        for rect, color in [(PLATFORM_A, 'tab:green'), (PLATFORM_B, 'tab:green')]:
            x_min, y_min, x_max, y_max = rect
            self.ax.add_patch(
                Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, facecolor=color, alpha=0.3, edgecolor='black')
            )

        if candidates:
            for idx, cand in enumerate(candidates):
                start = cand['start_pos']
                end = cand['end_pos']
                mid = (start + end) / 2.0
                is_current = idx == crossing_idx
                color = 'tab:orange' if is_current else 'tab:purple'
                self.ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=3, alpha=0.8, zorder=4)
                self.ax.scatter(mid[0], mid[1], marker='x', c=color, s=80, zorder=5)
                self.ax.annotate(str(idx), mid, textcoords='offset points', xytext=(5, 5), color=color)

        self.ax.add_patch(Polygon(corners, closed=True, facecolor='tab:blue', edgecolor='black', alpha=0.8))

        for (cx, cy), triggered in zip(corners, sensors):
            self.ax.scatter(cx, cy, c='red' if triggered else 'lime', s=60, zorder=5, edgecolors='black')

        self.ax.arrow(
            state[0],
            state[1],
            0.3 * np.cos(state[2]),
            0.3 * np.sin(state[2]),
            head_width=0.1,
            color='black',
        )

        title = f'phase={phase}'
        if candidates is not None:
            title += f' | candidates found={len(candidates)}'
        self.ax.set_title(title)

        x_min = min(PLATFORM_A[0], PLATFORM_B[0]) - 1
        x_max = max(PLATFORM_A[2], PLATFORM_B[2]) + 1
        y_min = min(PLATFORM_A[1], PLATFORM_B[1]) - 1
        y_max = max(PLATFORM_A[3], PLATFORM_B[3]) + 1
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_aspect('equal')
        self.ax.grid(True)

        plt.pause(0.001)
