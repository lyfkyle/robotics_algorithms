"""Autonomous gap discovery and crossing demo for a differential drive robot.

The robot starts at the center of Platform A, not knowing where any gap is. It:
  1. WANDERs in a random direction until a drop sensor detects an edge.
  2. ALIGNs itself so it is parallel to that edge (exactly one lateral pair of corners
     triggered).
  3. TRACEs the full perimeter of the platform (bug-style edge following), recording each
     straight edge as a gap candidate, without attempting to cross any of them.
  4. Tries to CROSS each candidate edge, in the order discovered, by tilting itself and
     creeping a single leading corner across the gap - retreating immediately whenever 2+ drop
     sensors trigger at once (safety rule) - until it finds floor on the far side (success) or
     concludes the candidate is a true cliff (no floor within a bounded probe distance) and
     moves on to the next candidate.

Run with: python test_zone_transition.py
"""

import matplotlib.pyplot as plt
import numpy as np

from gap_traversal.behavior import DT, GapTraversalController
from gap_traversal.robot import DiffDriveRect
from gap_traversal.viz import Visualizer
from gap_traversal.world import PLATFORM_A, read_drop_sensors

MAX_STEPS = 20000


def main():
    # length = fore-aft extent (along heading), width = lateral extent. width > length so the
    # front/back edges are the long edges and heading is normal to them (needed so the leading
    # edge used to probe the gap has enough spread between its two corners).
    robot = DiffDriveRect(wheel_radius=0.03, wheel_dist=0.15, length=0.25, width=0.45, dt=DT)

    start_x = (PLATFORM_A[0] + PLATFORM_A[2]) / 2.0
    start_y = (PLATFORM_A[1] + PLATFORM_A[3]) / 2.0
    start_theta = np.random.uniform(-np.pi, np.pi)
    state = np.array([start_x, start_y, start_theta])

    controller = GapTraversalController()
    viz = Visualizer()

    print(f'Starting at ({start_x:.2f}, {start_y:.2f}), heading {np.degrees(start_theta):.1f} deg')

    last_phase = controller.phase
    for step in range(MAX_STEPS):
        state = controller.step(robot, state)

        if controller.log:
            print(f'[step {step}] {controller.log}')
        elif controller.phase != last_phase:
            print(f'[step {step}] -> {controller.phase}')
        last_phase = controller.phase

        corners = robot.get_corner_positions(state)
        sensors = read_drop_sensors(corners)
        viz.render(
            state,
            corners,
            sensors,
            controller.phase,
            candidates=controller.candidates,
            crossing_idx=controller.crossing_idx,
        )

        if controller.phase == 'DONE':
            break

    print(f'\nResult: {controller.result}')
    print(f'Candidates discovered: {len(controller.candidates)}')
    print('Done. Close the plot window to exit.')

    plt.ioff()
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
