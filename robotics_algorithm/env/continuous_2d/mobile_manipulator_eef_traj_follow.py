import numpy as np
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Affine2D
from typing_extensions import override

from robotics_algorithm.env.base_env import ContinuousSpace, DeterministicEnv, FullyObservableEnv
from robotics_algorithm.robot.mobile_manipulator import MobileManipulator2Link


class MMEefTrajFollowEnv(DeterministicEnv, FullyObservableEnv):
    """A 2D mobile manipulator (differential-drive base + planar 2-link arm) tracking a reference
    end-effector trajectory.

    State: [x, y, theta, q1, q2]
    Action: [v, omega, q1_dot, q2_dot]

    The reward is the (negative) tracking error between the end-effector position and the
    reference trajectory waypoint, plus a small control effort penalty. The reference waypoint is
    selected using an internal counter that cycles through `ref_eef_traj` once per `horizon` calls
    to `reward_func`, matching how `DirectShooting`/`DirectCollocation` invoke it (exactly
    `horizon` calls, in order, per cost function evaluation).
    """

    BASE_RADIUS = 0.1
    LINK_WIDTH = 0.01

    def __init__(
        self,
        start_state: np.ndarray,
        ref_eef_traj: np.ndarray,
        link1_length: float = 0.3,
        link2_length: float = 0.2,
        dt: float = 0.1,
        ctrl_cost_w: float = 1e-3,
    ):
        super().__init__()

        self.robot_model = MobileManipulator2Link(link1_length, link2_length, dt=dt)

        self.start_state = np.asarray(start_state, dtype=float)
        self.goal_state = self.start_state.copy()  # the reference trajectory is a closed loop

        # ref_eef_traj[0] corresponds to the start state, ref_eef_traj[i] is the target for the
        # i-th transition (i.e. the state reached after applying the i-th action).
        self.ref_eef_traj = np.asarray(ref_eef_traj)
        self.horizon = len(self.ref_eef_traj) - 1
        self.ctrl_cost_w = ctrl_cost_w
        self._call_cnt = 0

        self.state_space = ContinuousSpace(
            low=[-10.0, -10.0, -3 * np.pi, -2 * np.pi, -2 * np.pi],
            high=[10.0, 10.0, 3 * np.pi, 2 * np.pi, 2 * np.pi],
        )
        self.action_space = ContinuousSpace(low=[-0.5, -1.5, -1.5, -1.5], high=[0.5, 1.5, 1.5, 1.5])

    @override
    def reset(self):
        self.cur_state = self.start_state.copy()
        self.step_cnt = 0
        self._call_cnt = 0

        return self.cur_state, {}

    @override
    def state_transition_func(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return self.robot_model.control(state, action)

    @override
    def reward_func(self, state: np.ndarray, action: np.ndarray = None, new_state: np.ndarray = None) -> float:
        ref_idx = self._call_cnt % self.horizon + 1
        self._call_cnt += 1

        eef_pos = self.robot_model.eef_pos(new_state)
        track_cost = np.sum((eef_pos - self.ref_eef_traj[ref_idx]) ** 2)
        ctrl_cost = self.ctrl_cost_w * np.dot(action, action)

        return -(track_cost + ctrl_cost)

    def _draw_link(self, ax, start_pos, angle, length, width, color):
        """Draw a thin rectangle representing a link, from `start_pos` along `angle` for `length`."""
        rect = Rectangle((0, -width / 2), length, width, facecolor=color, edgecolor='k')
        transform = Affine2D().rotate(angle).translate(*start_pos) + ax.transData
        rect.set_transform(transform)
        ax.add_patch(rect)

    @override
    def render(self, ax, state: np.ndarray):
        """Draw a snapshot of the mobile manipulator (base + both arm links) at `state`.

        Args:
            ax: matplotlib axes to draw on.
            state (np.ndarray): [x, y, theta, q1, q2]
        """
        x, y, theta, q1, q2 = state
        base_pos = np.array([x, y])
        _, elbow_pos, _ = self.robot_model.fk(state)

        # Base as a circle, with a heading tick to show orientation.
        ax.add_patch(Circle(base_pos, self.BASE_RADIUS, facecolor='gray', edgecolor='k'))
        heading_tip = base_pos + self.BASE_RADIUS * np.array([np.cos(theta), np.sin(theta)])
        ax.plot([base_pos[0], heading_tip[0]], [base_pos[1], heading_tip[1]], 'k-')

        # Arm links as thin rectangles.
        self._draw_link(ax, base_pos, theta + q1, self.robot_model.l1, self.LINK_WIDTH, 'orange')
        self._draw_link(ax, elbow_pos, theta + q1 + q2, self.robot_model.l2, self.LINK_WIDTH, 'green')
