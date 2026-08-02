"""Reactive behaviour for autonomous gap discovery and crossing.

Phases:
  WANDER    - drive in a random initial direction until any drop sensor triggers (edge found).
  ALIGN     - find the heading at which the robot body is truly tangent to the local edge, using
              a simple retreat/rotate/forward search cycle: back away from whatever corner(s)
              just triggered, rotate a small step counter-clockwise, then creep forward again
              watching the sensors. For a straight edge, the front-right and rear-right corners
              are offset only along the heading axis, so they trigger at the same forward-creep
              distance (same tick) exactly when the heading is parallel to the edge; any heading
              error makes one trigger noticeably before the other, or makes a left-side corner
              trigger instead. So each forward-creep attempt either confirms front-right/rear-
              right triggered together (within a small tick tolerance, no other corner involved)
              - in which case a final small counter-clockwise nudge is applied and the phase
              hands off to TRACE - or it fails (wrong corner triggered, corners too far apart in
              time, or nothing re-triggers at all), in which case the cycle repeats: retreat,
              rotate a bit further counter-clockwise, creep forward again.
  TRACE     - bug-style edge following around the whole platform perimeter (without crossing
              anything), recording each straight edge as a gap candidate. When a corner of the
              platform is reached, the robot first backs straight up - before pivoting in place -
              by (width - length) / 2 (clamped to >= 0). This is exactly the distance needed so
              that, once the 90 degree in-place pivot completes, the lead corner (front-right)
              sits right on the new edge instead of hovering short of or past it: the corner is
              first detected one lead-corner-length (length/2) past the platform's true corner
              point, but after pivoting the lead corner ends up offset by half the width instead,
              so backing up by the difference before pivoting reconciles the two. The pose right
              after this pivot is still only an approximate guess at the new edge, though - TRACE
              records the segment's true start position only once its own steer-into-the-edge
              corrections cause a drop sensor to actually be observed transitioning from over-
              the-void to detecting floor, and uses that sensor's corner position as the
              geometrically accurate start of the edge, rather than the assumed pivot pose. A new
              straight edge is then traced. Each newly completed edge is compared against every
              previously recorded candidate by its infinite-line signature (heading mod pi, plus
              perpendicular offset from the origin); if it matches one already discovered, the
              robot has come back around onto an edge it has already recorded, so the perimeter
              loop is closed and no new edge remains to discover - the recorded candidates (not
              counting the duplicate) are final. This works for any convex polygon-shaped
              platform, not just rectangles.
  CROSS     - visit each recorded candidate edge in discovery order and attempt to cross it by
              first backing off to a pose fully on solid ground (all sensors reading floor), then
              rotating to a heading tilted from the outward (edge-normal) direction by an angle
              that is calculated - not searched for - to make the crossing safe. With the tilt
              angle alpha (measured from the outward-normal direction) and footprint length L
              (fore-aft) / width W (lateral), translating forward without further rotation moves
              each corner's position along the crossing axis at a fixed offset from the robot's
              center: front-right leads, then rear-right, then front-left, then rear-left trails
              (in that order once alpha exceeds arctan(L/W)). The forward distance between one
              corner finishing its crossing and the next one starting is exactly L*cos(alpha) or
              W*sin(alpha) - L*cos(alpha); balancing these two (alpha = arctan(2L/W)) maximizes
              the minimum such clearance, which is the largest gap width this footprint can ever
              bridge without 2 corners hanging over the void at once. Creeping forward from the
              backed-off pose then naturally crosses the gap one corner at a time - front-right,
              then rear-right, both carried across by plain forward translation. Once front-left
              also finishes crossing (its sensor goes back to floor-detected), only rear-left is
              left hanging over the gap; from that point on the controller stops translating and
              instead rotates the robot in place (anti-clockwise), since the other 3 corners are
              already solidly grounded on the far platform and rotating is what swings rear-left
              across the remaining stretch. The crossing is complete once rear-left's sensor
              reads floor again. The 2+-sensor safety rule is kept only as a defensive fallback
              (it should not normally trigger). If the leading corner never finds floor within a
              bounded probe distance, the candidate is a true cliff and the controller moves on to
              the next one.
  DONE      - terminal phase, `result` holds 'SUCCESS' or 'FAILURE'.
"""

import numpy as np

from gap_traversal.robot import normalize_angle
from gap_traversal.world import read_drop_sensors

# Corner index convention (matches DiffDriveRect.get_corner_positions): 0=FL, 1=FR, 2=RR, 3=RL

DT = 0.1  # must match the DiffDriveRect instance's dt used by the main loop

WANDER_SPEED = 0.4

ALIGN_RETREAT_TICKS = 5  # ticks to back away from a triggered corner before rotating further
ALIGN_ROTATE_ANG_VEL = np.radians(20)  # in-place counter-clockwise search rotation speed
ALIGN_ROTATE_TICKS = 5  # ticks of rotation applied per search cycle (small heading increment)
ALIGN_FORWARD_MAX_TICKS = 20  # cap on forward creep per cycle before giving up and rotating more
ALIGN_SYNC_TICK_TOLERANCE = 2  # max ticks apart FR/RR may trigger and still count as "same time"
ALIGN_FINAL_ROTATE_TICKS = 3  # small extra counter-clockwise nudge once aligned, before TRACE

CREEP_SPEED = 0.3
TRACE_CORRECT_ANG_VEL = np.radians(25)
REACQUIRE_ANG_VEL = np.radians(40)

CORNER_TURN_ANG_VEL = np.radians(45)
CORNER_TURN_ANGLE = np.pi / 2

TILT_ROTATE_ANG_VEL = np.radians(30)
CROSS_BACKOFF_MARGIN = 0.1  # extra clearance beyond the footprint's half-diagonal when backing off

RETREAT_SPEED = 0.3
FULL_RETREAT_TICKS = 15

MAX_PROBE_DISTANCE = 1.0

LOOP_LINE_ANGLE_TOLERANCE = np.radians(15)  # max heading difference (mod pi) to call two edges the same line
LOOP_LINE_OFFSET_TOLERANCE = 0.2  # max perpendicular offset difference to call two edges the same line

ANGLE_EPS = np.radians(3)


class GapTraversalController:
    """Reactive controller implementing wander -> align -> trace -> cross behaviour."""

    def __init__(self):
        self.phase = 'WANDER'
        self.result = None

        # ALIGN bookkeeping (retreat / rotate CCW / creep forward search cycle)
        self._align_stage = 'RETREAT'
        self._align_ticks_remaining = ALIGN_RETREAT_TICKS
        self._forward_ticks = 0
        self._lead_trigger_tick = None
        self._trail_trigger_tick = None

        # TRACE bookkeeping
        self.trace_side = None
        self.sign = None
        self._segment_start_corner = None
        self._segment_start_pending = False
        self._prev_trace_sensors = None
        self.candidates = []
        self._corner_stage = 'BACKUP'
        self._corner_ticks_remaining = 0

        # CROSS bookkeeping
        self.crossing_idx = 0
        self._outward_heading = None
        self._target_heading = None
        self._creep_start_pos = None
        self._cross_prev_sensors = None
        self._cross_final_rotate = False
        self._cross_rl_triggered = False
        self._retreat_ticks_remaining = 0

        self.log = None

    def step(self, robot, state: np.ndarray) -> np.ndarray:
        """Compute the next state given the current state. Owns all state advancement,
        including phase transitions and (for CROSS setup) direct pose resets.

        Args:
            robot: a DiffDriveRect instance.
            state (np.ndarray): current [x, y, theta].

        Returns:
            new state [x, y, theta].
        """
        self.log = None

        if self.phase == 'WANDER':
            return self._step_wander(robot, state)
        elif self.phase == 'ALIGN':
            return self._step_align(robot, state)
        elif self.phase == 'TRACE':
            return self._step_trace(robot, state)
        elif self.phase == 'CORNERING':
            return self._step_cornering(robot, state)
        elif self.phase == 'CROSS_SETUP':
            return self._step_cross_setup(robot, state)
        elif self.phase == 'CROSS_TILT_ROTATE':
            return self._step_cross_tilt_rotate(robot, state)
        elif self.phase == 'CROSS_CREEP':
            return self._step_cross_creep(robot, state)
        elif self.phase == 'CROSS_RETREAT':
            return self._step_cross_retreat(robot, state)
        else:  # DONE
            return state

    # ---------------------------------------------------------------- WANDER
    def _step_wander(self, robot, state):
        corners = robot.get_corner_positions(state)
        sensors = read_drop_sensors(corners)
        if sensors.any():
            # The robot always traces the edge with the void on its right side (FR+RR), no
            # matter which corner(s) tripped first. ALIGN's pivot only ever rotates clockwise
            # (driving the left wheel only, about the fixed right wheel), so if the left
            # corners tripped first, that clockwise sweep is exactly what brings the right pair
            # around onto the edge next - never search for a 'left' pattern here, since the
            # pivot direction could never converge on it.
            self.trace_side = 'right'
            self.sign = 1.0
            self.phase = 'ALIGN'
            self._align_stage = 'RETREAT'
            self._align_ticks_remaining = ALIGN_RETREAT_TICKS
            self.log = 'Edge found -> ALIGN (retreat, then rotate CCW and creep forward to test)'
            return state

        return robot.control(state, np.array([WANDER_SPEED, 0.0]))

    # ----------------------------------------------------------------- ALIGN
    def _step_align(self, robot, state):
        if self._align_stage == 'RETREAT':
            return self._align_stage_retreat(robot, state)
        elif self._align_stage == 'ROTATE':
            return self._align_stage_rotate(robot, state)
        elif self._align_stage == 'FORWARD':
            return self._align_stage_forward(robot, state)
        else:  # FINAL_ROTATE
            return self._align_stage_final_rotate(robot, state)

    def _align_lead_trail_other(self):
        """Return (lead_idx, trail_idx, other_a_idx, other_b_idx) for the current trace_side."""
        if self.trace_side == 'right':
            return 1, 2, 0, 3
        return 0, 3, 1, 2

    def _align_stage_retreat(self, robot, state):
        if self._align_ticks_remaining <= 0:
            self._align_ticks_remaining = ALIGN_ROTATE_TICKS
            self._align_stage = 'ROTATE'
            self.log = 'Retreated from edge -> rotate CCW a bit further'
            return state

        self._align_ticks_remaining -= 1
        return robot.control(state, np.array([-RETREAT_SPEED, 0.0]))

    def _align_stage_rotate(self, robot, state):
        if self._align_ticks_remaining <= 0:
            self._forward_ticks = 0
            self._lead_trigger_tick = None
            self._trail_trigger_tick = None
            self._align_stage = 'FORWARD'
            self.log = 'Rotated -> creep forward to test alignment'
            return state

        self._align_ticks_remaining -= 1
        return robot.control(state, np.array([0.0, ALIGN_ROTATE_ANG_VEL]))

    def _align_stage_forward(self, robot, state):
        corners = robot.get_corner_positions(state)
        sensors = read_drop_sensors(corners)
        lead_idx, trail_idx, other_a, other_b = self._align_lead_trail_other()

        if sensors[other_a] or sensors[other_b]:
            # Wrong (left-side) corner dropped - overshot rotation the wrong way.
            self._align_ticks_remaining = ALIGN_RETREAT_TICKS
            self._align_stage = 'RETREAT'
            self.log = 'Wrong corner dropped while creeping forward -> retreat & rotate further'
            return state

        if sensors[lead_idx] and self._lead_trigger_tick is None:
            self._lead_trigger_tick = self._forward_ticks
        if sensors[trail_idx] and self._trail_trigger_tick is None:
            self._trail_trigger_tick = self._forward_ticks

        if self._lead_trigger_tick is not None and self._trail_trigger_tick is not None:
            if abs(self._lead_trigger_tick - self._trail_trigger_tick) <= ALIGN_SYNC_TICK_TOLERANCE:
                self._align_ticks_remaining = ALIGN_FINAL_ROTATE_TICKS
                self._align_stage = 'FINAL_ROTATE'
                self.log = 'Front-right/rear-right triggered in sync -> final CCW nudge, then TRACE'
            else:
                self._align_ticks_remaining = ALIGN_RETREAT_TICKS
                self._align_stage = 'RETREAT'
                self.log = 'Front-right/rear-right triggered too far apart -> retreat & rotate further'
            return state

        if self._forward_ticks >= ALIGN_FORWARD_MAX_TICKS:
            # Crept forward this far without re-triggering anything else - good enough, move on.
            self._start_new_trace_segment()
            self.candidates = []
            self.phase = 'TRACE'
            self.log = 'No edge re-triggered while creeping forward -> good enough, TRACE'
            return state

        self._forward_ticks += 1
        return robot.control(state, np.array([CREEP_SPEED, 0.0]))

    def _align_stage_final_rotate(self, robot, state):
        if self._align_ticks_remaining <= 0:
            self._start_new_trace_segment()
            self.candidates = []
            self.phase = 'TRACE'
            self.log = f'Fully aligned (side={self.trace_side}) -> TRACE'
            return state

        self._align_ticks_remaining -= 1
        return robot.control(state, np.array([0.0, ALIGN_ROTATE_ANG_VEL]))

    def _start_new_trace_segment(self):
        """Mark a new TRACE segment as starting, without yet knowing its precise start position.

        The pose right after ALIGN/CORNERING is only an approximate guess at where the new edge
        begins. TRACE itself will steer slightly into the edge until a drop sensor is actually
        observed transitioning to detecting floor; that observed corner position is a far more
        accurate marker of the true edge start, and is captured in `_step_trace` instead.
        """
        self._segment_start_corner = None
        self._segment_start_pending = True
        self._prev_trace_sensors = None

    # ----------------------------------------------------------------- TRACE
    def _step_trace(self, robot, state):
        corners = robot.get_corner_positions(state)
        sensors = read_drop_sensors(corners)
        count = sensors.sum()

        lead_idx, trail_idx = (1, 2) if self.trace_side == 'right' else (0, 3)

        if self._segment_start_pending:
            if self._prev_trace_sensors is not None:
                for idx in (lead_idx, trail_idx):
                    if self._prev_trace_sensors[idx] and not sensors[idx]:
                        # This corner just transitioned from over-the-void to floor detected -
                        # its current position is a sensor-verified marker of the edge start.
                        self._segment_start_corner = corners[idx].copy()
                        self._segment_start_pending = False
                        break
            self._prev_trace_sensors = sensors.copy()

        if count >= 3:
            self._finalize_segment(robot, state)
            return state

        if sensors[lead_idx] and sensors[trail_idx]:
            action = np.array([CREEP_SPEED, np.radians(5)])  # small CCW nudge to keep the edge on the right
        elif sensors[lead_idx] and not sensors[trail_idx]:
            action = np.array([CREEP_SPEED, self.sign * TRACE_CORRECT_ANG_VEL])
        elif sensors[trail_idx] and not sensors[lead_idx]:
            action = np.array([CREEP_SPEED, -self.sign * TRACE_CORRECT_ANG_VEL])
        else:
            action = np.array([CREEP_SPEED * 0.5, -self.sign * REACQUIRE_ANG_VEL])

        return robot.control(state, action)

    def _finalize_segment(self, robot, state):
        lead_idx, _trail_idx, _other_a, _other_b = self._align_lead_trail_other()
        end_corner = robot.get_corner_positions(state)[lead_idx]

        # Rare fallback: a sensor-verified start was never observed before this segment already
        # finished (e.g. the very next corner arrived immediately) - fall back to the current lead
        # corner rather than leaving start_pos unset.
        start_corner = self._segment_start_corner if self._segment_start_corner is not None else end_corner

        candidate = {
            'start_pos': start_corner.copy(),
            'end_pos': end_corner.copy(),
            'heading': state[2],
            'trace_side': self.trace_side,
        }

        if any(self._same_line(candidate, existing) for existing in self.candidates):
            self.crossing_idx = 0
            self.phase = 'CROSS_SETUP'
            self.log = f'Edge matches one already discovered -> perimeter loop closed, {len(self.candidates)} candidate(s) -> CROSS'
            return

        self.candidates.append(candidate)
        self.log = f'Corner detected -> {len(self.candidates)} candidate(s) recorded so far'
        # Back straight up before pivoting so the lead (front-right) corner lands exactly on the
        # new edge once the 90 degree pivot completes (see module docstring for the derivation).
        backup_dist = max((robot.width - robot.length) / 2.0, 0.0)
        self._corner_ticks_remaining = round(backup_dist / (RETREAT_SPEED * DT))
        self._corner_stage = 'BACKUP'
        self.phase = 'CORNERING'

    @staticmethod
    def _line_signature(candidate):
        """Return (angle_mod_pi, perpendicular_offset) identifying the infinite line an edge lies on."""
        theta_mod = candidate['heading'] % np.pi
        nx, ny = -np.sin(theta_mod), np.cos(theta_mod)
        px, py = candidate['start_pos']
        offset = nx * px + ny * py
        return theta_mod, offset

    @classmethod
    def _same_line(cls, candidate_a, candidate_b):
        angle_a, offset_a = cls._line_signature(candidate_a)
        angle_b, offset_b = cls._line_signature(candidate_b)
        angle_diff = abs(normalize_angle(angle_a - angle_b))
        angle_diff = min(angle_diff, abs(np.pi - angle_diff))  # angles are mod pi, so pi-wraparound also counts as 0
        return angle_diff <= LOOP_LINE_ANGLE_TOLERANCE and abs(offset_a - offset_b) <= LOOP_LINE_OFFSET_TOLERANCE

    # ------------------------------------------------------------- CORNERING
    def _step_cornering(self, robot, state):
        if self._corner_stage == 'BACKUP':
            return self._step_cornering_backup(robot, state)
        return self._step_cornering_rotate(robot, state)

    def _step_cornering_backup(self, robot, state):
        if self._corner_ticks_remaining <= 0:
            self._corner_ticks_remaining = round(CORNER_TURN_ANGLE / (CORNER_TURN_ANG_VEL * DT))
            self._corner_stage = 'ROTATE'
            return state

        self._corner_ticks_remaining -= 1
        return robot.control(state, np.array([-RETREAT_SPEED, 0.0]))

    def _step_cornering_rotate(self, robot, state):
        if self._corner_ticks_remaining <= 0:
            self._start_new_trace_segment()
            self.phase = 'TRACE'
            return state

        self._corner_ticks_remaining -= 1
        action = np.array([0.0, self.sign * CORNER_TURN_ANG_VEL])
        return robot.control(state, action)

    # ------------------------------------------------------------ CROSS_SETUP
    def _step_cross_setup(self, robot, state):
        cand = self.candidates[self.crossing_idx]
        mid = (cand['start_pos'] + cand['end_pos']) / 2.0
        trace_heading = cand['heading']
        side_sign = 1.0 if cand['trace_side'] == 'right' else -1.0
        self._outward_heading = normalize_angle(trace_heading - side_sign * np.pi / 2.0)

        # Analytically computed tilt angle (not searched for): balances the forward-progress
        # clearance between front-right/rear-right crossing and rear-right/front-left crossing,
        # maximizing the largest gap width this footprint can ever safely bridge.
        tilt = np.arctan2(2.0 * robot.length, robot.width)
        self._target_heading = normalize_angle(self._outward_heading + tilt)

        # Back off from the edge to a pose fully on solid ground before rotating, so every
        # sensor reads floor at the start of the attempt regardless of the eventual tilt.
        backoff_dist = 0.5 * np.hypot(robot.length, robot.width) + CROSS_BACKOFF_MARGIN
        start_pos = mid - backoff_dist * np.array([np.cos(self._outward_heading), np.sin(self._outward_heading)])

        self.phase = 'CROSS_TILT_ROTATE'
        self.log = f'Attempting candidate {self.crossing_idx} (tilt={np.degrees(tilt):.1f} deg) -> CROSS_TILT_ROTATE'

        return np.array([start_pos[0], start_pos[1], self._outward_heading])

    # ------------------------------------------------------- CROSS_TILT_ROTATE
    def _step_cross_tilt_rotate(self, robot, state):
        angle_diff = normalize_angle(self._target_heading - state[2])
        if abs(angle_diff) < ANGLE_EPS:
            self._creep_start_pos = state[:2].copy()
            self._cross_prev_sensors = None
            self._cross_final_rotate = False
            self._cross_rl_triggered = False
            self.phase = 'CROSS_CREEP'
            return state

        ang_vel = np.sign(angle_diff) * TILT_ROTATE_ANG_VEL
        return robot.control(state, np.array([0.0, ang_vel]))

    # -------------------------------------------------------------- CROSS_CREEP
    def _step_cross_creep(self, robot, state):
        corners = robot.get_corner_positions(state)
        sensors = read_drop_sensors(corners)
        count = sensors.sum()
        progress = np.linalg.norm(state[:2] - self._creep_start_pos)

        if count >= 2:
            # Should not normally happen given the computed tilt angle - defensive fallback only.
            self.log = f'2+ sensors triggered during crossing (candidate {self.crossing_idx}) -> retreat, try next'
            self._retreat_ticks_remaining = FULL_RETREAT_TICKS
            self.phase = 'CROSS_RETREAT'
            return robot.control(state, np.array([-RETREAT_SPEED, 0.0]))

        # Corners cross the gap one at a time, in order FR (1) -> RR (2) -> FL (0) -> RL (3).
        # Once FL finishes crossing (its sensor goes from triggered back to floor-detected), RL
        # is the only corner left to bring across; forward translation alone was already carrying
        # it, but from here on we finish by rotating in place (anti-clockwise) instead, since the
        # other 3 corners are already solidly on the far platform.
        if (
            not self._cross_final_rotate
            and self._cross_prev_sensors is not None
            and self._cross_prev_sensors[0]
            and not sensors[0]
        ):
            self._cross_final_rotate = True
            self.log = f'FL crossed (candidate {self.crossing_idx}) -> final CCW rotation to bring RL across'
        self._cross_prev_sensors = sensors.copy()

        if self._cross_final_rotate:
            # Wait for RL to actually swing over the gap (sensor triggers) before treating a
            # floor reading as "cleared" - right when final rotation starts, RL is typically
            # still trailing on solid ground and hasn't triggered yet, so a floor reading here
            # doesn't yet mean success; keep rotating until it has dropped at least once.
            if sensors[3]:
                self._cross_rl_triggered = True
            elif self._cross_rl_triggered:
                self.result = 'SUCCESS'
                self.phase = 'DONE'
                self.log = f'Candidate {self.crossing_idx} crossed successfully! RL cleared the gap.'
                return state
            return robot.control(state, np.array([0.3, -TILT_ROTATE_ANG_VEL]))

        if progress > MAX_PROBE_DISTANCE:
            self.log = f'Candidate {self.crossing_idx} is a true cliff (no floor found) -> retreat, try next'
            self._retreat_ticks_remaining = FULL_RETREAT_TICKS
            self.phase = 'CROSS_RETREAT'
            return robot.control(state, np.array([-RETREAT_SPEED, 0.0]))

        # No sensor has triggered yet (or all have cleared again) and we haven't reached the
        # final rotation stage - the robot simply hasn't crept far enough forward yet to reach
        # the gap; keep creeping straight ahead.
        return robot.control(state, np.array([CREEP_SPEED, 0.0]))

    # ------------------------------------------------------------ CROSS_RETREAT
    def _step_cross_retreat(self, robot, state):
        if self._retreat_ticks_remaining <= 0:
            self.crossing_idx += 1
            if self.crossing_idx >= len(self.candidates):
                self.result = 'FAILURE'
                self.phase = 'DONE'
                self.log = 'All candidates failed.'
            else:
                self.phase = 'CROSS_SETUP'
            return state

        self._retreat_ticks_remaining -= 1
        return robot.control(state, np.array([-RETREAT_SPEED, 0.0]))
