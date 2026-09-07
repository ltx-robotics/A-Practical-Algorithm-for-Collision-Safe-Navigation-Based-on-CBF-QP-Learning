"""PC-side mirror of the board's final safety layer.

The board runs this state machine itself and it cannot be overridden from here
(PLAN.MD 10.2). The mirror exists so the PC can predict what the board is about
to do, cross-check the state the board reports, and count every activation.

Deliberately NOT part of the training environment: modelling the board layer in
simulation would mask `vanilla` collisions and destroy the safety comparison
(PLAN.MD 10.2). It belongs here and only here.

Rules, transcribed from `update_tof_safety` in board/main.py:

* forward disabled when any ONE of CL/C/CR reads <= 100 mm on three consecutive
  board samples; v is forced to 0 but omega still executes, so the robot can
  turn away instead of locking up;
* released only when ALL of CL/C/CR are >= 110 mm for three consecutive samples;
* a single sample <= 60 mm on any front channel brakes completely and latches
  until that same three-frame release condition is met.

Note the asymmetry that matters for the paper: only the three FORWARD channels
take part. L and R are not watched, so a side contact - which is what actually
produces collisions in this task - is neither prevented nor recorded by the
board.

The labels returned here are the ones the board puts in the telemetry
`state` field, so `predicted_board_state` and `board_state` can be compared
column against column in the log. Two caveats when reading a disagreement:

* the board samples the filter on its own 100 ms clock and sends telemetry on
  another, so the mirror can legitimately lag by one frame;
* the board reports non-ToF states too (COMMAND_TIMEOUT_STOP, SENSOR_NOT_READY,
  REMOTE_BRAKE, POSE_RESET, ...). Nothing here can predict those.
"""

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

HARD_STOP_FRONT_MM = 100.0
HARD_STOP_RELEASE_MM = 110.0
CRITICAL_STOP_FRONT_MM = 60.0
CONSECUTIVE_FRAMES = 3

STATE_ACTIVE = "COMMAND_ACTIVE"
STATE_TRANSIENT = "TOF_TRANSIENT_LOW_IGNORED"
STATE_BLOCKED = "TOF_FORWARD_BLOCKED"
STATE_CRITICAL = "HARD_TOF_STOP_CRITICAL"

#: Board states in which the command the PC sent is executed unchanged.
PASSTHROUGH_STATES = (STATE_ACTIVE, STATE_TRANSIENT, "TOF_CLEAR", "POSE_RESET")


@dataclass
class BoardSafetyMirror:
    """Reproduces the board's ToF state machine from the front channels."""

    low_counts: List[int] = field(default_factory=lambda: [0, 0, 0])
    release_count: int = 0
    forward_blocked: bool = False
    critical: bool = False
    state: str = STATE_ACTIVE
    blocked_activations: int = 0
    critical_activations: int = 0

    def reset(self) -> None:
        """Clear the latch. Activation counters survive - they are the record."""
        self.low_counts = [0, 0, 0]
        self.release_count = 0
        self.forward_blocked = False
        self.critical = False
        self.state = STATE_ACTIVE

    def update(self, front_mm: Sequence[float]) -> str:
        """Feed one board sample of (CL, C, CR) in millimetres."""
        values = [float(v) for v in front_mm]
        if len(values) != 3:
            raise ValueError("expected exactly CL, C, CR")

        for index, value in enumerate(values):
            if value <= HARD_STOP_FRONT_MM:
                self.low_counts[index] = min(self.low_counts[index] + 1, CONSECUTIVE_FRAMES)
            else:
                self.low_counts[index] = 0

        # One sample is enough: the critical test runs before everything else
        # and re-latches even from an already blocked state.
        if min(values) <= CRITICAL_STOP_FRONT_MM:
            if not self.critical:
                self.critical_activations += 1
            self.critical = True
            self.forward_blocked = True
            self.release_count = 0
            self.state = STATE_CRITICAL
            return self.state

        if self.forward_blocked:
            if min(values) >= HARD_STOP_RELEASE_MM:
                self.release_count += 1
            else:
                self.release_count = 0
            if self.release_count >= CONSECUTIVE_FRAMES:
                self.reset()
                return self.state
            self.state = STATE_CRITICAL if self.critical else STATE_BLOCKED
            return self.state

        if max(self.low_counts) >= CONSECUTIVE_FRAMES:
            self.forward_blocked = True
            self.release_count = 0
            self.blocked_activations += 1
            self.state = STATE_BLOCKED
            return self.state

        # Below 100 mm but not yet three frames running: the board drives the
        # command unchanged and only flags the sample.
        self.state = STATE_TRANSIENT if max(self.low_counts) > 0 else STATE_ACTIVE
        return self.state

    def apply(self, linear_mps: float, angular_rps: float) -> Tuple[float, float]:
        """What the board will actually execute for a requested command."""
        if self.critical:
            return 0.0, 0.0
        if self.forward_blocked:
            return 0.0, angular_rps      # stop translating, keep turning
        return linear_mps, angular_rps
