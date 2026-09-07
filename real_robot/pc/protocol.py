"""Wire format between the PC and the Alvik board.

Single source of truth for the packet layout and the unit conversions, so the
runtime never touches raw strings and the units can be unit-tested without
hardware. The board speaks cm/s, deg/s, mm and deg; everything above this module
is SI (m/s, rad/s, m, rad).

Board -> PC, UDP port 4211, 18 comma-separated fields:

    TEL,seq,board_ms,x_mm,y_mm,theta_deg,L,CL,C,CR,R,C_front,
        command_v,command_w,feedback_v,feedback_w,state,last_command_seq

PC -> board, UDP port 4212:

    CMD,<seq>,<linear_cm_s>,<angular_deg_s>
    RESET,<seq>,<x_mm>,<y_mm>,<theta_deg>
    BRAKE,<seq>

The board ignores any packet whose sequence is not strictly greater than the
last one it accepted, so sequence numbers must increase on every send including
heartbeats.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .board_safety import PASSTHROUGH_STATES

COMMAND_PORT = 4212
TELEMETRY_PORT = 4211
TELEMETRY_FIELDS = 18

# Caps enforced by the board itself. The PC compensates the measured drive gains
# before sending, so a 3 cm/s desired speed leaves here as about 3.3 cm/s; these
# caps exist so a compensation bug cannot command something wild.
BOARD_MAX_LINEAR_CM_S = 3.4
BOARD_MAX_ANGULAR_DEG_S = 30.0


@dataclass
class Telemetry:
    """One decoded board frame, converted to SI."""

    sequence: int
    board_ms: int
    pose: np.ndarray                 # (x, y, theta) in m, m, rad
    channels: np.ndarray             # L, CL, C, CR, R in m (raw, not median filtered)
    c_front_m: float                 # display-only: C minus the 8 mm bumper offset
    command_v: float                 # what the board last received, m/s
    command_omega: float             # rad/s
    feedback_v: Optional[float]      # get_drive_speed(), m/s, None when invalid
    feedback_omega: Optional[float]  # rad/s
    state: str                       # board safety state machine label
    last_command_sequence: int

    @property
    def board_blocking(self) -> bool:
        """True when the board is not executing the command as sent.

        Listed the other way round on purpose: the board has nine or so states
        and only a few of them are pass-through, so an unrecognised state must
        read as blocking rather than as fine.
        """
        return self.state not in PASSTHROUGH_STATES


def _optional(value: float) -> Optional[float]:
    return None if not math.isfinite(value) else value


def decode_telemetry(payload: bytes) -> Optional[Telemetry]:
    """Parse one telemetry datagram, or None if it is not a valid TEL frame."""
    try:
        fields = payload.decode("ascii", "ignore").strip().split(",")
    except Exception:
        return None
    if len(fields) != TELEMETRY_FIELDS or fields[0] != "TEL":
        return None
    try:
        numbers = [float(value) for value in fields[1:16]]
        state = fields[16]
        last_sequence = int(float(fields[17]))
    except ValueError:
        return None

    pose = np.array([numbers[2] * 1e-3, numbers[3] * 1e-3,
                     math.radians(numbers[4])], dtype=np.float64)
    channels = np.array(numbers[5:10], dtype=np.float64) * 1e-3
    feedback_v = _optional(numbers[13] * 1e-2)
    feedback_w = _optional(math.radians(numbers[14]))
    return Telemetry(
        sequence=int(numbers[0]),
        board_ms=int(numbers[1]),
        pose=pose,
        channels=channels,
        c_front_m=numbers[10] * 1e-3,
        command_v=numbers[11] * 1e-2,
        command_omega=math.radians(numbers[12]),
        feedback_v=feedback_v,
        feedback_omega=feedback_w,
        state=state,
        last_command_sequence=last_sequence,
    )


def encode_command(sequence: int, linear_mps: float, angular_rps: float) -> bytes:
    """A drive command, converted to the board's cm/s and deg/s and capped.

    The caller is expected to have applied the inverse-gain compensation
    already; this only guards against a runaway value.
    """
    linear_cm_s = float(np.clip(linear_mps * 100.0, 0.0, BOARD_MAX_LINEAR_CM_S))
    angular_deg_s = float(np.clip(math.degrees(angular_rps),
                                  -BOARD_MAX_ANGULAR_DEG_S, BOARD_MAX_ANGULAR_DEG_S))
    return ("CMD,%d,%.4f,%.4f" % (sequence, linear_cm_s, angular_deg_s)).encode("ascii")


def encode_brake(sequence: int) -> bytes:
    return ("BRAKE,%d" % sequence).encode("ascii")


def encode_reset(sequence: int, pose: Tuple[float, float, float]) -> bytes:
    """Reset the board's odometry to a known pose, in mm and deg."""
    return ("RESET,%d,%.2f,%.2f,%.3f"
            % (sequence, pose[0] * 1000.0, pose[1] * 1000.0,
               math.degrees(pose[2]))).encode("ascii")
