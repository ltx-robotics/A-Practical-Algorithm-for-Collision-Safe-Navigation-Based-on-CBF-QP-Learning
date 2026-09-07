

import math
from typing import Optional, Sequence, Tuple

import numpy as np

TWO_PI = 2.0 * math.pi


def wrap_angle(angle):
    """Wrap to [-pi, pi). Both endpoints denote the same heading."""
    return (np.asarray(angle) + math.pi) % TWO_PI - math.pi


def rotation_matrix(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def world_to_body(pose: Sequence[float], point: Sequence[float]) -> np.ndarray:
    """Express a world point in the robot body frame (forward, left)."""
    x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])
    dx = float(point[0]) - x
    dy = float(point[1]) - y
    c, s = math.cos(theta), math.sin(theta)
    return np.array([c * dx + s * dy, -s * dx + c * dy], dtype=np.float64)


def body_to_world(pose: Sequence[float], vec: Sequence[float]) -> np.ndarray:
    x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])
    c, s = math.cos(theta), math.sin(theta)
    return np.array([x + c * vec[0] - s * vec[1],
                     y + s * vec[0] + c * vec[1]], dtype=np.float64)


def goal_body_features(pose: Sequence[float], goal: Sequence[float]) -> Tuple[float, float, float]:
    """(forward component, left component, distance) of the goal.

    Using the body-frame vector instead of (distance, bearing) removes the
    discontinuity a relative angle has at +/- pi.
    """
    body = world_to_body(pose, goal)
    return float(body[0]), float(body[1]), float(math.hypot(body[0], body[1]))


def look_ahead_point(pose: Sequence[float], look_ahead_m: float) -> np.ndarray:
    """Virtual control point used by the CBF, ahead of the axle midpoint."""
    x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])
    return np.array([x + look_ahead_m * math.cos(theta),
                     y + look_ahead_m * math.sin(theta)], dtype=np.float64)


def sensor_origin(pose: Sequence[float], offset_x_m: float) -> np.ndarray:
    """World position of the ToF window."""
    return body_to_world(pose, (offset_x_m, 0.0))


def unicycle_step(pose: Sequence[float], v: float, omega: float, dt: float) -> np.ndarray:
    """Exact arc integration of the unicycle model over one control period."""
    x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])
    theta_next = theta + omega * dt
    if abs(omega) < 1e-9:
        x += v * math.cos(theta) * dt
        y += v * math.sin(theta) * dt
    else:
        radius = v / omega
        x += radius * (math.sin(theta_next) - math.sin(theta))
        y -= radius * (math.cos(theta_next) - math.cos(theta))
    return np.array([x, y, float(wrap_angle(theta_next))], dtype=np.float64)


# --------------------------------------------------------------------- action

def tanh_to_normalized(action):
    """Map a tanh-squashed action in [-1, 1]^2 to the normalized action box.

    Normalized action space is z = (v / v_max, omega / omega_max) with
    z_v in [0, 1] (no reverse) and z_omega in [-1, 1]. Works for numpy arrays
    and torch tensors.
    """
    forward = 0.5 * (action[..., 0:1] + 1.0)
    turn = action[..., 1:2]
    if isinstance(action, np.ndarray):
        return np.concatenate([forward, turn], axis=-1)
    import torch  # local import: keeps this module usable without torch
    return torch.cat([forward, turn], dim=-1)


def normalized_to_real(z, max_linear: float, max_angular: float):
    """Normalized action -> desired true body speeds (m/s, rad/s)."""
    v = z[..., 0:1] * max_linear
    w = z[..., 1:2] * max_angular
    if isinstance(z, np.ndarray):
        return np.concatenate([v, w], axis=-1)
    import torch
    return torch.cat([v, w], dim=-1)


def real_to_normalized(u, max_linear: float, max_angular: float):
    v = u[..., 0:1] / max_linear
    w = u[..., 1:2] / max_angular
    if isinstance(u, np.ndarray):
        return np.concatenate([v, w], axis=-1)
    import torch
    return torch.cat([v, w], dim=-1)


def compensate_command(v_desired: float, omega_desired: float,
                       gain_linear: float, gain_ccw: float, gain_cw: float
                       ) -> Tuple[float, float]:
    """Inverse-gain compensation applied by the PC before sending a command.

    The policy action is a desired *true* body speed. The measured open-loop
    gains are below one, so the command has to be scaled up. The discrete
    `move()` +3 mm and `rotate()` x1.035 corrections belong to those APIs only
    and must never be stacked on top of this.
    """
    command_v = v_desired / gain_linear
    gain_w = gain_ccw if omega_desired >= 0.0 else gain_cw
    command_w = omega_desired / gain_w
    return command_v, command_w


# ------------------------------------------------------------------ obstacles

def as_obstacle_array(obstacles: Optional[Sequence[Sequence[float]]]) -> np.ndarray:
    if obstacles is None or len(obstacles) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    array = np.asarray(obstacles, dtype=np.float64).reshape(-1, 3)
    return array


def obstacle_distances(point: Sequence[float], obstacles: np.ndarray) -> np.ndarray:
    """Surface distances from a point to each obstacle (negative when inside)."""
    if obstacles.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    delta = obstacles[:, :2] - np.asarray(point, dtype=np.float64)[None, :]
    return np.linalg.norm(delta, axis=1) - obstacles[:, 2]


def is_collision(pose: Sequence[float], obstacles: np.ndarray, body_radius_m: float) -> bool:
    """Physical collision test, evaluated on the true pose without any margin."""
    if obstacles.shape[0] == 0:
        return False
    return bool(np.any(obstacle_distances(pose[:2], obstacles) < body_radius_m))


def true_clearance(pose: Sequence[float], obstacles: np.ndarray,
                   body_radius_m: float, margin_m: float) -> float:
    """Ground-truth barrier value used for the safety metric.

    Positive means the robot circle plus the CBF margin is clear of every
    obstacle. This uses the true pose and the true obstacle geometry, neither
    of which the actor or the QP is allowed to see.
    """
    if obstacles.shape[0] == 0:
        return float("inf")
    return float(np.min(obstacle_distances(pose[:2], obstacles)) - body_radius_m - margin_m)


def segment_circle_min_distance(p0: Sequence[float], p1: Sequence[float],
                                center: Sequence[float]) -> float:
    """Shortest distance from a circle centre to a segment (for path checks)."""
    a = np.asarray(p0, dtype=np.float64)
    b = np.asarray(p1, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64)
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-12:
        return float(np.linalg.norm(c - a))
    t = float(np.clip((c - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(c - (a + t * ab)))
