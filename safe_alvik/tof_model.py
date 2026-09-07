

import math
from typing import List, Optional, Sequence

import numpy as np

from .config import CHANNEL_NAMES, RandomizationConfig, ToFConfig, ZONE_NAMES
from .geometry import sensor_origin

DEG = math.pi / 180.0

# Index of each independent zone inside the five reported channels.
ZONE_TO_CHANNEL = (0, 1, 3, 4)   # L, CL, CR, R
CHANNEL_C = 2


def zone_ray_angles(cfg: ToFConfig) -> np.ndarray:
    """Ray angles in radians relative to the robot heading, shape (4, rays)."""
    half = 0.5 * cfg.zone_width_deg
    if cfg.rays_per_zone == 1:
        offsets = np.zeros(1, dtype=np.float64)
    else:
        offsets = np.linspace(-half, half, cfg.rays_per_zone, dtype=np.float64)
    centers = np.asarray(cfg.zone_centers_deg, dtype=np.float64)[:, None]
    return (centers + offsets[None, :]) * DEG


def cast_rays(origin: np.ndarray, angles: np.ndarray, obstacles: np.ndarray,
              max_range_m: float) -> np.ndarray:
    """Nearest positive ray/circle hit for every angle, clipped to max range.

    `angles` is any shape; the return has the same shape.
    """
    shape = angles.shape
    flat = angles.reshape(-1)
    result = np.full(flat.shape, max_range_m, dtype=np.float64)
    if obstacles.shape[0] == 0:
        return result.reshape(shape)

    directions = np.stack([np.cos(flat), np.sin(flat)], axis=1)      # (N, 2)
    oc = origin[None, :] - obstacles[:, :2]                          # (M, 2)
    b = 2.0 * (directions @ oc.T)                                    # (N, M)
    c_term = (oc * oc).sum(axis=1) - obstacles[:, 2] ** 2            # (M,)
    disc = b * b - 4.0 * c_term[None, :]
    hit = disc >= 0.0
    sqrt_disc = np.sqrt(np.where(hit, disc, 0.0))
    t_near = 0.5 * (-b - sqrt_disc)
    t_far = 0.5 * (-b + sqrt_disc)
    # Prefer the near root; fall back to the far root when the origin is inside.
    t = np.where(t_near > 0.0, t_near, t_far)
    valid = hit & (t > 0.0)
    t = np.where(valid, t, np.inf)
    nearest = t.min(axis=1)
    return np.minimum(nearest, max_range_m).reshape(shape)


def ideal_zone_distances(true_pose: Sequence[float], obstacles: np.ndarray,
                         cfg: ToFConfig, sensor_offset_x_m: float) -> np.ndarray:
    """Noise-free distance reported by each of the four zones, in metres."""
    origin = sensor_origin(true_pose, sensor_offset_x_m)
    angles = zone_ray_angles(cfg) + float(true_pose[2])
    return cast_rays(origin, angles, obstacles, cfg.max_range_m).min(axis=1)


def to_channels(zone_distances: np.ndarray) -> np.ndarray:
    """(L, CL, CR, R) -> (L, CL, C, CR, R) with C = (CL + CR) / 2."""
    channels = np.empty(5, dtype=np.float64)
    channels[0] = zone_distances[0]
    channels[1] = zone_distances[1]
    channels[3] = zone_distances[2]
    channels[4] = zone_distances[3]
    channels[CHANNEL_C] = 0.5 * (zone_distances[1] + zone_distances[2])
    return channels


class ToFSensor:
    """Geometry + calibrated error model + three-frame median filter.

    The calibration curve was measured on the C channel only. Until the four
    zones are calibrated individually, the same curve is used as a shared prior
    for all of them and the noise multiplier is widened to cover that modelling
    assumption (PLAN.MD 3.4).
    """

    def __init__(self, tof: ToFConfig, randomization: RandomizationConfig,
                 sensor_offset_x_m: float, bumper_offset_m: float = 0.008,
                 rng: Optional[np.random.Generator] = None):
        self.cfg = tof
        self.rnd = randomization
        self.sensor_offset_x_m = float(sensor_offset_x_m)
        self.bumper_offset_m = float(bumper_offset_m)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self._truth_mm = np.asarray(tof.calib_truth_mm, dtype=np.float64)
        self._bias_mm = np.asarray(tof.calib_bias_mm, dtype=np.float64)
        self._std_mm = np.asarray(tof.calib_std_mm, dtype=np.float64)
        self.noise_multiplier = 1.0
        self._history: List[np.ndarray] = []
        self._last_zones: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ setup
    def reset(self) -> None:
        self._history = []
        self._last_zones = None
        if self.rnd.enabled:
            low, high = self.rnd.tof_noise_multiplier_range
            self.noise_multiplier = float(self.rng.uniform(low, high))
        else:
            self.noise_multiplier = 1.0

    # ------------------------------------------------------------- error model
    def bias_std_m(self, distance_m: np.ndarray):
        """Interpolated bias and std for a sensor reading, in metres.

        The table is indexed by the manually measured truth, which was taken
        from the front bumper; the sensor window sits 8 mm behind it.
        """
        truth_mm = (np.asarray(distance_m, dtype=np.float64) - self.bumper_offset_m) * 1000.0
        bias = np.interp(truth_mm, self._truth_mm, self._bias_mm)
        std = np.interp(truth_mm, self._truth_mm, self._std_mm)
        return bias / 1000.0, std / 1000.0

    def corrupt(self, ideal_zones: np.ndarray) -> np.ndarray:
        """Apply bias, noise, previous-frame hold and single-frame false lows."""
        if not self.rnd.enabled:
            return ideal_zones.copy()

        bias, std = self.bias_std_m(ideal_zones)
        noisy = ideal_zones + bias + self.rng.normal(0.0, 1.0, ideal_zones.shape) * std * self.noise_multiplier

        # A saturated zone reports the modelling ceiling, not a noisy value.
        saturated = ideal_zones >= self.cfg.max_range_m - 1e-12
        noisy = np.where(saturated, self.cfg.max_range_m, noisy)

        if self._last_zones is not None and self.rnd.tof_hold_prob > 0.0:
            hold = self.rng.random(noisy.shape) < self.rnd.tof_hold_prob
            noisy = np.where(hold, self._last_zones, noisy)

        if self.rnd.tof_false_low_prob > 0.0:
            spike = self.rng.random(noisy.shape) < self.rnd.tof_false_low_prob
            low, high = self.rnd.tof_false_low_range_mm
            fake = self.rng.uniform(low, high, noisy.shape) / 1000.0
            noisy = np.where(spike, fake, noisy)

        noisy = np.clip(noisy, 0.01, self.cfg.max_range_m)
        self._last_zones = noisy.copy()
        return noisy

    # --------------------------------------------------------------- pipeline
    def push(self, channels: np.ndarray) -> np.ndarray:
        """Append a frame and return the per-channel median of the last N."""
        self._history.append(np.asarray(channels, dtype=np.float64).copy())
        if len(self._history) > self.cfg.median_frames:
            self._history.pop(0)
        return np.median(np.stack(self._history, axis=0), axis=0)

    def measure(self, true_pose: Sequence[float], obstacles: np.ndarray) -> np.ndarray:
        """One sensor frame: geometry -> error model -> median filter.

        Returns the five filtered channels (L, CL, C, CR, R) in metres, exactly
        as the policy and the CBF see them. Distances are raw sensor readings;
        the 8 mm window-to-bumper offset is display-only and is not subtracted.
        """
        ideal = ideal_zone_distances(true_pose, obstacles, self.cfg, self.sensor_offset_x_m)
        return self.push(to_channels(self.corrupt(ideal)))

    def repeat_last(self) -> np.ndarray:
        """Re-emit the current filtered frame (used when telemetry is dropped)."""
        if not self._history:
            return np.full(5, self.cfg.max_range_m, dtype=np.float64)
        return np.median(np.stack(self._history, axis=0), axis=0)

    # -------------------------------------------------------------- endpoints
    def endpoints(self, odom_pose: Sequence[float], channels: np.ndarray) -> np.ndarray:
        """Obstacle-surface endpoints in the odometry frame, shape (k, 2).

        Only the four independent zones contribute, and only when the reading
        is inside the acceptance range. C is a synthetic average and would
        duplicate CL/CR, so it is skipped.

        Modelling limitation, carried through to the CBF: a zone reports a
        range but not a bearing, so the endpoint is placed on the zone centre
        line. The true surface point lies within +/- 7.5 deg of it, i.e. up to
        2 d sin(3.75 deg) = 0.131 d away - 3.3 cm at the 0.25 m distances where
        the QP actually intervenes. The 30 mm obstacle margin is sized to
        absorb this, and it is one reason the safety layer must be described as
        an engineering CBF rather than a formal guarantee.
        """
        origin = sensor_origin(odom_pose, self.sensor_offset_x_m)
        theta = float(odom_pose[2])
        points = []
        for zone_index, channel_index in enumerate(ZONE_TO_CHANNEL):
            distance = float(channels[channel_index])
            if distance >= self.cfg.endpoint_accept_range_m:
                continue
            angle = theta + self.cfg.zone_centers_deg[zone_index] * DEG
            points.append((origin[0] + distance * math.cos(angle),
                           origin[1] + distance * math.sin(angle)))
        if not points:
            return np.zeros((0, 2), dtype=np.float64)
        return np.asarray(points, dtype=np.float64)
