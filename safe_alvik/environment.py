

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .cbf_qp import build_constraint_rows
from .config import Config, OBS_DIM, StageConfig
from .geometry import (as_obstacle_array, goal_body_features, is_collision,
                       obstacle_distances, segment_circle_min_distance,
                       true_clearance, unicycle_step, wrap_angle)
from .obstacle_memory import ObstacleMemory
from .tof_model import ToFSensor

DEG = math.pi / 180.0


@dataclass
class Scenario:
    start_pose: np.ndarray
    goal: np.ndarray
    obstacles: np.ndarray
    direct_path_blocked: bool = False
    straight_path_gap_m: float = float("inf")

    @property
    def n_obstacles(self) -> int:
        return int(self.obstacles.shape[0])


@dataclass
class EpisodeRandomization:
    """Per-episode actuation and odometry disturbances."""

    gain_linear: float = 1.0
    gain_angular: float = 1.0
    motor_tau_s: float = 0.15
    command_delay_steps: int = 0
    odom_scale: float = 1.0
    odom_heading_walk_rad: float = 0.0

    @classmethod
    def sample(cls, cfg: Config, rng: np.random.Generator) -> "EpisodeRandomization":
        rnd = cfg.randomization
        # `odom_scale` is the RAW odometry scale, centred on the calibrated
        # gain; the environment divides it back out, so a disabled-randomization
        # episode has perfect odometry.
        if not rnd.enabled:
            return cls(odom_scale=cfg.robot.odom_distance_gain)
        low, high = rnd.gain_clip
        return cls(
            gain_linear=float(np.clip(rng.normal(1.0, rnd.gain_linear_std), low, high)),
            gain_angular=float(np.clip(rng.normal(1.0, rnd.gain_angular_std), low, high)),
            motor_tau_s=float(rng.uniform(*rnd.motor_tau_range_s)),
            command_delay_steps=int(rng.random() < rnd.command_delay_prob),
            odom_scale=float(rng.normal(cfg.robot.odom_distance_gain, rnd.odom_scale_std)),
            odom_heading_walk_rad=float(rnd.odom_heading_walk_deg * DEG),
        )


class ScenarioGenerator:
    """Samples starts, goals and obstacles for one curriculum stage."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    # ------------------------------------------------------------------ public
    def sample(self, rng: np.random.Generator, stage: StageConfig,
               progress: float = 1.0,
               blocking_probability: Optional[float] = None) -> Scenario:
        cfg = self.cfg
        block_prob = (cfg.obstacles.blocking_probability
                      if blocking_probability is None else blocking_probability)

        for _ in range(cfg.obstacles.max_sampling_attempts):
            start, goal = self._sample_endpoints(rng, stage, progress)
            count = int(rng.integers(stage.n_obstacles_min, stage.n_obstacles_max + 1))
            if count == 0:
                return Scenario(start, goal, as_obstacle_array(None), False, float("inf"))

            want_block = bool(rng.random() < block_prob)
            obstacles = self._sample_obstacles(rng, start, goal, count, want_block)
            if obstacles is None:
                continue
            if not self._path_exists(start, goal, obstacles):
                continue
            gap = self._straight_path_gap(start, goal, obstacles)
            blocked = gap < 0.0
            if blocked != want_block:
                continue
            return Scenario(start, goal, obstacles, blocked, gap)

        # Never spin forever: fall back to an obstacle-free episode and let the
        # caller see it in the logs.
        start, goal = self._sample_endpoints(rng, stage, progress)
        return Scenario(start, goal, as_obstacle_array(None), False, float("inf"))

    # ----------------------------------------------------------------- helpers
    def _sample_endpoints(self, rng: np.random.Generator, stage: StageConfig,
                          progress: float) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        limit = cfg.map.half_extent_m - cfg.robot.body_radius_m
        if not stage.randomize_start_goal:
            base = np.asarray(cfg.map.start_pose, dtype=np.float64).copy()
            jitter = stage.start_pose_jitter_m
            base[0] += rng.uniform(-jitter, jitter)
            base[1] += rng.uniform(-jitter, jitter)
            base[2] = wrap_angle(base[2] + rng.uniform(-1.0, 1.0)
                                 * stage.start_heading_jitter_deg * DEG)
            return base, np.asarray(cfg.map.goal_position, dtype=np.float64)

        d_min, d_max = stage.start_distance_range_m
        reach = d_min + (d_max - d_min) * float(np.clip(progress, 0.0, 1.0))
        for _ in range(200):
            start_xy = rng.uniform(-limit, limit, size=2)
            distance = rng.uniform(d_min, max(d_min, reach))
            bearing = rng.uniform(-math.pi, math.pi)
            goal = start_xy + distance * np.array([math.cos(bearing), math.sin(bearing)])
            if np.all(np.abs(goal) <= limit):
                start = np.array([start_xy[0], start_xy[1],
                                  rng.uniform(-math.pi, math.pi)])
                return start, goal
        start = np.asarray(self.cfg.map.start_pose, dtype=np.float64).copy()
        return start, np.asarray(self.cfg.map.goal_position, dtype=np.float64)

    def _sample_obstacles(self, rng: np.random.Generator, start: np.ndarray,
                          goal: np.ndarray, count: int,
                          want_block: bool) -> Optional[np.ndarray]:
        cfg = self.cfg
        r_lo, r_hi = cfg.obstacles.radius_range_m
        body = cfg.robot.body_radius_m
        clearance = body + cfg.obstacles.start_goal_clearance_m
        rows: List[List[float]] = []

        for index in range(count):
            placed = False
            must_block = want_block and index == 0
            for _ in range(cfg.obstacles.max_sampling_attempts):
                radius = float(rng.uniform(r_lo, r_hi))
                if must_block:
                    centre = self._blocking_centre(rng, start[:2], goal, radius)
                else:
                    limit = cfg.map.half_extent_m - radius
                    centre = rng.uniform(-limit, limit, size=2)
                    if not want_block:
                        gap = segment_circle_min_distance(start[:2], goal, centre)
                        if gap < radius + body + 0.005:
                            continue
                if np.any(np.abs(centre) > cfg.map.half_extent_m - radius):
                    continue
                if np.linalg.norm(centre - start[:2]) < radius + clearance:
                    continue
                if np.linalg.norm(centre - goal) < radius + clearance:
                    continue
                if any(np.linalg.norm(centre - np.asarray(other[:2]))
                       < radius + other[2] + cfg.obstacles.min_surface_gap_m
                       for other in rows):
                    continue
                rows.append([float(centre[0]), float(centre[1]), radius])
                placed = True
                break
            if not placed:
                return None
        return as_obstacle_array(rows)

    def _blocking_centre(self, rng: np.random.Generator, start: np.ndarray,
                         goal: np.ndarray, radius: float) -> np.ndarray:
        """A centre close enough to the start-goal segment to block the corridor."""
        body = self.cfg.robot.body_radius_m
        direction = np.asarray(goal, dtype=np.float64) - np.asarray(start, dtype=np.float64)
        length = float(np.linalg.norm(direction))
        if length < 1e-9:
            return np.asarray(start, dtype=np.float64)
        unit = direction / length
        normal = np.array([-unit[1], unit[0]])
        along = float(rng.uniform(0.3, 0.7)) * length
        offset = float(rng.uniform(-1.0, 1.0)) * max(radius + body - 0.01, 0.0)
        return np.asarray(start, dtype=np.float64) + along * unit + offset * normal

    def _straight_path_gap(self, start: np.ndarray, goal: np.ndarray,
                           obstacles: np.ndarray) -> float:
        """Signed clearance of the straight corridor; negative means blocked."""
        if obstacles.shape[0] == 0:
            return float("inf")
        body = self.cfg.robot.body_radius_m
        gaps = [segment_circle_min_distance(start[:2], goal, row[:2]) - row[2] - body
                for row in obstacles]
        return float(min(gaps))

    def _path_exists(self, start: np.ndarray, goal: np.ndarray,
                     obstacles: np.ndarray, resolution: float = 0.02) -> bool:
        """Grid reachability with the body radius inflated into the obstacles.

        The search area extends past the printed square because nothing stops
        the robot from leaving it.
        """
        cfg = self.cfg
        limit = cfg.map.half_extent_m + 0.05
        axis = np.arange(-limit, limit + resolution, resolution)
        size = axis.shape[0]
        grid_x, grid_y = np.meshgrid(axis, axis, indexing="ij")
        free = np.ones((size, size), dtype=bool)
        for row in obstacles:
            distance = np.hypot(grid_x - row[0], grid_y - row[1])
            free &= distance > (row[2] + cfg.robot.body_radius_m)

        def to_cell(point):
            return (int(np.clip(round((point[0] + limit) / resolution), 0, size - 1)),
                    int(np.clip(round((point[1] + limit) / resolution), 0, size - 1)))

        start_cell, goal_cell = to_cell(start[:2]), to_cell(goal)
        if not free[start_cell] or not free[goal_cell]:
            return False

        seen = np.zeros_like(free)
        seen[start_cell] = True
        queue = deque([start_cell])
        while queue:
            cx, cy = queue.popleft()
            if (cx, cy) == goal_cell:
                return True
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < size and 0 <= ny < size and free[nx, ny] and not seen[nx, ny]:
                    seen[nx, ny] = True
                    queue.append((nx, ny))
        return False


class AlvikEnv:
    """Single-agent environment. Actions are desired true body speeds."""

    def __init__(self, cfg: Config, seed: int = 0):
        self.cfg = cfg.validate()
        self.rng = np.random.default_rng(seed)
        self.generator = ScenarioGenerator(cfg)
        self.sensor = ToFSensor(cfg.tof, cfg.randomization, cfg.robot.tof_offset_x_m,
                                cfg.robot.tof_to_bumper_m, self.rng)
        self.memory = ObstacleMemory(cfg.memory, cfg.tof.endpoint_accept_range_m)
        self.goal_norm = 2.0 * cfg.map.half_extent_m
        self.max_rows = cfg.memory.max_tracks

        self.scenario: Optional[Scenario] = None
        self.true_pose = np.zeros(3)
        self.odom_pose = np.zeros(3)
        self.odom_reported = np.zeros(3)
        self.channels = np.full(5, cfg.tof.max_range_m)
        self.constraints: Tuple[np.ndarray, np.ndarray, np.ndarray] = build_constraint_rows(
            self.odom_reported, None, cfg.robot, cfg.cbf, self.max_rows)
        self.step_index = 0
        self.time_s = 0.0
        self.path_length_m = 0.0
        self._applied = np.zeros(2)
        self._speed_state = np.zeros(2)
        self._command_queue: deque = deque()
        self._disturbance = EpisodeRandomization()
        self._previous_distance = 0.0

    # -------------------------------------------------------------------- api
    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.sensor.rng = self.rng

    def sample_scenario(self, stage: StageConfig, progress: float = 1.0,
                        blocking_probability: Optional[float] = None) -> Scenario:
        return self.generator.sample(self.rng, stage, progress, blocking_probability)

    def reset(self, scenario: Optional[Scenario] = None,
              stage: Optional[StageConfig] = None,
              progress: float = 1.0) -> np.ndarray:
        if scenario is None:
            stage = stage or self.cfg.curriculum[-1]
            scenario = self.sample_scenario(stage, progress)
        self.scenario = scenario

        self.true_pose = np.asarray(scenario.start_pose, dtype=np.float64).copy()
        self.odom_pose = self.true_pose.copy()
        self.odom_reported = self.true_pose.copy()
        self.step_index = 0
        self.time_s = 0.0
        self.path_length_m = 0.0
        self._applied = np.zeros(2)
        self._speed_state = np.zeros(2)
        self._disturbance = EpisodeRandomization.sample(self.cfg, self.rng)
        self._command_queue = deque([np.zeros(2)] * self._disturbance.command_delay_steps)

        self.sensor.reset()
        self.memory.reset()
        self.channels = self.sensor.measure(self.true_pose, scenario.obstacles)
        self.memory.update(self.sensor.endpoints(self.odom_reported, self.channels), self.time_s)
        self._previous_distance = float(np.linalg.norm(self.true_pose[:2] - scenario.goal))
        self._refresh_constraints()
        return self.observation()

    def step(self, action_real: Sequence[float]):
        cfg = self.cfg
        dt = cfg.robot.control_dt_s
        target = np.array([float(np.clip(action_real[0], 0.0, cfg.robot.max_linear_mps)),
                           float(np.clip(action_real[1], -cfg.robot.max_angular_rps,
                                         cfg.robot.max_angular_rps))], dtype=np.float64)

        commanded = self._delayed(target)
        average, end_state = self._motor_response(commanded, dt)
        self._speed_state = end_state
        self._applied = average.copy()

        previous_xy = self.true_pose[:2].copy()
        self.true_pose = unicycle_step(self.true_pose, average[0], average[1], dt)
        self.path_length_m += float(np.linalg.norm(self.true_pose[:2] - previous_xy))
        self._update_odometry(average, dt)

        self.step_index += 1
        self.time_s += dt

        dropped = (cfg.randomization.enabled
                   and self.rng.random() < cfg.randomization.telemetry_drop_prob)
        if dropped:
            self.channels = self.sensor.repeat_last()
        else:
            self.channels = self.sensor.measure(self.true_pose, self.scenario.obstacles)
            self.odom_reported = self.odom_pose.copy()
        self.memory.update(self.sensor.endpoints(self.odom_reported, self.channels), self.time_s)
        self._refresh_constraints()

        obstacles = self.scenario.obstacles
        goal = self.scenario.goal
        distance_true = float(np.linalg.norm(self.true_pose[:2] - goal))
        distance_odom = float(np.linalg.norm(self.odom_reported[:2] - goal))

        collision = is_collision(self.true_pose, obstacles, cfg.robot.body_radius_m)
        stopped = distance_odom <= cfg.map.controller_goal_radius_m
        success = bool(stopped and distance_true <= cfg.map.success_radius_m)
        timeout = self.step_index >= cfg.robot.max_episode_steps
        done = bool(collision or stopped or timeout)

        reward = cfg.reward.progress_coeff * (self._previous_distance - distance_true)
        reward -= cfg.reward.step_distance_penalty * (self._previous_distance / self.goal_norm)
        if success:
            reward += cfg.reward.goal_bonus
        if collision:
            reward -= cfg.reward.collision_penalty
        if timeout and not (collision or stopped):
            reward -= cfg.reward.timeout_penalty
        self._previous_distance = distance_true

        clearance = true_clearance(self.true_pose, obstacles,
                                   cfg.robot.body_radius_m, cfg.robot.obstacle_margin_m)
        info = {
            "collision": collision,
            "success": success,
            "controller_stopped": stopped,
            "timeout": timeout,
            "telemetry_dropped": bool(dropped),
            "distance_true": distance_true,
            "distance_odom": distance_odom,
            "true_clearance": clearance,
            "min_channel_m": float(np.min(self.channels)),
            "applied_v": float(self._applied[0]),
            "applied_omega": float(self._applied[1]),
            "path_length_m": self.path_length_m,
            "odom_error_m": float(np.linalg.norm(self.odom_reported[:2] - self.true_pose[:2])),
            "step": self.step_index,
        }
        return self.observation(), float(reward), done, info

    # ------------------------------------------------------------------ model
    def _delayed(self, target: np.ndarray) -> np.ndarray:
        if self._disturbance.command_delay_steps <= 0:
            return target
        self._command_queue.append(target.copy())
        return self._command_queue.popleft()

    def _motor_response(self, commanded: np.ndarray, dt: float):
        """First-order lag; returns the average speed over the step and the
        speed at its end. Averaging the exponential instead of taking one Euler
        sample keeps the integrated displacement right at dt = 0.2 s."""
        tau = self._disturbance.motor_tau_s
        gains = np.array([self._disturbance.gain_linear, self._disturbance.gain_angular])
        target = commanded * gains
        if not self.cfg.randomization.enabled or tau <= 1e-6:
            return target, target
        decay = math.exp(-dt / tau)
        previous = self._speed_state
        end = target + (previous - target) * decay
        average = target + (previous - target) * (tau / dt) * (1.0 - decay)
        return average, end

    def _update_odometry(self, achieved: np.ndarray, dt: float) -> None:
        """Integrate what the robot believes it did.

        The PC divides the reported odometry by the calibrated distance gain,
        the same pattern used for the drive commands, so only the residual scale
        error survives. The gain is a linear-travel property and is deliberately
        not applied to the heading: angular odometry has its own error, and the
        discrete `rotate()` x1.035 factor belongs to that API, not to continuous
        drive control.
        """
        scale = self._disturbance.odom_scale / self.cfg.robot.odom_distance_gain
        omega = achieved[1]
        walk = self._disturbance.odom_heading_walk_rad
        if walk > 0.0:
            omega += float(self.rng.normal(0.0, walk)) / dt
        self.odom_pose = unicycle_step(self.odom_pose, achieved[0] * scale, omega, dt)

    def _refresh_constraints(self) -> None:
        self.constraints = build_constraint_rows(
            self.odom_reported, self.memory.points(), self.cfg.robot,
            self.cfg.cbf, self.max_rows)

    # ------------------------------------------------------------ observation
    def observation(self) -> np.ndarray:
        cfg = self.cfg
        goal_bx, goal_by, goal_distance = goal_body_features(self.odom_reported,
                                                             self.scenario.goal)
        ranges = np.clip(self.channels, 0.0, cfg.tof.max_range_m)
        normalized = ranges / cfg.tof.max_range_m
        proximity = 1.0 - normalized
        memory_features = self.memory.quadrant_features(self.odom_reported)

        obs = np.empty(OBS_DIM, dtype=np.float32)
        obs[0] = goal_bx / self.goal_norm
        obs[1] = goal_by / self.goal_norm
        obs[2] = goal_distance / self.goal_norm
        obs[3:8] = normalized
        obs[8:13] = proximity
        obs[13:17] = memory_features
        obs[17] = self._applied[0] / cfg.robot.max_linear_mps
        obs[18] = self._applied[1] / cfg.robot.max_angular_rps
        return obs
