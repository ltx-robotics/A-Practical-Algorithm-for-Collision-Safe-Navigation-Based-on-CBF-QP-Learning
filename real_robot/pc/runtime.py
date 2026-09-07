"""Real-robot decision loop: telemetry in, drive command out. No I/O here.

Keeping this free of sockets means the whole chain - odometry correction,
median filter, endpoint memory, 19-dim observation, CBF-QP, gain compensation -
can be tested without a robot, which is the only way to be sure the vector the
actor sees on hardware is the vector it saw in training.

Everything downstream of the raw telemetry is the same `safe_alvik` code the
policy trained with, imported rather than reimplemented.

Two things differ from simulation on purpose:

* the ToF error model is NOT applied. In training, `ToFSensor.corrupt` injects
  the calibrated bias and noise to imitate a real sensor; here the sensor is
  real, so only the three-frame median runs.
* the board's reported odometry overestimates travel by the calibrated
  +2.7% (`robot.odom_distance_gain`), so incremental displacement is divided by
  it. This is applied per frame rather than to the absolute pose, so it stays
  correct on curved paths. Skipping it makes the robot stop about 20 mm short
  every time - the entire budget between the 50 mm stop radius and the 70 mm
  success radius (PLAN.MD 11.2.2).
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from safe_alvik.cbf_qp import build_constraint_rows
from safe_alvik.config import ALGORITHM_SEMANTICS, Config, OBS_DIM
from safe_alvik.geometry import compensate_command, goal_body_features, wrap_angle
from safe_alvik.obstacle_memory import ObstacleMemory
from safe_alvik.sac import PolicyRunner
from safe_alvik.tof_model import ToFSensor

from .board_safety import BoardSafetyMirror


@dataclass
class StepResult:
    observation: np.ndarray
    z_nominal: np.ndarray
    z_safe: np.ndarray
    desired: np.ndarray            # desired true body speed, (m/s, rad/s)
    command: np.ndarray            # after inverse-gain compensation
    odom_pose: np.ndarray          # corrected
    channels: np.ndarray           # median filtered, metres
    goal_distance: float
    n_constraints: int
    feasible: bool
    solve_ms: float
    board_state: str
    predicted_board_state: str
    reached: bool
    memory_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))


class RealRobotRuntime:
    def __init__(self, cfg: Config, actor_state: Dict[str, Any], mode: str = "diff_qp"):
        if mode != "diff_qp":
            raise ValueError("PLAN.MD 10.1 restricts the real robot to diff_qp, got %r" % mode)
        self.cfg = cfg.validate()
        self.policy = PolicyRunner(cfg, mode, actor_state)
        # Randomization off: only the median filter is wanted, never the
        # synthetic bias/noise - the sensor here is the real one.
        quiet = Config.from_dict(cfg.to_dict())
        quiet.randomization.enabled = False
        self.sensor = ToFSensor(quiet.tof, quiet.randomization,
                                quiet.robot.tof_offset_x_m, quiet.robot.tof_to_bumper_m)
        self.memory = ObstacleMemory(cfg.memory, cfg.tof.endpoint_accept_range_m)
        self.board = BoardSafetyMirror()
        self.goal = np.asarray(cfg.map.goal_position, dtype=np.float64)
        self.goal_norm = 2.0 * cfg.map.half_extent_m
        self.reset(cfg.map.start_pose)

    # ------------------------------------------------------------------ setup
    def reset(self, start_pose) -> None:
        self.odom_pose = np.asarray(start_pose, dtype=np.float64).copy()
        self._last_board_pose: Optional[np.ndarray] = None
        self._last_board_ms: Optional[int] = None
        self.applied = np.zeros(2)
        self.time_s = 0.0
        self.steps = 0
        self.sensor.reset()
        self.memory.reset()
        self.board.reset()

    # ------------------------------------------------------------- odometry
    def _advance_odometry(self, telemetry) -> None:
        """Integrate the board's reported motion with the calibrated correction."""
        board_pose = telemetry.pose
        if self._last_board_pose is None:
            self._last_board_pose = board_pose.copy()
            return
        delta = board_pose[:2] - self._last_board_pose[:2]
        delta_theta = wrap_angle(board_pose[2] - self._last_board_pose[2])
        self._last_board_pose = board_pose.copy()

        gain = self.cfg.robot.odom_distance_gain
        self.odom_pose[:2] += delta / gain
        self.odom_pose[2] = wrap_angle(self.odom_pose[2] + float(delta_theta))

    def _applied_speeds(self, telemetry, dt: float) -> np.ndarray:
        """What the robot actually did last step, for the observation.

        Board feedback first (`get_drive_speed()`), then an odometry estimate,
        then the command times the calibrated gain - the fallback order
        parameter_opus.txt section 7 asks for.
        """
        if telemetry.feedback_v is not None and telemetry.feedback_omega is not None:
            return np.array([telemetry.feedback_v, telemetry.feedback_omega])
        if dt > 1e-6 and self._last_board_pose is not None:
            return np.array([float(np.linalg.norm(self.odom_pose[:2] - self._previous_xy)) / dt,
                             float(wrap_angle(self.odom_pose[2] - self._previous_theta)) / dt])
        robot = self.cfg.robot
        gain_w = robot.gain_angular_ccw if telemetry.command_omega >= 0 else robot.gain_angular_cw
        return np.array([telemetry.command_v * robot.gain_linear,
                         telemetry.command_omega * gain_w])

    # -------------------------------------------------------- board tracking
    def note_board_sample(self, telemetry) -> str:
        """Feed the mirror a board frame the control loop is not acting on.

        The board evaluates its latch on every 100 ms sample it takes; the PC
        decides at 5 Hz. Replaying the skipped frames here is what keeps the
        mirror aligned with the board instead of drifting by a factor of two.
        """
        return self.board.update(telemetry.channels[[1, 2, 3]] * 1000.0)

    # ---------------------------------------------------------------- observe
    def _observation(self) -> np.ndarray:
        cfg = self.cfg
        goal_bx, goal_by, goal_distance = goal_body_features(self.odom_pose, self.goal)
        normalized = np.clip(self.channels, 0.0, cfg.tof.max_range_m) / cfg.tof.max_range_m
        obs = np.empty(OBS_DIM, dtype=np.float32)
        obs[0] = goal_bx / self.goal_norm
        obs[1] = goal_by / self.goal_norm
        obs[2] = goal_distance / self.goal_norm
        obs[3:8] = normalized
        obs[8:13] = 1.0 - normalized
        obs[13:17] = self.memory.quadrant_features(self.odom_pose)
        obs[17] = self.applied[0] / cfg.robot.max_linear_mps
        obs[18] = self.applied[1] / cfg.robot.max_angular_rps
        return obs

    # ------------------------------------------------------------------- step
    def step(self, telemetry) -> StepResult:
        cfg = self.cfg
        dt = cfg.robot.control_dt_s
        if self._last_board_ms is not None:
            measured = (telemetry.board_ms - self._last_board_ms) / 1000.0
            if 0.02 < measured < 1.0:
                dt = measured
        self._last_board_ms = telemetry.board_ms

        self._previous_xy = self.odom_pose[:2].copy()
        self._previous_theta = float(self.odom_pose[2])
        self._advance_odometry(telemetry)
        self.applied = self._applied_speeds(telemetry, dt)
        self.time_s += dt
        self.steps += 1

        self.channels = self.sensor.push(telemetry.channels)
        self.memory.update(self.sensor.endpoints(self.odom_pose, self.channels), self.time_s)
        points = self.memory.points()
        constraints = build_constraint_rows(self.odom_pose, points, cfg.robot,
                                            cfg.cbf, cfg.memory.max_tracks)

        # Mirror the board using the same millimetre samples it sees (CL, C, CR),
        # raw rather than median filtered - the board has no filter.
        predicted = self.note_board_sample(telemetry)

        observation = self._observation()
        result = self.policy.act(observation, constraints, deterministic=True)
        desired = result["action_real"]
        command_v, command_w = compensate_command(
            float(desired[0]), float(desired[1]),
            cfg.robot.gain_linear, cfg.robot.gain_angular_ccw, cfg.robot.gain_angular_cw)

        goal_distance = float(np.linalg.norm(self.odom_pose[:2] - self.goal))
        return StepResult(
            observation=observation,
            z_nominal=result["z_nominal"],
            z_safe=result["z_safe"],
            desired=np.asarray(desired, dtype=np.float64),
            command=np.array([command_v, command_w]),
            odom_pose=self.odom_pose.copy(),
            channels=self.channels.copy(),
            goal_distance=goal_distance,
            n_constraints=int(constraints[2].sum()),
            feasible=bool(result["qp"].get("feasible", True)),
            solve_ms=float(result["qp"].get("solve_ms", 0.0)),
            board_state=telemetry.state,
            predicted_board_state=predicted,
            reached=goal_distance <= cfg.map.controller_goal_radius_m,
            memory_points=points.copy(),
        )

    # ------------------------------------------------------------- provenance
    def describe(self) -> Dict[str, Any]:
        cfg = self.cfg
        return {
            "algorithm_semantics": ALGORITHM_SEMANTICS,
            "control_hz": 1.0 / cfg.robot.control_dt_s,
            "max_linear_mps": cfg.robot.max_linear_mps,
            "max_angular_deg_s": math.degrees(cfg.robot.max_angular_rps),
            "odom_distance_gain": cfg.robot.odom_distance_gain,
            "gain_linear": cfg.robot.gain_linear,
            "gain_angular_ccw": cfg.robot.gain_angular_ccw,
            "gain_angular_cw": cfg.robot.gain_angular_cw,
            "memory_ttl_s": cfg.memory.ttl_s,
            "memory_confirm_frames": cfg.memory.confirm_frames,
            "cbf_alpha": cfg.cbf.alpha,
            "cbf_safety_radius_m": cfg.robot.safety_radius_m,
            "controller_goal_radius_m": cfg.map.controller_goal_radius_m,
            "success_radius_m": cfg.map.success_radius_m,
        }
