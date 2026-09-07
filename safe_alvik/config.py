
import copy
import json
import math
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEG = math.pi / 180.0

# Observation layout (PLAN.MD 3.2). Index order is frozen: training, Webots and
# the real-robot runtime must all produce exactly this vector.
OBS_NAMES: Tuple[str, ...] = (
    "goal_body_x_norm",
    "goal_body_y_norm",
    "goal_distance_norm",
    "L_norm", "CL_norm", "C_norm", "CR_norm", "R_norm",
    "L_prox", "CL_prox", "C_prox", "CR_prox", "R_prox",
    "memory_front_left", "memory_front_right",
    "memory_rear_left", "memory_rear_right",
    "applied_v_norm", "applied_omega_norm",
)
OBS_DIM = len(OBS_NAMES)
ACT_DIM = 2

# Five reported ToF outputs. C is synthesised as (CL + CR) / 2 by the carrier
# board and is NOT an independent zone.
CHANNEL_NAMES: Tuple[str, ...] = ("L", "CL", "C", "CR", "R")
# The four independent horizontal zones, in the order used internally.
ZONE_NAMES: Tuple[str, ...] = ("L", "CL", "CR", "R")

MODES: Tuple[str, ...] = ("vanilla", "external_qp", "diff_qp")
# Bumped whenever the meaning of a mode changes, so old checkpoints can never
# be silently mixed into a new comparison.
ALGORITHM_SEMANTICS = "opus-2026-08-20"
GOAL_OBSERVATION_MODE = "body_vector"


@dataclass
class MapConfig:
    """0.8 m x 0.8 m printed map, origin at the centre, no physical walls."""

    half_extent_m: float = 0.4                                   # MEASURED
    start_pose: Tuple[float, float, float] = (-0.26, -0.26, 0.0)  # MEASURED
    goal_position: Tuple[float, float] = (0.26, 0.26)             # MEASURED
    controller_goal_radius_m: float = 0.05
    success_radius_m: float = 0.07                                # MEASURED
    # The printed border is invisible to the ToF sensor, so it is ignored by
    # the CBF, the reward, the termination rule and the safety statistics.
    consider_map_boundary: bool = False


@dataclass
class RobotConfig:
    body_radius_m: float = 0.066            # MEASURED (circumscribed circle)
    obstacle_margin_m: float = 0.030
    look_ahead_m: float = 0.030
    max_linear_mps: float = 0.030           # MEASURED ceiling
    max_angular_rps: float = 25.0 * DEG     # MEASURED ceiling, 0.436332 rad/s
    allow_reverse: bool = False
    control_dt_s: float = 0.2               # 5 Hz, matches telemetry
    max_episode_steps: int = 300            # 60 s
    tof_offset_x_m: float = 0.037           # MEASURED, from the wheel axle mid
    tof_to_bumper_m: float = 0.008          # MEASURED, window to front edge
    # Inverse-gain compensation applied by the PC before sending a command.
    gain_linear: float = 0.908              # MEASURED
    gain_angular_ccw: float = 0.882         # MEASURED
    gain_angular_cw: float = 0.907          # MEASURED
    # Odometry reports 99.65 mm for a 100 mm command while the map-measured
    # true distance was ~97 mm, so the odometry overestimates travel by +2.7%
    # (calibration_and_measurement/encoder-and-odometry note, runs R01-R04).
    # The PC divides reported odometry by this factor, exactly as it divides
    # commands by the drive gains, and the real-robot runtime must do the same.
    # Without it the bias alone eats 0.735 * 0.027 = 19.8 mm, i.e. the entire
    # budget between the 50 mm stop radius and the 70 mm success radius.
    # MEASURED but PRELIMINARY: the calibration note itself asks for 200 mm and
    # 400 mm trials with per-trial map measurements before this is final, which
    # is why the residual is randomized generously.
    odom_distance_gain: float = 1.027

    @property
    def safety_radius_m(self) -> float:
        """Endpoint safety radius used by the CBF.

        Only body radius + obstacle margin. The look-ahead distance is already
        accounted for by moving the virtual control point forward and must not
        be added a second time.
        """
        return self.body_radius_m + self.obstacle_margin_m


@dataclass
class ToFConfig:
    """VL53L7CX modelled as four independent 15 deg zones inside a 60 deg FoV.

    Directions are fixed by real-robot checks: an obstacle to the front-left
    lowers L/CL, one to the front-right lowers R/CR. Mathematical angles are
    positive to the left.
    """

    zone_centers_deg: Tuple[float, ...] = (22.5, 7.5, -7.5, -22.5)  # MEASURED
    zone_width_deg: float = 15.0
    rays_per_zone: int = 7
    max_range_m: float = 1.2
    median_frames: int = 3
    endpoint_accept_range_m: float = 0.70
    # Static calibration, white matt target, 0 deg, run R01, comparing
    # (C_raw - 8 mm) against the manually measured truth.        MEASURED
    calib_truth_mm: Tuple[float, ...] = (50., 75., 100., 150., 200., 250., 300., 400., 500.)
    calib_bias_mm: Tuple[float, ...] = (5.12, 5.17, 3.39, 5.13, 1.93, -0.61, -2.93, -1.67, 0.67)
    calib_std_mm: Tuple[float, ...] = (0.68, 0.88, 0.77, 0.87, 0.77, 0.86, 1.06, 4.59, 1.79)


@dataclass
class MemoryConfig:
    """Short-term memory of ToF surface endpoints in the odometry frame."""

    cluster_radius_m: float = 0.08
    max_tracks: int = 4
    # 8.0 s = 0.24 m of travel at 3 cm/s.
    #
    # Two corrections, both away from the 1.4 s inherited from
    # parameter_opus.txt (which came from a 0.25 m/s project where it covered
    # 35 cm; at 3 cm/s it covers 4.2 cm, less than one body radius).
    #
    # A random-policy sweep first suggested 5.0 s (diagnostics/qp_memory_sweep.py:
    # collisions 10.0% -> 6.0% -> 1.5% for 1.4 / 3.0 / 5.0 s). That sweep was
    # misleading. Traced on a failing episode, the mechanism is: the robot drives
    # PAST an obstacle, the obstacle leaves the 60 deg forward FoV (it spends
    # over 90% of the episode outside it), the track expires while the surface is
    # still only 0.12 m away, and from then on the robot is blind to something
    # right beside it until it clips it. A random policy wanders, so it rarely
    # sustains the combination "obstacle close, out of FoV, still driving one
    # way"; a goal-directed policy does it on every single detour. This is not
    # specific to corridors - any time the robot goes around something. At 5.0 s
    # it costs 47% collisions on the corridor layout and 11% on the held-out set:
    #
    #   ttl        corridor        held-out
    #   5.0 s      0.47/0.47       0.86/0.11
    #   6.5 s      0.83/0.13       0.86/0.03
    #   8.0 s      0.93/0.00       0.88/0.03
    #   10.0 s     0.83/0.03       0.87/0.01     (stale memory -> timeouts)
    #
    # Lesson worth keeping: tuning a safety parameter against a random policy
    # cannot cover behaviours only a trained policy produces. See PLAN.MD 11.2.7.
    ttl_s: float = 8.0
    update_alpha: float = 0.5   # EMA weight on a matched endpoint
    # Frames a surface must be seen in before it constrains the QP. One
    # frame is enough for a ToF false low to defeat the 3-frame median
    # (two spikes inside one window) and create a phantom obstacle that
    # then survives the whole TTL. Measured on an empty map: with
    # confirm_frames=1 a phantom constrains 14.7% of steps at ttl 5.0 s.
    confirm_frames: int = 2


@dataclass
class CBFConfig:
    enabled: bool = True
    alpha: float = 1.0
    robust_velocity_mps: float = 0.006
    weight_linear: float = 2.0     # deviation weights, linear:angular = 2:1
    weight_angular: float = 1.0
    # Rows are normalized and z is O(1), so this sits just above float32 eps.
    feasibility_tol: float = 1e-6


@dataclass
class ObstacleConfig:
    radius_range_m: Tuple[float, float] = (0.04, 0.07)
    blocking_probability: float = 0.85
    min_surface_gap_m: float = 0.02
    start_goal_clearance_m: float = 0.03   # extra on top of body + obstacle r
    max_sampling_attempts: int = 200


@dataclass
class RandomizationConfig:
    """Domain randomisation, PLAN.MD 5.3. Resampled once per episode."""

    enabled: bool = True
    gain_linear_std: float = 0.08
    gain_angular_std: float = 0.08
    gain_clip: Tuple[float, float] = (0.80, 1.20)
    motor_tau_range_s: Tuple[float, float] = (0.10, 0.30)
    command_delay_prob: float = 0.3
    # Residual scale error left after the calibrated correction above. +/-2%
    # covers the uncertainty in the preliminary "~97 mm" figure.
    odom_scale_std: float = 0.02
    # Measured odometry heading change over a 100 mm run was +0.076 deg
    # (R01-R04), i.e. 0.0046 deg per 0.2 s control step at 3 cm/s. Modelling it
    # as a random walk of 0.05 deg/step accumulates to ~0.65 deg over the
    # 0.735 m task, which matches the measurement while staying conservative
    # (a mostly systematic effect treated as noise). The earlier 0.3 deg/step
    # accumulated to ~3.9 deg, roughly 7x the measured value, and was the
    # single largest cause of held-out failures in the G2 acceptance run.
    odom_heading_walk_deg: float = 0.05
    tof_noise_multiplier_range: Tuple[float, float] = (0.5, 2.0)
    tof_hold_prob: float = 0.05
    tof_false_low_prob: float = 0.02
    tof_false_low_range_mm: Tuple[float, float] = (60.0, 120.0)
    telemetry_drop_prob: float = 0.05


@dataclass
class RewardConfig:
    progress_coeff: float = 30.0
    step_distance_penalty: float = 0.01
    goal_bonus: float = 50.0
    collision_penalty: float = 60.0
    timeout_penalty: float = 15.0


@dataclass
class StageConfig:
    """One curriculum stage. `until_step` is the earliest exit boundary; the
    promotion gates must also be met before the next stage starts."""

    name: str = "stage"
    until_step: int = 200000
    n_obstacles_min: int = 1
    n_obstacles_max: int = 2
    randomize_start_goal: bool = False
    start_distance_range_m: Tuple[float, float] = (0.30, 0.735)
    start_pose_jitter_m: float = 0.02
    start_heading_jitter_deg: float = 10.0
    promote_min_success: float = 0.0
    promote_max_collision: float = 1.0


def default_curriculum() -> List[StageConfig]:
    return [
        StageConfig(name="S0_goal_only", until_step=40000,
                    n_obstacles_min=0, n_obstacles_max=0,
                    randomize_start_goal=True,
                    promote_min_success=0.80, promote_max_collision=1.0),
        StageConfig(name="S1_single_obstacle", until_step=100000,
                    n_obstacles_min=1, n_obstacles_max=1,
                    randomize_start_goal=False,
                    promote_min_success=0.60, promote_max_collision=0.10),
        StageConfig(name="S2_one_to_two", until_step=200000,
                    n_obstacles_min=1, n_obstacles_max=2,
                    randomize_start_goal=False),
    ]


@dataclass
class SACConfig:
    hidden_sizes: Tuple[int, ...] = (256, 256)
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    target_entropy: float = -float(ACT_DIM)
    alpha_init: float = 0.2
    # 0.01, not 0.05: with a 0.05 floor the entropy coefficient pinned to it
    # within 12k steps in all three methods, so automatic entropy tuning never
    # actually ran. Given room it settles at ~0.025 on its own. Measured on
    # seed 42, the floor changes no qualitative conclusion but does weaken the
    # vanilla baseline (0.62 -> 0.75 final success). See PLAN.MD 11.2.4.
    alpha_min: float = 0.01
    start_steps: int = 5000
    updates_per_step: int = 1
    replay_capacity: int = 100000
    log_std_min: float = -20.0
    log_std_max: float = 2.0


@dataclass
class TrainConfig:
    total_steps: int = 200000
    eval_interval: int = 25000
    eval_episodes: int = 100          # held-out target distribution
    stage_eval_episodes: int = 30     # cheaper check on the current stage
    holdout_seed_start: int = 1000
    log_interval: int = 1000
    checkpoint_interval: int = 25000
    # The held-out target distribution is always the final stage, 100% blocking.
    holdout_n_obstacles_min: int = 1
    holdout_n_obstacles_max: int = 2
    holdout_blocking_probability: float = 1.0


@dataclass
class Config:
    map: MapConfig = field(default_factory=MapConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    tof: ToFConfig = field(default_factory=ToFConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    cbf: CBFConfig = field(default_factory=CBFConfig)
    obstacles: ObstacleConfig = field(default_factory=ObstacleConfig)
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    curriculum: List[StageConfig] = field(default_factory=default_curriculum)

    # ---------------------------------------------------------------- checks
    def validate(self) -> "Config":
        r, m, t, c = self.robot, self.map, self.tof, self.cbf

        if r.max_linear_mps <= 0:
            raise ValueError("max_linear_mps must be positive")
        if r.allow_reverse:
            raise ValueError(
                "reverse motion is not part of this experiment; the board and "
                "the calibration only cover forward motion")
        if r.max_angular_rps <= 0 or r.control_dt_s <= 0:
            raise ValueError("angular limit and control period must be positive")
        if r.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")

        if m.success_radius_m < m.controller_goal_radius_m:
            raise ValueError(
                "the independent success radius must not be tighter than the "
                "odometry stop radius")
        if m.consider_map_boundary:
            raise ValueError(
                "the printed border has no wall and is invisible to the ToF "
                "sensor; consider_map_boundary must stay False")

        if len(t.zone_centers_deg) != len(ZONE_NAMES):
            raise ValueError("exactly four independent ToF zones are modelled")
        if not all(a > b for a, b in zip(t.zone_centers_deg, t.zone_centers_deg[1:])):
            raise ValueError("zone_centers_deg must be ordered left to right")
        if t.rays_per_zone < 1:
            raise ValueError("rays_per_zone must be >= 1")
        if t.median_frames < 1:
            raise ValueError("median_frames must be >= 1")
        n = len(t.calib_truth_mm)
        if len(t.calib_bias_mm) != n or len(t.calib_std_mm) != n:
            raise ValueError("ToF calibration tables must have equal length")
        if not all(a < b for a, b in zip(t.calib_truth_mm, t.calib_truth_mm[1:])):
            raise ValueError("calib_truth_mm must be strictly increasing")
        if t.endpoint_accept_range_m > t.max_range_m:
            raise ValueError("endpoint acceptance range exceeds the ToF range")

        if c.alpha <= 0:
            raise ValueError("CBF alpha must be positive")
        if c.robust_velocity_mps < 0:
            raise ValueError("robust velocity bound must be non-negative")
        if c.weight_linear <= 0 or c.weight_angular <= 0:
            raise ValueError("QP deviation weights must be positive")

        # The single most dangerous copy/paste error in this project: adding the
        # look-ahead into the endpoint safety radius double-compensates it.
        expected = r.body_radius_m + r.obstacle_margin_m
        if abs(r.safety_radius_m - expected) > 1e-12:
            raise ValueError("safety radius must be body radius + margin only")

        if not self.curriculum:
            raise ValueError("at least one curriculum stage is required")
        prev = 0
        for stage in self.curriculum:
            if stage.until_step <= prev:
                raise ValueError("curriculum boundaries must strictly increase")
            if stage.n_obstacles_min < 0 or stage.n_obstacles_max < stage.n_obstacles_min:
                raise ValueError("invalid obstacle count range in stage " + stage.name)
            prev = stage.until_step
        if self.curriculum[-1].until_step != self.train.total_steps:
            raise ValueError(
                "the last curriculum boundary must equal train.total_steps "
                "(got %d vs %d)" % (self.curriculum[-1].until_step, self.train.total_steps))

        if self.sac.alpha_min < 0:
            raise ValueError("entropy coefficient floor must be non-negative")
        if self.sac.start_steps < 0 or self.sac.batch_size <= 0:
            raise ValueError("invalid SAC schedule")
        return self

    # ------------------------------------------------------------ (de)serial
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=False)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Config":
        cfg = cls()
        if data:
            _apply_overrides(cfg, data)
        return cfg.validate()

    @classmethod
    def from_json(cls, path: Optional[str]) -> "Config":
        if not path:
            return cls().validate()
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def _apply_overrides(target: Any, data: Dict[str, Any]) -> None:
    """Recursively overwrite dataclass fields from a plain dict."""
    known = {f.name: f for f in fields(target)}
    for key, value in data.items():
        if key.startswith("_"):
            continue                      # documentation keys in config files
        if key not in known:
            raise ValueError("unknown config key: %s" % key)
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply_overrides(current, value)
        elif key == "curriculum":
            stages = []
            for item in value:
                stage = StageConfig()
                _apply_overrides(stage, item)
                stages.append(stage)
            setattr(target, key, stages)
        elif isinstance(current, tuple) and isinstance(value, list):
            setattr(target, key, tuple(value))
        else:
            setattr(target, key, copy.deepcopy(value))


def rescale_schedule(cfg: Config, total_steps: int) -> Config:
    """Shrink or stretch every step-based schedule to a new budget.

    Used by `--steps` so a short smoke run or timing benchmark exercises the
    same code path as the full experiment (curriculum promotion, validation,
    checkpointing, gradient updates) instead of silently skipping it. The main
    experiment must be run at the config's own budget, unscaled.
    """
    total_steps = int(total_steps)
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    old = cfg.train.total_steps
    if total_steps == old:
        return cfg.validate()
    ratio = total_steps / float(old)

    count = len(cfg.curriculum)
    if total_steps < count:
        raise ValueError("cannot fit %d curriculum stages into %d steps"
                         % (count, total_steps))
    previous = 0
    for index, stage in enumerate(cfg.curriculum):
        boundary = int(round(stage.until_step * ratio))
        boundary = min(max(boundary, previous + 1), total_steps - (count - 1 - index))
        stage.until_step = boundary
        previous = boundary
    cfg.curriculum[-1].until_step = total_steps

    scale = lambda value, floor=1: max(floor, int(round(value * ratio)))
    cfg.train.eval_interval = scale(cfg.train.eval_interval)
    cfg.train.checkpoint_interval = scale(cfg.train.checkpoint_interval)
    cfg.train.log_interval = scale(cfg.train.log_interval)
    cfg.sac.start_steps = scale(cfg.sac.start_steps, floor=0)
    cfg.train.total_steps = total_steps
    return cfg.validate()


def stage_for_step(cfg: Config, step: int, stage_index: int = 0) -> int:
    """Index of the curriculum stage that step belongs to, ignoring gates."""
    for index, stage in enumerate(cfg.curriculum):
        if step < stage.until_step:
            return max(index, 0)
    return len(cfg.curriculum) - 1
