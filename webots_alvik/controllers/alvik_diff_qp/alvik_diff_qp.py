

import argparse
import csv
import glob
import math
import os
import sys

import numpy as np

CONTROLLER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CONTROLLER_DIR, "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "webots_alvik"))

from controller import Supervisor  # noqa: E402  (provided by Webots)

from safe_alvik.cbf_qp import build_constraint_rows  # noqa: E402
from safe_alvik.config import Config, OBS_DIM  # noqa: E402
from safe_alvik.geometry import (as_obstacle_array, compensate_command,  # noqa: E402
                                 goal_body_features, is_collision, true_clearance,
                                 unicycle_step, wrap_angle)
from safe_alvik.io_utils import load_checkpoint  # noqa: E402
from safe_alvik.obstacle_memory import ObstacleMemory  # noqa: E402
from safe_alvik.sac import PolicyRunner  # noqa: E402
from safe_alvik.tof_model import ToFSensor, to_channels  # noqa: E402

from generate_world import LAYOUTS, TRACK_WIDTH_M, WHEEL_RADIUS_M  # noqa: E402

ZONE_NAMES = ("L", "CL", "CR", "R")


def parse_args(argv):
    """Layout comes from the world file's controllerArgs; the batch runner passes
    everything else through the environment so the generated worlds stay
    untouched between runs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default=os.environ.get("ALVIK_LAYOUT", "A"))
    parser.add_argument("--episodes", type=int,
                        default=int(os.environ.get("ALVIK_EPISODES", "20")))
    parser.add_argument("--checkpoint", default=os.environ.get("ALVIK_CHECKPOINT") or None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=int(os.environ.get("ALVIK_SEED", "0")))
    parser.add_argument("--framing", action="store_true",
                        default=bool(os.environ.get("ALVIK_FRAMING")),
                        help="render a few steps, export one image, quit")
    parser.add_argument("--record", type=int,
                        default=int(os.environ.get("ALVIK_RECORD", "0")),
                        help="record this many SUCCESSFUL episodes to mp4 and stop")
    parser.add_argument("--selftest", action="store_true",
                        default=bool(os.environ.get("ALVIK_SELFTEST")),
                        help="compare the Webots 3-D ray casts against the 2-D "
                             "training model over a grid of poses, then quit")
    known, _ = parser.parse_known_args(argv)
    return known


def _write_selftest(lines):
    path = os.path.join(PROJECT_ROOT, "webots_alvik", "outputs", "tof_selftest.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def selftest(runner, samples: int = 400) -> int:
    """Are the Webots readings the same as the 2-D model the policy trained on?

    This is the one question only this stage can answer. Any systematic gap
    here propagates straight into the observation and the CBF constraints.
    """
    from safe_alvik.tof_model import ideal_zone_distances

    lines = []

    def emit(text):
        print(text, flush=True)
        lines.append(text)

    emit("selftest: starting, %d poses" % samples)
    rng = np.random.default_rng(0)
    rows = []
    limit = runner.cfg.map.half_extent_m
    for _ in range(samples):
        pose = np.array([rng.uniform(-limit, limit), rng.uniform(-limit, limit),
                         rng.uniform(-math.pi, math.pi)])
        if is_collision(pose, runner.obstacles, runner.cfg.robot.body_radius_m):
            continue
        runner.true_pose = pose
        runner._write_pose()
        if runner.robot.step(runner.basic_step) == -1:
            break
        runner.robot.step(runner.basic_step)
        webots = runner.read_zones()
        model = ideal_zone_distances(pose, runner.obstacles, runner.cfg.tof,
                                     runner.cfg.robot.tof_offset_x_m)
        rows.append((webots, model))

    if not rows:
        emit("selftest: no usable poses")
        _write_selftest(lines)
        return 1
    web = np.array([r[0] for r in rows])
    mod = np.array([r[1] for r in rows])
    delta = web - mod
    near = mod < runner.cfg.tof.endpoint_accept_range_m
    emit("\nToF self-test: %d poses, 4 zones each" % len(rows))
    emit("  %-26s %9s %9s %9s" % ("", "mean_mm", "p95_mm", "max_mm"))
    emit("  %-26s %9.2f %9.2f %9.2f" % ("all readings |webots-model|",
                                         1000 * np.abs(delta).mean(),
                                         1000 * np.percentile(np.abs(delta), 95),
                                         1000 * np.abs(delta).max()))
    if near.any():
        emit("  %-26s %9.2f %9.2f %9.2f" % ("readings inside 0.70 m",
                                             1000 * np.abs(delta[near]).mean(),
                                             1000 * np.percentile(np.abs(delta[near]), 95),
                                             1000 * np.abs(delta[near]).max()))
        emit("  %-26s %+9.2f" % ("signed bias inside 0.70 m", 1000 * delta[near].mean()))
    for index, zone in enumerate(ZONE_NAMES):
        d = delta[:, index]
        emit("  zone %-3s mean %+7.2f mm   max |.| %7.2f mm" % (zone, 1000 * d.mean(),
                                                                 1000 * np.abs(d).max()))
    disagree = np.abs(delta) > 0.005
    emit("  readings differing by more than 5 mm: %.2f%%" % (100.0 * disagree.mean()))
    _write_selftest(lines)
    return 0


# Training roots in priority order. runs_g4_ttl8 is the current experiment
# (memory TTL 8.0 s); runs_g4 is the earlier TTL 5.0 s run, kept for the
# sensitivity comparison and used only if the newer root is absent.
TRAINING_ROOTS = ("runs_g4_ttl8", "runs_g4")


def default_checkpoint() -> str:
    """Best diff_qp checkpoint by the PLAN.MD 7.3 ordering: success up, then
    collision down."""
    import json
    for root in TRAINING_ROOTS:
        best, best_key = None, None
        pattern = os.path.join(PROJECT_ROOT, root, "diff_qp", "main_seed*",
                               "checkpoints", "best.pt")
        for path in sorted(glob.glob(pattern)):
            summary = os.path.join(os.path.dirname(os.path.dirname(path)),
                                   "evaluation_summary.json")
            if not os.path.exists(summary):
                continue
            data = json.load(open(summary, encoding="utf-8"))
            key = (data["success"], -data["collision"])
            if best_key is None or key > best_key:
                best, best_key = path, key
        if best is not None:
            print("checkpoint: %s (success %.2f, collision %.2f)"
                  % (best, best_key[0], -best_key[1]), flush=True)
            return best
    raise SystemExit("no evaluated diff_qp checkpoint found under %s"
                     % " or ".join(TRAINING_ROOTS))


class WebotsRunner:
    def __init__(self, args):
        self.robot = Supervisor()
        self.basic_step = int(self.robot.getBasicTimeStep())

        state = load_checkpoint(args.checkpoint or default_checkpoint())
        if state.get("mode") != "diff_qp":
            raise SystemExit("refusing to run mode %r: PLAN.MD 10.1 restricts Webots "
                             "and the real robot to diff_qp" % state.get("mode"))
        self.cfg = Config.from_dict(state["config"])
        self.checkpoint_step = state.get("step")
        self.policy = PolicyRunner(self.cfg, "diff_qp", state["actor"])

        self.control_dt = self.cfg.robot.control_dt_s
        self.substeps = max(1, int(round(self.control_dt * 1000.0 / self.basic_step)))

        self.rng = np.random.default_rng(args.seed)
        self.sensor_model = ToFSensor(self.cfg.tof, self.cfg.randomization,
                                      self.cfg.robot.tof_offset_x_m,
                                      self.cfg.robot.tof_to_bumper_m, self.rng)
        self.memory = ObstacleMemory(self.cfg.memory, self.cfg.tof.endpoint_accept_range_m)

        self.tof = []
        for zone in ZONE_NAMES:
            row = []
            for index in range(self.cfg.tof.rays_per_zone):
                device = self.robot.getDevice("tof_%s_%d" % (zone, index))
                device.enable(self.basic_step)
                row.append(device)
            self.tof.append(row)

        self.motors = [self.robot.getDevice("left wheel motor"),
                       self.robot.getDevice("right wheel motor")]
        for motor in self.motors:
            motor.setPosition(float("inf"))
            motor.setVelocity(0.0)

        node = self.robot.getSelf()
        self.translation = node.getField("translation")
        self.rotation = node.getField("rotation")
        self.z = self.translation.getSFVec3f()[2]

        self.obstacles = self._read_obstacles(args.layout)
        self.goal = np.asarray(self.cfg.map.goal_position, dtype=np.float64)

    # ------------------------------------------------------------------ scene
    def _read_obstacles(self, layout):
        """Ground truth for collision judging, read from the scene tree.

        Cross-checked against the layout table so a hand-edited world cannot
        silently disagree with what the batch runner thinks it is running.
        """
        children = self.robot.getRoot().getField("children")
        rows = []
        for index in range(children.getCount()):
            node = children.getMFNode(index)
            name_field = node.getField("name")
            if name_field is None or not name_field.getSFString().startswith("obstacle_"):
                continue
            position = node.getField("translation").getSFVec3f()
            shape = node.getField("boundingObject").getSFNode()
            radius = shape.getField("radius").getSFFloat()
            rows.append([position[0], position[1], radius])
        found = as_obstacle_array(sorted(rows))
        expected = as_obstacle_array(sorted([list(o) for o in LAYOUTS[layout]]))
        if found.shape != expected.shape or not np.allclose(found, expected, atol=1e-6):
            raise SystemExit("world obstacles %s do not match layout %s %s"
                             % (found.tolist(), layout, expected.tolist()))
        return found

    # ------------------------------------------------------------------ episode
    def reset(self, episode: int):
        cfg = self.cfg
        stage = cfg.curriculum[-1]
        base = np.asarray(cfg.map.start_pose, dtype=np.float64).copy()
        jitter = stage.start_pose_jitter_m
        base[0] += self.rng.uniform(-jitter, jitter)
        base[1] += self.rng.uniform(-jitter, jitter)
        base[2] = wrap_angle(base[2] + self.rng.uniform(-1.0, 1.0)
                             * stage.start_heading_jitter_deg * math.pi / 180.0)

        self.true_pose = base.copy()
        self.odom_pose = base.copy()
        self.odom_reported = base.copy()
        self._write_pose()

        self.sensor_model.reset()
        self.memory.reset()
        self.applied = np.zeros(2)
        self.speed_state = np.zeros(2)
        self.path_length = 0.0
        self.time_s = 0.0

        # Per-episode disturbances, drawn exactly as the training environment does.
        rnd = cfg.randomization
        low, high = rnd.gain_clip
        self.residual_gain = np.array([
            float(np.clip(self.rng.normal(1.0, rnd.gain_linear_std), low, high)),
            float(np.clip(self.rng.normal(1.0, rnd.gain_angular_std), low, high))])
        self.motor_tau = float(self.rng.uniform(*rnd.motor_tau_range_s))
        self.odom_scale = float(self.rng.normal(cfg.robot.odom_distance_gain,
                                                rnd.odom_scale_std))
        self.odom_walk = rnd.odom_heading_walk_deg * math.pi / 180.0
        # The training environment also delays commands by 0 or 1 control step
        # and drops 5% of telemetry frames. Both were missing here, which made
        # this stage quietly easier than the environment the policy trained in.
        from collections import deque
        self.command_delay = int(self.rng.random() < rnd.command_delay_prob)
        self.command_queue = deque([np.zeros(2)] * self.command_delay)

    def _write_pose(self):
        self.translation.setSFVec3f([float(self.true_pose[0]), float(self.true_pose[1]), self.z])
        self.rotation.setSFRotation([0.0, 0.0, 1.0, float(self.true_pose[2])])

    # ------------------------------------------------------------------ sensing
    def read_zones(self) -> np.ndarray:
        """Nearest hit per zone, straight from the Webots 3-D ray casts."""
        values = []
        for row in self.tof:
            values.append(min(device.getValue() for device in row))
        return np.clip(np.asarray(values, dtype=np.float64), 0.0, self.cfg.tof.max_range_m)

    def observe(self) -> np.ndarray:
        cfg = self.cfg
        dropped = (cfg.randomization.enabled
                   and self.rng.random() < cfg.randomization.telemetry_drop_prob)
        if dropped:
            channels = self.sensor_model.repeat_last()
        else:
            channels = self.sensor_model.push(
                to_channels(self.sensor_model.corrupt(self.read_zones())))
            self.odom_reported = self.odom_pose.copy()
        self.channels = channels
        self.memory.update(self.sensor_model.endpoints(self.odom_reported, channels), self.time_s)

        goal_bx, goal_by, goal_distance = goal_body_features(self.odom_reported, self.goal)
        normalized = np.clip(channels, 0.0, cfg.tof.max_range_m) / cfg.tof.max_range_m
        obs = np.empty(OBS_DIM, dtype=np.float32)
        norm = 2.0 * cfg.map.half_extent_m
        obs[0] = goal_bx / norm
        obs[1] = goal_by / norm
        obs[2] = goal_distance / norm
        obs[3:8] = normalized
        obs[8:13] = 1.0 - normalized
        obs[13:17] = self.memory.quadrant_features(self.odom_reported)
        obs[17] = self.applied[0] / cfg.robot.max_linear_mps
        obs[18] = self.applied[1] / cfg.robot.max_angular_rps
        return obs

    def constraints(self):
        return build_constraint_rows(self.odom_reported, self.memory.points(),
                                     self.cfg.robot, self.cfg.cbf, self.cfg.memory.max_tracks)

    # ------------------------------------------------------------------ acting
    def apply(self, desired: np.ndarray):
        """Desired body speed -> compensated command -> achieved motion.

        The PC compensates the measured open-loop gains, the robot then applies
        its true gain; the two very nearly cancel and what is left is the
        residual the policy was trained to tolerate.
        """
        robot = self.cfg.robot
        if self.command_delay > 0:
            self.command_queue.append(np.asarray(desired, dtype=np.float64).copy())
            desired = self.command_queue.popleft()
        command_v, command_w = compensate_command(
            float(desired[0]), float(desired[1]),
            robot.gain_linear, robot.gain_angular_ccw, robot.gain_angular_cw)
        gain_w = robot.gain_angular_ccw if command_w >= 0 else robot.gain_angular_cw
        target = np.array([command_v * robot.gain_linear * self.residual_gain[0],
                           command_w * gain_w * self.residual_gain[1]])

        decay = math.exp(-self.control_dt / self.motor_tau)
        previous = self.speed_state
        average = target + (previous - target) * (self.motor_tau / self.control_dt) * (1.0 - decay)
        self.speed_state = target + (previous - target) * decay
        self.applied = average.copy()
        return average, (command_v, command_w)

    def integrate(self, average: np.ndarray):
        """Sub-step the unicycle model so the motion renders smoothly."""
        dt = self.basic_step / 1000.0
        for _ in range(self.substeps):
            previous = self.true_pose[:2].copy()
            self.true_pose = unicycle_step(self.true_pose, average[0], average[1], dt)
            self.path_length += float(np.linalg.norm(self.true_pose[:2] - previous))
            scale = self.odom_scale / self.cfg.robot.odom_distance_gain
            omega = average[1]
            if self.odom_walk > 0.0:
                omega += float(self.rng.normal(0.0, self.odom_walk * math.sqrt(dt / self.control_dt))) / dt
            self.odom_pose = unicycle_step(self.odom_pose, average[0] * scale, omega, dt)
            self._write_pose()
            for motor, speed in zip(self.motors, self._wheel_speeds(average)):
                motor.setVelocity(speed)
            if self.robot.step(self.basic_step) == -1:
                return False
            self.time_s += dt
        return True

    @staticmethod
    def _wheel_speeds(average):
        half = 0.5 * TRACK_WIDTH_M
        left = (average[0] - average[1] * half) / WHEEL_RADIUS_M
        right = (average[0] + average[1] * half) / WHEEL_RADIUS_M
        return left, right

    # ------------------------------------------------------------------ episode
    def run_episode(self, episode: int):
        cfg = self.cfg
        self.reset(episode)
        if self.robot.step(self.basic_step) == -1:
            return None
        # Prime the three-frame median before the first decision.
        for _ in range(cfg.tof.median_frames):
            self.sensor_model.push(to_channels(self.sensor_model.corrupt(self.read_zones())))
            if self.robot.step(self.basic_step) == -1:
                return None

        record = {"episode": episode, "steps": 0, "interventions": 0, "infeasible": 0,
                  "min_clearance": float("inf"), "min_channel": float("inf"),
                  "collision": False, "success": False, "timeout": False,
                  "controller_stopped": False, "solve_ms": []}

        for step in range(cfg.robot.max_episode_steps):
            obs = self.observe()
            result = self.policy.act(obs, self.constraints(), deterministic=True)
            average, _ = self.apply(result["action_real"])
            if not self.integrate(average):
                return None

            record["steps"] = step + 1
            delta = result["z_safe"] - result["z_nominal"]
            if abs(float(delta[0])) > 1e-4 or abs(float(delta[1])) > 1e-4:
                record["interventions"] += 1
            if not result["qp"].get("feasible", True):
                record["infeasible"] += 1
            record["solve_ms"].append(float(result["qp"].get("solve_ms", 0.0)))

            clearance = true_clearance(self.true_pose, self.obstacles,
                                       cfg.robot.body_radius_m, cfg.robot.obstacle_margin_m)
            record["min_clearance"] = min(record["min_clearance"], clearance)
            record["min_channel"] = min(record["min_channel"], float(np.min(self.channels)))

            if is_collision(self.true_pose, self.obstacles, cfg.robot.body_radius_m):
                record["collision"] = True
                break
            distance_odom = float(np.linalg.norm(self.odom_reported[:2] - self.goal))
            if distance_odom <= cfg.map.controller_goal_radius_m:
                record["controller_stopped"] = True
                break
        else:
            record["timeout"] = True

        distance_true = float(np.linalg.norm(self.true_pose[:2] - self.goal))
        record["success"] = bool(record["controller_stopped"]
                                 and distance_true <= cfg.map.success_radius_m)
        record["final_distance_true"] = distance_true
        record["final_distance_odom"] = float(np.linalg.norm(self.odom_reported[:2] - self.goal))
        record["odom_error_m"] = float(np.linalg.norm(self.odom_reported[:2] - self.true_pose[:2]))
        record["path_length_m"] = self.path_length
        straight = float(np.linalg.norm(self.goal - np.asarray(self.cfg.map.start_pose[:2])))
        record["detour_ratio"] = self.path_length / straight if straight > 1e-9 else float("nan")
        record["qp_intervention_rate"] = record["interventions"] / max(record["steps"], 1)
        record["qp_infeasible_rate"] = record["infeasible"] / max(record["steps"], 1)
        record["qp_solve_ms_max"] = max(record["solve_ms"]) if record["solve_ms"] else 0.0
        record["qp_solve_ms_mean"] = (sum(record["solve_ms"]) / len(record["solve_ms"])
                                      if record["solve_ms"] else 0.0)
        del record["solve_ms"]
        for motor in self.motors:
            motor.setVelocity(0.0)
        return record


def record_videos(runner, args) -> int:
    """Record successful episodes to mp4, discarding the failed takes.

    Recording needs rendering, so this must not run with --no-rendering. Each
    episode is started as its own recording; if it does not end in a success the
    file is deleted and the next attempt begins, so what lands on disk is always
    a complete run from start to goal.
    """
    folder = os.path.join(PROJECT_ROOT, "webots_alvik", "outputs", "videos")
    os.makedirs(folder, exist_ok=True)
    kept, attempt = 0, 0
    while kept < args.record and attempt < args.record * 8:
        path = os.path.join(folder, "layout_%s_take%02d.mp4" % (args.layout, attempt))
        if attempt == 0:
            runner.robot.step(runner.basic_step)
            runner.robot.exportImage(
                os.path.join(folder, "framing_%s.jpg" % args.layout), 92)
        runner.robot.movieStartRecording(path, 1280, 720, 0, 92, 1, False)
        record = runner.run_episode(attempt)
        runner.robot.movieStopRecording()
        while not runner.robot.movieIsReady():
            if runner.robot.step(runner.basic_step) == -1:
                break
        attempt += 1
        if record is None:
            break
        if record["success"]:
            final = os.path.join(folder, "layout_%s_%02d.mp4" % (args.layout, kept + 1))
            if os.path.exists(final):
                os.remove(final)
            os.rename(path, final)
            kept += 1
            print("[%s] kept %s  (%d steps, final distance %.3f m)"
                  % (args.layout, os.path.basename(final), record["steps"],
                     record["final_distance_true"]), flush=True)
        else:
            if os.path.exists(path):
                os.remove(path)
            print("[%s] discarded take %d (%s)"
                  % (args.layout, attempt, "collision" if record["collision"]
                     else ("timeout" if record["timeout"] else "stopped short")), flush=True)
    print("[%s] recorded %d successful episode(s) in %d attempts"
          % (args.layout, kept, attempt), flush=True)
    runner.robot.simulationQuit(0 if kept else 1)
    return 0 if kept else 1


def main(argv=None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    runner = WebotsRunner(args)
    if args.selftest:
        # Webots swallows a controller exception into a bare "Terminating", which
        # is indistinguishable from a clean exit. Capture it into the report file.
        try:
            code = selftest(runner)
        except Exception:
            import traceback
            _write_selftest(["selftest failed:", traceback.format_exc()])
            traceback.print_exc()
            code = 1
        runner.robot.simulationQuit(code)
        return code
    output = args.output or os.path.join(PROJECT_ROOT, "webots_alvik", "outputs",
                                         "layout_%s.csv" % args.layout)
    os.makedirs(os.path.dirname(output), exist_ok=True)

    if args.framing:
        # exportImage before the renderer has drawn anything gives a blank
        # buffer, so step first.
        runner.reset(0)
        for _ in range(60):
            if runner.robot.step(runner.basic_step) == -1:
                break
        out = os.path.join(PROJECT_ROOT, "webots_alvik", "outputs", "videos",
                           "framing_%s.jpg" % args.layout)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        runner.robot.exportImage(out, 92)
        for _ in range(5):
            runner.robot.step(runner.basic_step)
        print("framing image -> %s" % out, flush=True)
        runner.robot.simulationQuit(0)
        return 0

    if args.record:
        return record_videos(runner, args)

    rows = []
    for episode in range(args.episodes):
        record = runner.run_episode(episode)
        if record is None:
            break
        record["layout"] = args.layout
        rows.append(record)
        print("[%s] episode %2d/%d  success=%d collision=%d steps=%3d dist=%.3f "
              "interv=%.2f infeas=%.2f"
              % (args.layout, episode + 1, args.episodes, record["success"],
                 record["collision"], record["steps"], record["final_distance_true"],
                 record["qp_intervention_rate"], record["qp_infeasible_rate"]), flush=True)

    if rows:
        with open(output, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow({k: (int(v) if isinstance(v, bool) else v)
                                 for k, v in row.items()})
        success = sum(r["success"] for r in rows) / len(rows)
        collision = sum(r["collision"] for r in rows) / len(rows)
        print("[%s] %d episodes  success=%.2f collision=%.2f -> %s"
              % (args.layout, len(rows), success, collision, output), flush=True)
    # Webots does not quit when a controller returns, even under --batch: the
    # process sits there until the caller's timeout fires, which costs about a
    # quarter of an hour per layout and looks identical to a hung run.
    runner.robot.simulationQuit(0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
