

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, Wedge

from safe_alvik.cbf_qp import barrier_values
from safe_alvik.config import Config
from safe_alvik.environment import AlvikEnv, Scenario
from safe_alvik.geometry import as_obstacle_array, look_ahead_point, sensor_origin, wrap_angle
from safe_alvik.io_utils import load_checkpoint
from safe_alvik.sac import PolicyRunner

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "webots_alvik"))
from generate_world import LAYOUTS  # noqa: E402

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

FFMPEG = os.path.join(os.environ.get("WEBOTS_HOME", r"D:\Webots"),
                      "msys64", "mingw64", "bin", "ffmpeg.exe")
if os.path.exists(FFMPEG):
    matplotlib.rcParams["animation.ffmpeg_path"] = FFMPEG

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEG = math.pi / 180.0


def alvik_outline():
    """Top-down silhouette of the real Alvik, from the reference-design STL.

    The mesh frame faces +Y and this project drives along +X, and ground contact
    sits 10.10 mm below the mesh origin - both worked out in
    webots_alvik/generate_world.py. Here only the yaw matters. Falls back to a
    circle of the measured body radius if the mesh is missing.
    """
    import struct
    path = os.path.join(ROOT, "webots_alvik", "meshes", "alvik_reference_design.stl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as handle:
        handle.read(80)
        count = struct.unpack("<I", handle.read(4))[0]
        raw = np.frombuffer(handle.read(count * 50), dtype=np.uint8).reshape(count, 50)
    tri = raw[:, 12:48].copy().view(np.float32).reshape(count, 3, 3)
    xy = tri.reshape(-1, 3)[:, :2] * 0.001
    xy = np.column_stack([xy[:, 1], -xy[:, 0]])          # -90 deg yaw: mesh +Y -> +X
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(xy)
        return xy[hull.vertices]
    except Exception:
        return None


def deployment_checkpoint():
    best = None
    for root in ("runs_g4_ttl8", "runs_g4"):
        for path in sorted(glob.glob(os.path.join(ROOT, root, "diff_qp", "main_seed*",
                                                  "checkpoints", "best.pt"))):
            summary = os.path.join(os.path.dirname(os.path.dirname(path)),
                                   "evaluation_summary.json")
            if not os.path.exists(summary):
                continue
            data = json.load(open(summary, encoding="utf-8"))
            key = (data["success"], -data["collision"])
            if best is None or key > best[0]:
                best = (key, path)
        if best is not None:
            return best[1]
    raise SystemExit("no evaluated diff_qp checkpoint found")


def rollout(cfg, runner, scenario, seed=0):
    """One episode, keeping everything the visualisation needs."""
    env = AlvikEnv(cfg, seed=seed)
    obs = env.reset(scenario)
    frames = []
    for step in range(cfg.robot.max_episode_steps):
        constraints = env.constraints
        result = runner.act(obs, constraints, deterministic=True)
        points = env.memory.points()
        frames.append({
            "true": env.true_pose.copy(),
            "odom": env.odom_reported.copy(),
            "channels": env.channels.copy(),
            "memory": points.copy(),
            "barriers": barrier_values(env.odom_reported, points, cfg.robot),
            "z_nom": result["z_nominal"].copy(),
            "z_safe": result["z_safe"].copy(),
            "feasible": bool(result["qp"].get("feasible", True)),
            "n_constraints": int(constraints[2].sum()),
        })
        obs, _, done, info = env.step(result["action_real"])
        if done:
            frames[-1]["info"] = info
            return frames, info
    return frames, info


def render(frames, info, scenario, cfg, path, layout, fps=5):
    obstacles = scenario.obstacles
    goal = np.asarray(cfg.map.goal_position)
    half = cfg.map.half_extent_m
    body = cfg.robot.body_radius_m
    safety = cfg.robot.safety_radius_m

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 6.2),
                                 gridspec_kw={"width_ratios": [1.35, 1]})
    ax.set_aspect("equal")
    ax.set_xlim(-half - 0.1, half + 0.1)
    ax.set_ylim(-half - 0.1, half + 0.1)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(alpha=0.2)
    ax.plot([-half, half, half, -half, -half], [-half, -half, half, half, -half],
            linestyle="--", color="#999999", linewidth=1)
    ax.annotate("打印地图边界（无实体墙，ToF 看不见）", (-half, -half - 0.06),
                fontsize=7.5, color="#777777")
    for row in obstacles:
        ax.add_patch(Circle(row[:2], row[2], color="#C44E52", alpha=0.85, zorder=3))
        ax.add_patch(Circle(row[:2], row[2] + safety, fill=False, linestyle=":",
                            color="#C44E52", alpha=0.7, zorder=2))
    ax.add_patch(Circle(goal, cfg.map.success_radius_m, color="#55A868", alpha=0.30, zorder=1))
    ax.annotate("终点", goal + np.array([0.0, 0.085]), ha="center", fontsize=9, color="#2f6b45")

    path_line, = ax.plot([], [], color="#4C72B0", linewidth=1.8, alpha=0.9, zorder=4)
    outline = alvik_outline()
    from matplotlib.patches import Polygon
    if outline is not None:
        robot_patch = Polygon(outline, closed=True, facecolor="#0E8286", alpha=0.85,
                              edgecolor="#0a5c5f", linewidth=1.2, zorder=6)
    else:
        robot_patch = Circle((0, 0), body, fill=False, color="#4C72B0", linewidth=2, zorder=6)
    ax.add_patch(robot_patch)
    # The collision test uses the circumscribed circle, so keep it visible.
    body_circle = Circle((0, 0), body, fill=False, color="#4C72B0", linewidth=1.0,
                         linestyle="--", alpha=0.6, zorder=6)
    ax.add_patch(body_circle)
    heading, = ax.plot([], [], color="#4C72B0", linewidth=2.4, zorder=6)
    fov = Wedge((0, 0), cfg.tof.endpoint_accept_range_m, 0, 1, color="#DDDDDD",
                alpha=0.35, zorder=0)
    ax.add_patch(fov)
    rays = [ax.plot([], [], color="#8C8C8C", linewidth=0.9, alpha=0.8, zorder=5)[0]
            for _ in range(4)]
    memory_dots, = ax.plot([], [], linestyle="none", marker="x", markersize=8,
                           color="#DD8452", markeredgewidth=2, zorder=7)
    title = ax.set_title("", fontsize=10)

    bx.set_xlim(0, len(frames))
    bx.set_xlabel("控制步")
    bx.grid(alpha=0.2)
    bx.axhline(0, color="k", linewidth=0.8)
    v_line, = bx.plot([], [], color="#4C72B0", label="v 名义 (mm/s)")
    vs_line, = bx.plot([], [], color="#55A868", label="v 安全 (mm/s)")
    h_line, = bx.plot([], [], color="#C44E52", label="最小 h (mm)")
    bx.legend(fontsize=8, loc="upper right")
    bx.set_title("策略输出 vs 安全层修正，以及最小 CBF 裕量", fontsize=10)
    limit = max(40, cfg.robot.max_linear_mps * 1000 * 1.2)
    bx.set_ylim(-limit, limit * 2.2)

    xs, ys, vn, vs, hs = [], [], [], [], []

    def update(index):
        f = frames[index]
        pose = f["true"]
        xs.append(pose[0]); ys.append(pose[1])
        path_line.set_data(xs, ys)
        body_circle.center = (pose[0], pose[1])
        if outline is not None:
            c, sn = math.cos(pose[2]), math.sin(pose[2])
            rotated = outline @ np.array([[c, sn], [-sn, c]])
            robot_patch.set_xy(rotated + pose[:2])
        else:
            robot_patch.center = (pose[0], pose[1])
        nose = pose[:2] + body * np.array([math.cos(pose[2]), math.sin(pose[2])])
        heading.set_data([pose[0], nose[0]], [pose[1], nose[1]])

        origin = sensor_origin(f["odom"], cfg.robot.tof_offset_x_m)
        fov.set_center(tuple(origin))
        fov.set_theta1(math.degrees(f["odom"][2]) - 30.0)
        fov.set_theta2(math.degrees(f["odom"][2]) + 30.0)
        for zone, (line, centre) in enumerate(zip(rays, cfg.tof.zone_centers_deg)):
            channel = f["channels"][[0, 1, 3, 4][zone]]
            angle = f["odom"][2] + centre * DEG
            end = origin + channel * np.array([math.cos(angle), math.sin(angle)])
            line.set_data([origin[0], end[0]], [origin[1], end[1]])
            line.set_alpha(0.9 if channel < cfg.tof.endpoint_accept_range_m else 0.15)
        if len(f["memory"]):
            memory_dots.set_data(f["memory"][:, 0], f["memory"][:, 1])
        else:
            memory_dots.set_data([], [])

        vn.append(f["z_nom"][0] * cfg.robot.max_linear_mps * 1000)
        vs.append(f["z_safe"][0] * cfg.robot.max_linear_mps * 1000)
        hs.append(1000 * float(np.min(f["barriers"])) if len(f["barriers"]) else np.nan)
        frame_index = list(range(len(vn)))
        v_line.set_data(frame_index, vn)
        vs_line.set_data(frame_index, vs)
        h_line.set_data(frame_index, hs)

        flag = "" if f["feasible"] else "   QP 不可行 → v=0"
        title.set_text("布局 %s   步 %3d/%d   约束 %d 条   记忆 %d 个端点%s"
                       % (layout, index + 1, len(frames), f["n_constraints"],
                          len(f["memory"]), flag))
        return path_line, robot_patch, heading, fov, memory_dots, v_line, vs_line, h_line

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps,
                                   blit=False, repeat=False)
    writer = animation.FFMpegWriter(fps=fps, bitrate=2400,
                                    metadata={"title": "Alvik SAC + diff CBF-QP"})
    anim.save(path, writer=writer, dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layout", nargs="+", default=["A", "B", "C"])
    parser.add_argument("--episodes", type=int, default=1,
                        help="successful episodes to render per layout")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=os.path.join(ROOT, "webots_alvik", "outputs", "videos"))
    args = parser.parse_args(argv)

    checkpoint = args.checkpoint or deployment_checkpoint()
    state = load_checkpoint(checkpoint)
    cfg = Config.from_dict(state["config"])
    runner = PolicyRunner(cfg, "diff_qp", state["actor"])
    print("checkpoint %s (step %s, TTL %.1f s)" % (checkpoint, state.get("step"),
                                                   cfg.memory.ttl_s))
    os.makedirs(args.output, exist_ok=True)

    for layout in args.layout:
        obstacles = as_obstacle_array([list(o) for o in LAYOUTS[layout]])
        rng = np.random.default_rng(0)
        kept = 0
        for attempt in range(30):
            start = np.asarray(cfg.map.start_pose, dtype=np.float64).copy()
            start[0] += rng.uniform(-0.02, 0.02)
            start[1] += rng.uniform(-0.02, 0.02)
            start[2] = wrap_angle(start[2] + rng.uniform(-1, 1) * 10 * DEG)
            scenario = Scenario(start, np.asarray(cfg.map.goal_position), obstacles)
            frames, info = rollout(cfg, runner, scenario, seed=attempt)
            if not info.get("success"):
                continue
            kept += 1
            path = os.path.join(args.output, "trajectory_%s_%02d.mp4" % (layout, kept))
            render(frames, info, scenario, cfg, path, layout)
            print("  %s  %d 步, 终点距离 %.3f m -> %s"
                  % (layout, len(frames), info["distance_true"], path), flush=True)
            if kept >= args.episodes:
                break
        if kept == 0:
            print("  %s: 30 次尝试内没有成功回合" % layout, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
