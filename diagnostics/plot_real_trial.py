
import argparse
import csv
import glob
import io
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 130
matplotlib.rcParams["savefig.bbox"] = "tight"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, PROJECT_ROOT)

BODY_R = 0.066
HALF = 0.4
START = (-0.26, -0.26)
GOAL = (0.26, 0.26)


def load(path):
    with io.open(path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("empty log: %s" % path)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log")
    parser.add_argument("--obstacle", nargs=3, type=float, action="append", metavar=("X", "Y", "R"),
                        help="障碍物标称位置与半径，可重复；仅用于画图与估算间隙")
    parser.add_argument("--out", default=None, help="输出目录（默认与日志同目录）")
    parser.add_argument("--name", default=None, help="输出文件名（不含扩展名）")
    parser.add_argument("--simple", action="store_true",
                        help="只画轨迹面板，放大字号，适合放进幻灯片")
    parser.add_argument("--title", default=None, help="自定义标题")
    parser.add_argument("--endpoints", action="store_true",
                        help="叠加机器人实际感知到的 ToF 端点。里程计有漂移时，"
                             "端点与标称障碍物之间的偏差就是漂移量，画出来比藏起来诚实")
    args = parser.parse_args(argv)

    matches = sorted(glob.glob(args.log))
    path = matches[-1] if matches else args.log
    rows = load(path)
    f = lambda k: np.array([float(r[k]) for r in rows])

    x, y = f("odom_x"), f("odom_y")
    zn = np.c_[f("z_nom_v"), f("z_nom_w")]
    zs = np.c_[f("z_safe_v"), f("z_safe_w")]
    active = np.abs(zn - zs).max(axis=1) > 1e-9
    infeasible = f("qp_feasible") == 0
    cons = f("n_constraints")
    channels = np.c_[f("L_mm"), f("CL_mm"), f("C_mm"), f("CR_mm"), f("R_mm")]
    obstacles = np.array(args.obstacle if args.obstacle else [], dtype=float).reshape(-1, 3)

    if args.simple:
        fig, ax = plt.subplots(figsize=(7.2, 7.2))
        grid = None
    else:
        fig = plt.figure(figsize=(13.5, 6.0))
        grid = fig.add_gridspec(3, 2, width_ratios=[1.05, 1.0], hspace=0.45, wspace=0.22)
        ax = fig.add_subplot(grid[:, 0])

    ax.add_patch(plt.Rectangle((-HALF, -HALF), 2 * HALF, 2 * HALF, fill=False,
                               edgecolor="0.6", linestyle="--", linewidth=1.2))
    for ox, oy, r in obstacles:
        ax.add_patch(plt.Circle((ox, oy), r, color="0.35", zorder=2))
        ax.add_patch(plt.Circle((ox, oy), r + BODY_R, color="#C44E52", alpha=0.13,
                                zorder=1, linewidth=0))
    if args.endpoints:
        import math as _math
        from safe_alvik.tof_model import ToFSensor
        from safe_alvik.config import Config
        from safe_alvik.io_utils import load_checkpoint
        sys.path.insert(0, PROJECT_ROOT)
        from real_robot.pc.run_diff_qp_real import deployment_checkpoint
        cfg = Config.from_dict(load_checkpoint(deployment_checkpoint())["config"])
        quiet = Config.from_dict(cfg.to_dict()); quiet.randomization.enabled = False
        sensor = ToFSensor(quiet.tof, quiet.randomization,
                           quiet.robot.tof_offset_x_m, quiet.robot.tof_to_bumper_m)
        pts = []
        for r in rows:
            pose = np.array([float(r["odom_x"]), float(r["odom_y"]),
                             _math.radians(float(r["odom_theta_deg"]))])
            ch = np.array([float(r[c]) for c in
                           ("L_mm", "CL_mm", "C_mm", "CR_mm", "R_mm")]) / 1000.0
            e = sensor.endpoints(pose, sensor.push(ch))
            if e is not None and len(e):
                pts.extend(e)
        if pts:
            P = np.array(pts)
            ax.scatter(P[:, 0], P[:, 1], s=5, color="#8172B2", alpha=0.35, zorder=2,
                       label="感知到的 ToF 端点")
    ax.plot(x, y, color="0.55", linewidth=1.6, zorder=3, label="里程计轨迹")
    ax.scatter(x[active], y[active], s=18, color="#C44E52", zorder=4, label="QP 介入")
    if infeasible.any():
        ax.scatter(x[infeasible], y[infeasible], s=70, facecolors="none",
                   edgecolors="#8B0000", linewidths=1.6, zorder=5, label="QP 不可行")
    ax.plot(*START, marker="o", markersize=9, color="#4C72B0", zorder=6)
    ax.plot(*GOAL, marker="*", markersize=17, color="#55A868", zorder=6)
    ax.add_patch(plt.Circle(GOAL, 0.07, fill=False, edgecolor="#55A868",
                            linestyle=":", linewidth=1.4))
    ax.set_aspect("equal")
    ax.set_xlim(-HALF - 0.05, HALF + 0.05)
    ax.set_ylim(-HALF - 0.05, HALF + 0.05)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(alpha=0.25)

    title = args.title or os.path.basename(path).replace(".csv", "")
    if len(obstacles):
        centres = obstacles[:, :2]
        clearance = (np.linalg.norm(x[:, None] - centres[None, :, 0], axis=1) * 0 +
                     np.min(np.hypot(x[:, None] - centres[None, :, 0],
                                     y[:, None] - centres[None, :, 1]) - obstacles[None, :, 2],
                            axis=1)) - BODY_R
        k = int(np.argmin(clearance))
        # 标注往远离最近那个障碍物的方向甩，否则会压在圆上看不清
        nearest = centres[int(np.argmin(np.hypot(x[k] - centres[:, 0],
                                                 y[k] - centres[:, 1])))]
        away = np.array([x[k], y[k]]) - nearest
        offset = away / (float(np.linalg.norm(away)) or 1.0) * 52.0
        ax.annotate("最近处 %+.3f m\n第 %d 步" % (clearance[k], k), xy=(x[k], y[k]),
                    xytext=(float(offset[0]), float(offset[1])),
                    textcoords="offset points", fontsize=9.5, color="#8B0000",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8),
                    arrowprops=dict(arrowstyle="->", color="#8B0000", linewidth=1.2))
        ax.set_title("%s\n里程计估计的最小表面间隙 %+.3f m（阴影=碰撞区，真实值需实测）"
                     % (title, clearance.min()), fontsize=9.5)
    else:
        ax.set_title(title, fontsize=9.5)
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)

    if grid is None:
        fig.tight_layout()
        out_dir = args.out or os.path.dirname(os.path.abspath(path))
        os.makedirs(out_dir, exist_ok=True)
        stem = args.name or (os.path.basename(path)[:-4] + "_track")
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out_dir, "%s.%s" % (stem, ext)))
        plt.close(fig)
        print("wrote %s" % os.path.join(out_dir, stem + ".png"))
        return 0

    steps = np.arange(len(rows))
    top = fig.add_subplot(grid[0, 1])
    top.plot(steps, zn[:, 0], color="0.6", linewidth=1.3, label="nominal v")
    top.plot(steps, zs[:, 0], color="#C44E52", linewidth=1.6, label="safe v")
    top.fill_between(steps, zs[:, 0], zn[:, 0], color="#C44E52", alpha=0.2, linewidth=0)
    top.set_ylabel("归一化 v")
    top.legend(fontsize=8, ncol=2)
    top.grid(alpha=0.25)
    top.set_title("安全层把 nominal 削成了什么（阴影=被削掉的部分）", fontsize=9)

    mid = fig.add_subplot(grid[1, 1], sharex=top)
    mid.step(steps, cons, where="post", color="#4C72B0", linewidth=1.4)
    mid.set_ylabel("约束条数")
    mid.set_ylim(-0.2, cons.max() + 0.5)
    mid.grid(alpha=0.25)

    bot = fig.add_subplot(grid[2, 1], sharex=top)
    for index, (name, colour) in enumerate(zip(["L", "CL", "C", "CR", "R"],
                                               ["#4C72B0", "#55A868", "0.35",
                                                "#C44E52", "#8172B2"])):
        bot.plot(steps, channels[:, index], color=colour, linewidth=1.0, label=name)
    bot.axhline(100, color="#8B0000", linestyle="--", linewidth=1.0)
    bot.text(len(steps) * 0.01, 118, "板端 100 mm 禁止前进线", fontsize=7.5, color="#8B0000")
    bot.set_ylabel("ToF (mm)")
    bot.set_xlabel("控制步")
    bot.set_yscale("log")
    bot.legend(fontsize=7.5, ncol=5)
    bot.grid(alpha=0.25)

    out_dir = args.out or os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, (args.name or os.path.basename(path)[:-4]) + "_trial")
    for ext in ("png", "pdf"):
        fig.savefig(out + "." + ext)
    plt.close(fig)
    print("wrote %s.png / .pdf" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
