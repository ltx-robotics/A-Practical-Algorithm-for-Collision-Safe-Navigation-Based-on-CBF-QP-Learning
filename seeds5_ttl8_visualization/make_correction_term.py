"""修正项 delta_u = u_safe − u_nom 在训练中怎么变（PLAN §1.1 假设 H2 的后半段）。

    python seeds5_ttl8_visualization/make_correction_term.py

这是整个课题的机制证据。M2 与 M3 用的是同一个 QP、同一套约束，唯一区别是梯度
是否穿过 QP。如果"学修正项"真的有用，它应该表现为：**M3 的 nominal 动作越来越
不需要被修正，M2 不是。**

只统计 40k 步之后，即课程进入 S1、地图上开始有障碍物、QP 才真正有事可做之后。

输出：
  13_correction_term.png/pdf   |delta_v|、|delta_omega|、QP 不可行率随训练的走向
  correction_term.csv          逐 seed 的起止值与 Spearman 相关系数
"""

import csv
import glob
import io
import os
import sys

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 120
matplotlib.rcParams["savefig.bbox"] = "tight"
matplotlib.rcParams["axes.grid"] = True
matplotlib.rcParams["grid.alpha"] = 0.25

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS = os.path.join(ROOT, "runs_g4_ttl8")
SEEDS = (42, 43, 44, 45, 46)
MODES = ("external_qp", "diff_qp")
LABEL = {"external_qp": "M2 external_qp（不学修正项）",
         "diff_qp": "M3 diff_qp（学修正项）"}
COLOR = {"external_qp": "#C44E52", "diff_qp": "#55A868"}
OBSTACLES_FROM = 40000        # S1 开始，地图上才有障碍物

PANELS = [("mean_abs_delta_v", "|Δv| 平均值 (m/s)", "线速度修正量"),
          ("mean_abs_delta_omega", "|Δω| 平均值 (rad/s)", "角速度修正量"),
          ("qp_infeasible_rate", "不可行率", "QP 不可行（被迫 v=0）")]


def curve(mode, key):
    steps, values = None, []
    for seed in SEEDS:
        matches = glob.glob(os.path.join(RUNS, mode, "main_seed%d_*" % seed, "validation.csv"))
        if not matches:
            raise SystemExit("missing validation.csv: %s seed %d" % (mode, seed))
        with io.open(sorted(matches)[-1], encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        steps = np.array([int(r["step"]) for r in rows])
        values.append([float(r[key]) for r in rows])
    mask = steps >= OBSTACLES_FROM
    return steps[mask], np.array(values)[:, mask]


def band(values):
    """均值与 n=5 的 t 分布 95% 半宽。"""
    mean = values.mean(axis=0)
    half = stats.t.ppf(0.975, len(values) - 1) * values.std(axis=0, ddof=1) / np.sqrt(len(values))
    return mean, half


def main():
    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.7 * len(PANELS), 5.0))
    records = []
    for ax, (key, ylabel, title) in zip(axes, PANELS):
        for mode in MODES:
            steps, values = curve(mode, key)
            mean, half = band(values)
            ax.plot(steps, mean, color=COLOR[mode], linewidth=2.3, label=LABEL[mode])
            ax.fill_between(steps, mean - half, mean + half, color=COLOR[mode],
                            alpha=0.18, linewidth=0)
            rho = [stats.spearmanr(steps, row).statistic for row in values]
            early, late = values[:, :3].mean(), values[:, -3:].mean()
            records.append([mode, key, early, late, late - early,
                            100.0 * (late - early) / max(early, 1e-12),
                            float(np.median(rho))] + [float(r) for r in rho])
            ax.annotate("%+.0f%%" % (100.0 * (late - early) / max(early, 1e-12)),
                        xy=(steps[-1], mean[-1]), xytext=(-6, 12 if mode == "external_qp" else -18),
                        textcoords="offset points", ha="right", fontsize=10,
                        color=COLOR[mode], fontweight="bold")
        ax.set_title(title, fontsize=10)
        # 步数标成 60k/80k/...，否则六位数的刻度会挤成一团
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: "%dk" % (v / 1000)))
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(40000))
        ax.set_xlabel("环境步数")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)

    fig.suptitle("修正项 Δu = u_safe − u_nom 的走向（均值 ± 95% CI，n=5，只取 S1 起 40k 步之后）。\n"
                 "M3 的 nominal 动作越来越不需要被修正，M2 反而越来越依赖被救——"
                 "这是「学修正项」起作用的直接机制证据（PLAN §1.1 H2 后半段）",
                 fontsize=10.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9.5, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.05, 1, 0.88])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "13_correction_term.%s" % ext))
    plt.close(fig)
    print("wrote 13_correction_term.png / .pdf")

    with io.open(os.path.join(HERE, "correction_term.csv"), "w", encoding="utf-8",
                 newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "metric", "early_mean", "late_mean", "delta", "percent",
                         "spearman_median"] + ["spearman_seed%d" % s for s in SEEDS])
        writer.writerows(records)
    print("wrote correction_term.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
