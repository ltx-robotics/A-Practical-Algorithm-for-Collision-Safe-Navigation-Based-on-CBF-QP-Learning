"""端点记忆 TTL 5.0 s vs 8.0 s：同 seed 配对对比。

    python seeds5_ttl8_visualization/make_ttl_comparison.py

两组运行只差 `memory.ttl_s` 一个变量（其余配置逐位相同，seed 也相同），
因此可以按 seed 配对。n=5 时 Wilcoxon 的下限就是 p=0.0625，所以逐 seed 的
连线本身是结果的一部分，不是附录。

vanilla 在这里是对照组：它不调用 QP，TTL 只通过观测里的四象限特征影响它。
如果 vanilla 也跟着大幅变化，说明差异不是来自安全层，整个对比就不成立。

输出：
  11_ttl_paired_outcomes.png/pdf   成功 / 碰撞 / 超时，逐 seed 连线
  12_ttl_qp_mechanism.png/pdf      QP 介入率与不可行率——机制侧的证据
  14_ttl_crutch_effect.png/pdf     推理时关掉 QP 后的碰撞率，TTL=5 与 TTL=8 并排
  ttl_comparison.csv               图 11–12 的全部数字
"""

import csv
import io
import os
import sys

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 120
matplotlib.rcParams["savefig.bbox"] = "tight"
matplotlib.rcParams["axes.grid"] = True
matplotlib.rcParams["grid.alpha"] = 0.25

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SEEDS = (42, 43, 44, 45, 46)
MODES = ("vanilla", "external_qp", "diff_qp")
LABEL = {"vanilla": "M1 vanilla（纯 SAC）",
         "external_qp": "M2 external_qp（不学修正项）",
         "diff_qp": "M3 diff_qp（学修正项）"}
SHORT = {"vanilla": "M1", "external_qp": "M2", "diff_qp": "M3"}
COLOR = {"vanilla": "#4C72B0", "external_qp": "#C44E52", "diff_qp": "#55A868"}
SOURCES = [("TTL 5.0 s", os.path.join(ROOT, "results", "g4_5seeds", "summary_by_seed.csv")),
           ("TTL 8.0 s", os.path.join(ROOT, "results", "g4_ttl8_5seeds", "summary_by_seed.csv"))]


def load(path):
    with io.open(path, encoding="utf-8") as handle:
        return {(r["mode"], int(r["seed"])): r for r in csv.DictReader(handle)}


DATA = [(tag, load(path)) for tag, path in SOURCES]


def series(table, mode, key):
    return np.array([float(table[(mode, seed)][key]) for seed in SEEDS])


def paired(mode, key):
    """(TTL=5 值, TTL=8 值, Wilcoxon p, 差值的 bootstrap 95% CI)。"""
    a = series(DATA[0][1], mode, key)
    b = series(DATA[1][1], mode, key)
    difference = b - a
    p = stats.wilcoxon(a, b).pvalue if np.any(difference) else 1.0
    rng = np.random.default_rng(20260823)      # 播种：论文里的数字不该每次跑都变
    boot = np.array([difference[rng.integers(0, 5, 5)].mean() for _ in range(10000)])
    return a, b, float(p), (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


def slope_panel(ax, key, title, ylabel):
    for index, mode in enumerate(MODES):
        a, b, p, ci = paired(mode, key)
        x = np.array([index * 1.0 - 0.16, index * 1.0 + 0.16])
        for va, vb in zip(a, b):
            ax.plot(x, [va, vb], color=COLOR[mode], alpha=0.35, linewidth=1.2,
                    marker="o", markersize=3.5, zorder=2)
        ax.plot(x, [a.mean(), b.mean()], color=COLOR[mode], linewidth=3.0,
                marker="o", markersize=8, zorder=3)
        excludes_zero = ci[0] > 0 or ci[1] < 0
        ax.annotate("%+.3f\np=%.3f%s" % (b.mean() - a.mean(), p, "*" if excludes_zero else ""),
                    xy=(index, max(a.max(), b.max())), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color=COLOR[mode])
    ax.set_xticks(range(len(MODES)))
    ax.set_xticklabels([SHORT[m] for m in MODES])
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.margins(y=0.22)


def figure(keys, titles, ylabels, name, suptitle):
    fig, axes = plt.subplots(1, len(keys), figsize=(4.4 * len(keys), 5.0))
    axes = np.atleast_1d(axes)
    for ax, key, title, ylabel in zip(axes, keys, titles, ylabels):
        slope_panel(ax, key, title, ylabel)
    handles = [Line2D([], [], color=COLOR[m], linewidth=3, label=LABEL[m]) for m in MODES]
    handles.append(Line2D([], [], color="0.45", linewidth=1.2, marker="o", markersize=3.5,
                          alpha=0.6, label="单个 seed"))
    fig.suptitle(suptitle, fontsize=10.5)
    fig.legend(handles=handles, fontsize=9, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "%s.%s" % (name, ext)))
    plt.close(fig)
    print("wrote %s.png / .pdf" % name)


def write_csv():
    keys = ["success", "collision", "timeout", "any_cbf_violation", "cbf_violation_rate",
            "final_distance_true", "detour_ratio", "min_true_clearance_m",
            "qp_intervention_rate", "qp_infeasible_rate", "mean_abs_delta_v",
            "mean_abs_delta_omega", "qp_solve_ms_mean", "steps_per_second"]
    available = set(DATA[0][1][(MODES[0], SEEDS[0])]) & set(DATA[1][1][(MODES[0], SEEDS[0])])
    missing = [k for k in keys if k not in available]
    if missing:
        print("skipping columns absent from one of the two summaries: %s" % ", ".join(missing))
    keys = [k for k in keys if k in available]
    path = os.path.join(HERE, "ttl_comparison.csv")
    with io.open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "metric", "ttl5_mean", "ttl8_mean", "delta",
                         "wilcoxon_p", "bootstrap_ci_low", "bootstrap_ci_high"]
                        + ["ttl5_seed%d" % s for s in SEEDS]
                        + ["ttl8_seed%d" % s for s in SEEDS])
        for mode in MODES:
            for key in keys:
                a, b, p, ci = paired(mode, key)
                writer.writerow([mode, key, a.mean(), b.mean(), b.mean() - a.mean(),
                                 p, ci[0], ci[1]] + list(a) + list(b))
    print("wrote ttl_comparison.csv")


# --------------------------------------------------------- 拐杖效应随 TTL 的变化
ABLATIONS = [("TTL 5.0 s", os.path.join(ROOT, "seeeds_total5_visualization",
                                        "ablation_5seeds.csv")),
             ("TTL 8.0 s", os.path.join(HERE, "ablation_5seeds.csv"))]
ABLATION_STEPS = (50000, 100000, 150000, 200000)


def ablation_rows(path):
    with io.open(path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for r in rows:
        r["step"] = int(r["step"])
        r["seed"] = int(r["seed"])
        r["collision"] = float(r["collision"])
    return rows


def crutch_figure():
    """关掉 QP 之后各方法自己站不站得住，以及这件事随 TTL 怎么变。"""
    available = [(tag, path) for tag, path in ABLATIONS if os.path.exists(path)]
    if len(available) < 2:
        print("skip 14: need both ablation caches")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharey=True)
    for ax, (tag, path) in zip(axes, available):
        rows = ablation_rows(path)
        for mode in MODES:
            values = np.array([[r["collision"] for r in rows
                                if r["mode"] == mode and r["step"] == st
                                and r["safety_layer"] == "off"]
                               for st in ABLATION_STEPS])
            if values.shape[1] != len(SEEDS):
                continue
            mean = values.mean(axis=1)
            half = stats.t.ppf(0.975, len(SEEDS) - 1) * values.std(axis=1, ddof=1) / np.sqrt(len(SEEDS))
            ax.plot(ABLATION_STEPS, mean, color=COLOR[mode], linewidth=2.4,
                    marker="o", markersize=6, label=LABEL[mode])
            ax.fill_between(ABLATION_STEPS, mean - half, mean + half,
                            color=COLOR[mode], alpha=0.16, linewidth=0)
        ax.set_title(tag, fontsize=11)
        ax.set_xlabel("环境步数")
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: "%dk" % (v / 1000)))
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(50000))
        ax.set_ylim(0.0, 1.05)
    axes[0].set_ylabel("关闭 CBF-QP 后的碰撞率")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("拐杖效应：同一 actor，推理时关掉 CBF-QP 之后的碰撞率（均值 ± 95% CI，n=5）。"
                 "M1 从未与 QP 一起训练，它这条线就是「不依赖」的参照。"
                 "记忆从 5 s 放宽到 8 s 让部署指标变好，却让 M3 的裸策略从与 M1 持平退到明显更差",
                 fontsize=10.5)
    fig.legend(handles, labels, fontsize=9, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.05, 1, 0.88])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "14_ttl_crutch_effect.%s" % ext))
    plt.close(fig)
    print("wrote 14_ttl_crutch_effect.png / .pdf")


def main():
    figure(["success", "collision", "timeout"],
           ["成功率", "碰撞率", "超时率"],
           ["成功率", "碰撞率", "超时率"],
           "11_ttl_paired_outcomes",
           "端点记忆 TTL 5.0 s → 8.0 s，同 seed 配对（n=5，其余配置逐位相同）。"
           "标注为均值差与 Wilcoxon p；* 表示差值的 bootstrap 95% CI 不含 0。"
           "M1 不调用 QP，是对照组")
    figure(["qp_intervention_rate", "qp_infeasible_rate"],
           ["QP 介入率（safe 动作 ≠ nominal）", "QP 不可行率（强制 v=0）"],
           ["介入率", "不可行率"],
           "12_ttl_qp_mechanism",
           "机制：记忆变长后 QP 看到的约束更多也更稳定。"
           "M2 的不可行率随之下降，M3 的介入率上升——两者指向同一个原因")
    crutch_figure()
    write_csv()
    return 0


if __name__ == "__main__":
    sys.exit(main())
