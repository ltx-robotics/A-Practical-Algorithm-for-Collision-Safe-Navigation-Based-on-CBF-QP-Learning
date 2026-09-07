"""Figures for the G4 experiment: 3 methods x 5 seeds x 200k steps.

    python seeeds_total5_visualization/make_figures.py

Reads `runs_g4/` (config `configs/main_80cm.json`, alpha_min=0.01) plus the
cached safety-layer ablation in `ablation_5seeds.csv`.

Bands are mean +/- 95% CI over the five seeds (t distribution, 4 dof).

Two things every figure here is built to keep honest:

* cross-method numbers are read at a MATCHED step (200k), never from `best.pt`.
  `best.pt` is the max over eight noisy validation points, so it rewards the
  method with the noisier curve - see figure 07.
* with n=5 the two-sided Wilcoxon signed-rank test bottoms out at p = 0.0625
  even when all five seeds agree, so no comparison here can reach p < 0.05.
  Figure 05 shows the per-seed signs, which is the evidence that matters.
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
# Rebound by main(): the same figures are produced for the TTL=5 run set
# (runs_g4) and the TTL=8 one (runs_g4_ttl8), so there is one script, not two
# copies that drift apart.
RUNS = os.path.join(ROOT, "runs_g4")
OUTDIR = HERE

SEEDS = (42, 43, 44, 45, 46)
MODES = ("vanilla", "external_qp", "diff_qp")
LABEL = {"vanilla": "M1 vanilla (纯 SAC)",
         "external_qp": "M2 external_qp (CBF-QP，不学修正项)",
         "diff_qp": "M3 diff_qp (CBF-QP，学修正项)"}
SHORT = {"vanilla": "M1", "external_qp": "M2", "diff_qp": "M3"}
COLOR = {"vanilla": "#4C72B0", "external_qp": "#C44E52", "diff_qp": "#55A868"}


# ------------------------------------------------------------------ data load
def run_dir(mode, seed):
    matches = sorted(glob.glob(os.path.join(RUNS, mode, "main_seed%d_*" % seed)))
    if not matches:
        raise SystemExit("missing run: %s seed %d" % (mode, seed))
    return matches[-1]


def read_csv(path):
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def curves(mode, key, filename="validation.csv"):
    """(steps, values) with values shaped (n_seeds, n_points)."""
    stack, steps = [], None
    for seed in SEEDS:
        rows = read_csv(os.path.join(run_dir(mode, seed), filename))
        stack.append([float(r[key]) for r in rows])
        steps = [int(r["step"]) for r in rows]
    length = min(len(v) for v in stack)
    return np.array(steps[:length]), np.array([v[:length] for v in stack])


def band(values):
    mean = values.mean(axis=0)
    half = stats.t.ppf(0.975, values.shape[0] - 1) * values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
    return mean, half


def final_value(mode, key):
    out = []
    for seed in SEEDS:
        rows = read_csv(os.path.join(run_dir(mode, seed), "validation.csv"))
        out.append(float(rows[-1][key]))
    return np.array(out)


def best_value(mode, key):
    out = []
    for seed in SEEDS:
        path = os.path.join(run_dir(mode, seed), "evaluation_summary.json")
        out.append(json.load(open(path, encoding="utf-8"))[key])
    return np.array(out, dtype=np.float64)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTDIR, "%s.%s" % (name, ext)))
    plt.close(fig)
    print("wrote %s.png / .pdf" % name)


# ------------------------------------------------------------------- figures
def fig01_success(_):
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for mode in MODES:
        steps, values = curves(mode, "success")
        mean, half = band(values)
        ax.plot(steps, mean, marker="o", color=COLOR[mode], label=LABEL[mode], linewidth=2.2)
        ax.fill_between(steps, mean - half, mean + half, color=COLOR[mode], alpha=0.18, linewidth=0)
    ax.set_xlabel("环境步数")
    ax.set_ylabel("held-out 成功率")
    ax.set_ylim(-0.03, 1.0)
    ax.set_title("held-out 成功率（100 个固定场景：1-2 障碍，100% 阻断直线）\n"
                 "均值 ± 95% CI，n=5 seeds", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "01_success_curve_ci")


def fig02_per_seed(_):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharey=True)
    for ax, mode in zip(axes, MODES):
        steps, values = curves(mode, "success")
        for index, seed in enumerate(SEEDS):
            ax.plot(steps, values[index], color=COLOR[mode], alpha=0.45, linewidth=1.2,
                    label="seed %d" % seed if False else None)
        mean, _ = band(values)
        ax.plot(steps, mean, color=COLOR[mode], linewidth=2.8)
        ax.set_title("%s   最终 %.2f ± %.2f" % (LABEL[mode], values[:, -1].mean(),
                                                stats.t.ppf(0.975, 4) * values[:, -1].std(ddof=1) / np.sqrt(5)),
                     fontsize=9.5)
        ax.set_xlabel("环境步数")
        ax.set_ylim(-0.03, 1.0)
    axes[0].set_ylabel("held-out 成功率")
    fig.suptitle("逐 seed 曲线（细线）与均值（粗线）：M1 的跨 seed 离散度明显大于 M3",
                 fontsize=11)
    save(fig, "02_success_per_seed")


def fig03_collision(_):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for mode in MODES:
        steps, values = curves(mode, "collision")
        mean, half = band(values)
        axes[0].plot(steps, mean, marker="o", color=COLOR[mode], label=LABEL[mode], linewidth=2.2)
        axes[0].fill_between(steps, mean - half, mean + half, color=COLOR[mode], alpha=0.18, linewidth=0)
        steps, values = curves(mode, "timeout")
        mean, half = band(values)
        axes[1].plot(steps, mean, marker="o", color=COLOR[mode], linewidth=2.2)
        axes[1].fill_between(steps, mean - half, mean + half, color=COLOR[mode], alpha=0.18, linewidth=0)
    axes[0].set_ylabel("碰撞率")
    axes[0].set_xlabel("环境步数")
    axes[0].set_title("碰撞率", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("超时率")
    axes[1].set_xlabel("环境步数")
    axes[1].set_title("超时率", fontsize=10)
    fig.suptitle("碰撞率必须与超时率联读：M2 的低碰撞率来自 82% 的回合根本不动",
                 fontsize=11)
    save(fig, "03_collision_and_timeout")


def fig04_infeasible(_):
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for mode in ("external_qp", "diff_qp"):
        steps, values = curves(mode, "qp_infeasible_rate")
        mean, half = band(values)
        ax.plot(steps, mean, marker="o", color=COLOR[mode], label=LABEL[mode], linewidth=2.2)
        ax.fill_between(steps, mean - half, mean + half, color=COLOR[mode], alpha=0.18, linewidth=0)
        for index in range(values.shape[0]):
            ax.plot(steps, values[index], color=COLOR[mode], alpha=0.3, linewidth=0.9)
    ax.set_xlabel("环境步数")
    ax.set_ylabel("QP 不可行步占比")
    ax.set_title("同一个 QP 函数，唯一差别是 actor 梯度是否穿过它\n"
                 "M2 在 5/5 个 seed 上发散到 0.36-0.58，M3 稳定在 0.07",
                 fontsize=11)
    ax.legend(fontsize=9)
    save(fig, "04_qp_infeasible_ci")


def fig05_paired(_):
    pairs = [("diff_qp", "vanilla"), ("diff_qp", "external_qp")]
    metrics = [("success", "成功率", 1), ("collision", "碰撞率", -1)]
    fig, axes = plt.subplots(len(metrics), len(pairs), figsize=(11, 8))
    for row, (key, label, better) in enumerate(metrics):
        for column, (a, b) in enumerate(pairs):
            ax = axes[row][column]
            va, vb = final_value(a, key), final_value(b, key)
            for index, seed in enumerate(SEEDS):
                gain = (va[index] - vb[index]) * better
                ax.plot([0, 1], [vb[index], va[index]], marker="o", markersize=5,
                        color="#55A868" if gain > 0 else ("#999999" if gain == 0 else "#C44E52"),
                        alpha=0.85, linewidth=1.4)
                ax.annotate("s%d" % seed, (1, va[index]), textcoords="offset points",
                            xytext=(6, -3), fontsize=7)
            ax.plot([0, 1], [vb.mean(), va.mean()], color="k", linewidth=2.6, zorder=4)
            try:
                p = stats.wilcoxon(va, vb).pvalue
            except Exception:
                p = float("nan")
            signs = "".join("+" if (x - y) * better > 0 else ("=" if x == y else "-")
                            for x, y in zip(va, vb))
            wins = signs.count("+")
            ax.set_xticks([0, 1])
            ax.set_xticklabels([SHORT[b], SHORT[a]])
            ax.set_xlim(-0.25, 1.35)
            ax.set_ylabel(label)
            ax.set_title("%s vs %s / %s\n%d/5 个 seed 支持 %s   p=%.4f   符号 %s"
                         % (SHORT[a], SHORT[b], label, wins, SHORT[a], p, signs), fontsize=9)
    fig.suptitle("按 seed 配对比较（200k 同步数）。绿=支持 M3，红=反向，灰=持平。\n"
                 "n=5 时双侧 Wilcoxon 的最小可能 p 值就是 0.0625，因此本实验任何比较都不可能达到 p<0.05；"
                 "逐 seed 符号才是证据。", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    save(fig, "05_paired_by_seed")


def fig06_outcomes(_):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharey=True)
    for ax, mode in zip(axes, MODES):
        steps, success = curves(mode, "success")
        _, collision = curves(mode, "collision")
        _, timeout = curves(mode, "timeout")
        s, c, t = success.mean(0), collision.mean(0), timeout.mean(0)
        other = np.clip(1.0 - s - c - t, 0.0, 1.0)
        ax.stackplot(steps, s, c, t, other,
                     colors=["#55A868", "#C44E52", "#8C8C8C", "#DDDDDD"],
                     labels=["成功", "碰撞", "超时", "停错位置"])
        ax.set_title(LABEL[mode], fontsize=9.5)
        ax.set_xlabel("环境步数")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("回合结局占比（5 seeds 均值）")
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("回合结局构成：M2 后期几乎全部超时", fontsize=11)
    save(fig, "06_outcome_composition")


def fig07_selection_bias(_):
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    width = 0.34
    x = np.arange(len(MODES))
    matched = [final_value(m, "success") for m in MODES]
    best = [best_value(m, "success") for m in MODES]
    ci = lambda v: stats.t.ppf(0.975, 4) * v.std(ddof=1) / np.sqrt(5)
    ax.bar(x - width / 2, [v.mean() for v in matched], width, yerr=[ci(v) for v in matched],
           capsize=4, color=[COLOR[m] for m in MODES], label="200k 同步数（主表）")
    ax.bar(x + width / 2, [v.mean() for v in best], width, yerr=[ci(v) for v in best],
           capsize=4, color=[COLOR[m] for m in MODES], alpha=0.45, hatch="//",
           label="best.pt（8 个验证点取最大）")
    for index, mode in enumerate(MODES):
        spread = matched[index].std(ddof=1)
        ax.annotate("跨 seed 标准差 %.3f" % spread, (index, 0.03), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[m].split(" ")[0] + "\n" + LABEL[m].split(" ")[1] for m in MODES])
    ax.set_ylabel("held-out 成功率")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9)
    ax.set_title("为什么主表必须用同步数：best.pt 是 8 个带噪声验证点的最大值，\n"
                 "曲线越抖的方法拿到的最大值越高——M1 因此在 best.pt 上反超 M3",
                 fontsize=10.5)
    save(fig, "07_best_vs_matched_step")


def fig08_ablation(path):
    if not os.path.exists(path):
        print("skip 08: no ablation cache")
        return
    rows = read_csv(path)
    for r in rows:
        r["step"] = int(r["step"])
        r["collision"] = float(r["collision"])
    steps = sorted({r["step"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
    handles = []
    for mode in MODES:
        for ax, tag, title in ((axes[0], "off", "关闭安全层：策略自身的碰撞率"),
                               (axes[1], "on", "开启安全层")):
            values = np.array([[r["collision"] for r in rows
                                if r["mode"] == mode and r["step"] == st and r["safety_layer"] == tag]
                               for st in steps]).squeeze()
            if values.ndim != 2:
                continue
            values = values.T
            mean, half = band(values)
            line, = ax.plot(steps, mean, marker="o", color=COLOR[mode], linewidth=2.2,
                            label=LABEL[mode])
            ax.fill_between(steps, mean - half, mean + half, color=COLOR[mode],
                            alpha=0.18, linewidth=0)
            if tag == "off":
                handles.append(line)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("环境步数")
            ax.set_ylim(-0.03, 1.05)
    axes[0].set_ylabel("碰撞率")
    fig.suptitle("安全层是拐杖还是增益：同一 actor，只切换推理时是否启用 CBF-QP。"
                 "均值 ± 95% CI，n=5", fontsize=10.5)
    fig.legend(handles=handles, labels=[LABEL[m] for m in MODES], fontsize=9,
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    save(fig, "08_safety_layer_ablation")


def fig09_tradeoff(_):
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    ci = lambda v: stats.t.ppf(0.975, 4) * v.std(ddof=1) / np.sqrt(5)
    for mode in MODES:
        s, c = final_value(mode, "success"), final_value(mode, "collision")
        ax.scatter(c, s, s=42, color=COLOR[mode], alpha=0.55, zorder=3)
        ax.errorbar(c.mean(), s.mean(), xerr=ci(c), yerr=ci(s), fmt="o", markersize=13,
                    color=COLOR[mode], capsize=5, linewidth=2.2, zorder=5,
                    markeredgecolor="k", markeredgewidth=0.6, label=LABEL[mode])
    ax.set_xlabel("碰撞率")
    ax.set_ylabel("成功率")
    ax.set_ylim(-0.05, 1.0)
    ax.set_title("安全-任务权衡（200k 同步数）。小点 = 单个 seed，大点 = 均值 ± 95% CI\n"
                 "M2 位于左下角：低碰撞率是瘫痪的副产物，不是安全性",
                 fontsize=10.5)
    ax.legend(fontsize=8.5, loc="lower right")
    save(fig, "09_safety_task_tradeoff")


def fig10_return(_):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for mode in MODES:
        steps, distance = curves(mode, "final_distance_true")
        mean, half = band(distance * 100.0)
        axes[0].plot(steps, mean, marker="o", color=COLOR[mode], label=LABEL[mode], linewidth=2.2)
        axes[0].fill_between(steps, mean - half, mean + half, color=COLOR[mode], alpha=0.18, linewidth=0)
        steps, epsteps = curves(mode, "steps")
        mean, half = band(epsteps)
        axes[1].plot(steps, mean, marker="o", color=COLOR[mode], linewidth=2.2)
        axes[1].fill_between(steps, mean - half, mean + half, color=COLOR[mode], alpha=0.18, linewidth=0)
    axes[0].axhline(7.0, color="k", linestyle=":", linewidth=1)
    axes[0].annotate("成功判据 7 cm", (0.03, 8.5), xycoords=("axes fraction", "data"), fontsize=8)
    axes[0].set_ylabel("平均终点真实距离 (cm)")
    axes[0].set_xlabel("环境步数")
    axes[0].set_title("终点精度", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[1].axhline(123, color="k", linestyle=":", linewidth=1)
    axes[1].annotate("直线最优 123 步", (0.03, 133), xycoords=("axes fraction", "data"), fontsize=8)
    axes[1].set_ylabel("平均回合步数")
    axes[1].set_xlabel("环境步数")
    axes[1].set_title("路径效率（300 步为超时上限）", fontsize=10)
    fig.suptitle("任务质量：均值 ± 95% CI，n=5", fontsize=11)
    save(fig, "10_task_quality")


def write_summary():
    ci = lambda v: float(stats.t.ppf(0.975, 4) * v.std(ddof=1) / np.sqrt(5))
    out = {"experiment": "G4", "seeds": list(SEEDS), "steps": 200000,
           "config": "configs/main_80cm.json (alpha_min=0.01)",
           "note": "cross-method numbers use the matched 200k step, not best.pt; "
                   "with n=5 the two-sided Wilcoxon floor is p=0.0625",
           "matched_step": {}, "best_checkpoint": {}, "paired": []}
    for mode in MODES:
        out["matched_step"][mode] = {}
        for key in ("success", "collision", "timeout", "final_distance_true",
                    "qp_intervention_rate", "qp_infeasible_rate", "steps"):
            v = final_value(mode, key)
            out["matched_step"][mode][key] = {"per_seed": v.tolist(),
                                              "mean": float(v.mean()), "ci95": ci(v)}
        v = best_value(mode, "success")
        c = best_value(mode, "collision")
        out["best_checkpoint"][mode] = {
            "success": {"per_seed": v.tolist(), "mean": float(v.mean()), "ci95": ci(v)},
            "collision": {"per_seed": c.tolist(), "mean": float(c.mean()), "ci95": ci(c)},
            "steps": [json.load(open(os.path.join(run_dir(mode, s), "evaluation_summary.json"),
                                     encoding="utf-8"))["checkpoint_step"] for s in SEEDS]}
    for a, b in (("diff_qp", "vanilla"), ("diff_qp", "external_qp"), ("vanilla", "external_qp")):
        for key in ("success", "collision", "timeout"):
            va, vb = final_value(a, key), final_value(b, key)
            try:
                p = float(stats.wilcoxon(va, vb).pvalue)
            except Exception:
                p = None
            difference = va - vb
            # Seeded: an unseeded bootstrap moved the reported CI in the third
            # decimal between runs, which is not something a thesis number should do.
            rng = np.random.default_rng(20260823)
            boot = np.array([np.mean(difference[rng.integers(0, 5, 5)]) for _ in range(10000)])
            out["paired"].append({
                "a": a, "b": b, "metric": key,
                "mean_difference": float(difference.mean()),
                "per_seed_difference": difference.tolist(),
                "seeds_favouring_a": int(sum(1 for d in difference if d > 0)),
                "wilcoxon_p": p,
                "bootstrap_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]})
    with open(os.path.join(OUTDIR, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2, ensure_ascii=False)
    print("wrote summary.json")


def main(argv=None) -> int:
    global RUNS, OUTDIR
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default=RUNS,
                        help="run root to read, e.g. runs_g4 or runs_g4_ttl8")
    parser.add_argument("--out", default=HERE, help="directory to write figures into")
    args = parser.parse_args(argv)

    RUNS, OUTDIR = os.path.abspath(args.runs), os.path.abspath(args.out)
    os.makedirs(OUTDIR, exist_ok=True)
    print("reading " + RUNS)
    print("writing " + OUTDIR)
    # The ablation cache is produced per run set and is optional; fig08 skips
    # itself when the set it is pointed at has none.
    ablation = os.path.join(OUTDIR, "ablation_5seeds.csv")
    fig01_success(None)
    fig02_per_seed(None)
    fig03_collision(None)
    fig04_infeasible(None)
    fig05_paired(None)
    fig06_outcomes(None)
    fig07_selection_bias(None)
    fig08_ablation(ablation)
    fig09_tradeoff(None)
    fig10_return(None)
    write_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
