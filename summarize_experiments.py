
import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

from safe_alvik.config import MODES
from safe_alvik.evaluation import better_than
from safe_alvik.io_utils import load_json, save_json

METRICS = [
    "success", "collision", "timeout", "any_cbf_violation", "cbf_violation_steps",
    "cbf_violation_rate", "final_distance_true",
    "return", "steps", "detour_ratio", "path_length_m", "min_true_clearance_m",
    "qp_intervention_rate", "qp_infeasible_rate", "mean_abs_delta_v",
    "mean_abs_delta_omega", "qp_solve_ms_mean", "qp_solve_ms_max", "blocked_success",
]


def collect(root: str) -> List[Dict[str, Any]]:
    rows = []
    for mode in MODES:
        mode_dir = os.path.join(root, mode)
        if not os.path.isdir(mode_dir):
            continue
        for name in sorted(os.listdir(mode_dir)):
            run_dir = os.path.join(mode_dir, name)
            summary_path = os.path.join(run_dir, "evaluation_summary.json")
            run_path = os.path.join(run_dir, "run_summary.json")
            if not os.path.exists(summary_path):
                continue
            summary = load_json(summary_path)
            row: Dict[str, Any] = {"mode": mode, "run": name, "run_dir": run_dir}
            if os.path.exists(run_path):
                run_summary = load_json(run_path)
                row["seed"] = run_summary.get("seed")
                row["wall_clock_s"] = run_summary.get("wall_clock_s")
                row["steps_per_second"] = run_summary.get("steps_per_second")
                row["total_steps"] = run_summary.get("total_steps")
                row["final_stage"] = run_summary.get("final_stage")
            for metric in METRICS:
                row[metric] = summary.get(metric)
            row["checkpoint"] = summary.get("checkpoint")
            row["checkpoint_step"] = summary.get("checkpoint_step")
            rows.append(row)
    return rows


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for mode in MODES:
        subset = [r for r in rows if r["mode"] == mode]
        if not subset:
            continue
        record: Dict[str, Any] = {"mode": mode, "seeds": len(subset)}
        for metric in METRICS:
            values = np.array([r[metric] for r in subset if r.get(metric) is not None],
                              dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size == 0:
                record[metric + "_mean"] = ""
                continue
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            half = 0.0
            if values.size > 1:
                try:
                    from scipy import stats
                    half = float(stats.t.ppf(0.975, values.size - 1) * std / np.sqrt(values.size))
                except Exception:
                    half = float(1.96 * std / np.sqrt(values.size))
            record[metric + "_mean"] = mean
            record[metric + "_std"] = std
            record[metric + "_ci95"] = half
        out.append(record)
    return out


def paired(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    try:
        from scipy import stats
    except Exception:
        stats = None
    by_mode = {mode: {r.get("seed"): r for r in rows if r["mode"] == mode} for mode in MODES}
    comparisons = []
    for i, first in enumerate(MODES):
        for second in MODES[i + 1:]:
            shared = sorted(set(by_mode.get(first, {})) & set(by_mode.get(second, {})))
            shared = [s for s in shared if s is not None]
            if len(shared) < 2:
                continue
            for metric in METRICS:
                a = np.array([by_mode[first][s].get(metric) for s in shared], dtype=np.float64)
                b = np.array([by_mode[second][s].get(metric) for s in shared], dtype=np.float64)
                keep = np.isfinite(a) & np.isfinite(b)
                if keep.sum() < 2:
                    continue
                a, b = a[keep], b[keep]
                record = {"metric": metric, "mode_a": first, "mode_b": second,
                          "n_seeds": int(keep.sum()),
                          "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
                          "mean_difference": float(np.mean(a - b))}
                if stats is not None and np.any(a != b):
                    try:
                        record["wilcoxon_p"] = float(stats.wilcoxon(a, b).pvalue)
                    except Exception:
                        record["wilcoxon_p"] = ""
                boot = np.array([np.mean((a - b)[np.random.randint(0, len(a), len(a))])
                                 for _ in range(2000)])
                record["bootstrap_ci_low"] = float(np.percentile(boot, 2.5))
                record["bootstrap_ci_high"] = float(np.percentile(boot, 97.5))
                comparisons.append(record)
    return comparisons


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="runs")
    parser.add_argument("--output", default=os.path.join("results", "main_80cm"))
    args = parser.parse_args(argv)

    rows = collect(args.root)
    if not rows:
        print("no evaluated runs found under %s; run evaluate.py first" % args.root)
        return 1
    os.makedirs(args.output, exist_ok=True)
    write_csv(os.path.join(args.output, "summary_by_seed.csv"), rows)
    write_csv(os.path.join(args.output, "aggregate_summary.csv"), aggregate(rows))
    write_csv(os.path.join(args.output, "paired_comparisons.csv"), paired(rows))

    best: Optional[Dict[str, Any]] = None
    for row in rows:
        if row["mode"] != "diff_qp":
            continue
        candidate = {k: row.get(k) for k in ("success", "collision",
                                             "final_distance_true", "return")}
        candidate = {k: (float(v) if v is not None else float("nan"))
                     for k, v in candidate.items()}
        if best is None or better_than(candidate, best["metrics"]):
            best = {"metrics": candidate, "run_dir": row["run_dir"],
                    "seed": row.get("seed"), "checkpoint": row.get("checkpoint")}
    if best is not None:
        save_json(os.path.join(args.output, "best_diff_qp.json"), best)
        print("best diff_qp: seed=%s success=%.3f -> %s"
              % (best["seed"], best["metrics"]["success"], best["checkpoint"]))
    print("wrote %d run rows to %s" % (len(rows), args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
