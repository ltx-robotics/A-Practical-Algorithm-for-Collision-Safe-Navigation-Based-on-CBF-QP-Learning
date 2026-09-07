

import argparse
import csv
import glob
import io
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, PROJECT_ROOT)

from safe_alvik.config import Config  # noqa: E402
from safe_alvik.environment import AlvikEnv  # noqa: E402
from safe_alvik.evaluation import evaluate, holdout_scenarios  # noqa: E402
from safe_alvik.io_utils import load_checkpoint  # noqa: E402
from safe_alvik.sac import PolicyRunner  # noqa: E402

EXTRA = ["final_distance_true", "cbf_violation_rate", "any_cbf_violation",
         "detour_ratio", "min_true_clearance_m", "qp_intervention_rate",
         "qp_infeasible_rate", "mean_abs_delta_v", "mean_abs_delta_omega"]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="runs_g4_ttl8")
    parser.add_argument("--out", default=None,
                        help="CSV path (default <runs>_ablation.csv next to the run root)")
    parser.add_argument("--modes", nargs="+", default=["vanilla", "external_qp", "diff_qp"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--steps", nargs="+", type=int,
                        default=[50000, 100000, 150000, 200000])
    parser.add_argument("--episodes", type=int, default=100)
    return parser


def checkpoint_path(runs, mode, seed, step):
    matches = sorted(glob.glob(os.path.join(
        PROJECT_ROOT, runs, mode, "main_seed%d_*" % seed,
        "checkpoints", "step_%07d.pt" % step)))
    return matches[-1] if matches else None


def score(state, layer_on, episodes, seed):
    """一次评估。off / on 用同一个 actor、同一批场景、同一个环境种子。"""
    cfg = Config.from_dict(state["config"])
    runner = PolicyRunner(cfg, "diff_qp" if layer_on else "vanilla", state["actor"], seed=seed)
    env = AlvikEnv(cfg, seed=seed + 500000)
    summary, _ = evaluate(env, runner, holdout_scenarios(cfg, episodes))
    return summary


def main(argv=None):
    args = build_parser().parse_args(argv)
    out = args.out or os.path.join(PROJECT_ROOT, "%s_ablation.csv" % args.runs)
    rows = []
    started = time.time()
    total = len(args.modes) * len(args.seeds) * len(args.steps)
    done = 0

    for mode in args.modes:
        for seed in args.seeds:
            for step in args.steps:
                path = checkpoint_path(args.runs, mode, seed, step)
                done += 1
                if path is None:
                    print("  missing checkpoint: %s seed %d step %d" % (mode, seed, step))
                    continue
                state = load_checkpoint(path)
                if state.get("mode") != mode:
                    raise SystemExit("checkpoint mode %r != %r at %s"
                                     % (state.get("mode"), mode, path))
                for layer_on in (False, True):
                    summary = score(state, layer_on, args.episodes, seed)
                    row = {"mode": mode, "seed": seed, "step": step,
                           "safety_layer": "on" if layer_on else "off",
                           "success": summary["success"], "collision": summary["collision"],
                           "timeout": summary["timeout"]}
                    row.update({k: summary.get(k) for k in EXTRA})
                    rows.append(row)
                elapsed = time.time() - started
                print("[%3d/%3d] %-12s seed %d step %6d   off %.2f/%.2f   on %.2f/%.2f"
                      "   (success/collision)   %.0f s elapsed, ~%.0f s left"
                      % (done, total, mode, seed, step,
                         rows[-2]["success"], rows[-2]["collision"],
                         rows[-1]["success"], rows[-1]["collision"],
                         elapsed, elapsed / done * (total - done)), flush=True)

    if not rows:
        raise SystemExit("nothing evaluated")
    with io.open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote %d rows to %s (%.0f s)" % (len(rows), out, time.time() - started))
    return 0


if __name__ == "__main__":
    sys.exit(main())
