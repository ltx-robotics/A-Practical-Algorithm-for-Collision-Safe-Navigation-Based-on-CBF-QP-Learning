"""Evaluate one checkpoint on the fixed held-out set.

    python evaluate.py --checkpoint runs/diff_qp/RUN/checkpoints/best.pt --episodes 100
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

from safe_alvik.config import Config
from safe_alvik.environment import AlvikEnv
from safe_alvik.evaluation import evaluate, holdout_scenarios
from safe_alvik.io_utils import load_checkpoint, save_json
from safe_alvik.sac import SACAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None,
                        help="directory for the CSV/JSON (defaults to the run directory)")
    parser.add_argument("--no-randomization", action="store_true",
                        help="disable domain randomization (diagnostic only, not a paper result)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    state = load_checkpoint(args.checkpoint, map_location=args.device)
    cfg = Config.from_dict(state["config"])
    if args.no_randomization:
        cfg.randomization.enabled = False

    agent = SACAgent(cfg, state["mode"], device=args.device, seed=args.seed)
    agent.load_state_dict(state)
    env = AlvikEnv(cfg, seed=args.seed + 500000)

    scenarios = holdout_scenarios(cfg, args.episodes)
    # Rollouts are single-sample and launch-bound, so score on the CPU copy.
    summary, episodes = evaluate(env, agent.eval_runner(), scenarios)
    summary["mode"] = state["mode"]
    summary["checkpoint"] = os.path.abspath(args.checkpoint)
    summary["checkpoint_step"] = state.get("step")
    summary["randomization"] = cfg.randomization.enabled

    output = args.output or os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint)))
    os.makedirs(output, exist_ok=True)
    csv_path = os.path.join(output, "evaluation_episodes.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episodes[0].keys()))
        writer.writeheader()
        for row in episodes:
            writer.writerow({k: (int(v) if isinstance(v, bool) else v) for k, v in row.items()})
    save_json(os.path.join(output, "evaluation_summary.json"), summary)

    print("%s  episodes=%d  success=%.3f  collision=%.3f  cbf_violation=%.4f  "
          "final_distance=%.3f m  intervention=%.3f"
          % (state["mode"], len(episodes), summary["success"], summary["collision"],
             summary["cbf_violation_rate"], summary["final_distance_true"],
             summary["qp_intervention_rate"]))
    print("wrote %s" % csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
