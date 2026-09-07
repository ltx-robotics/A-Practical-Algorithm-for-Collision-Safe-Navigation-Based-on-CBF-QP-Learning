"""Train one method with one seed.

    python train.py --config configs/main_80cm.json --mode diff_qp --seed 42 --device cuda
"""

import argparse
import os
import sys

import torch

from safe_alvik.config import Config, MODES, rescale_schedule
from safe_alvik.io_utils import make_run_dir, save_json
from safe_alvik.trainer import Trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="JSON config; defaults are used if omitted")
    parser.add_argument("--mode", required=True, choices=list(MODES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=None,
                        help="override the environment-step budget (rescales the schedule)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--name", default="main")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = Config.from_json(args.config)
    if args.steps is not None:
        cfg = rescale_schedule(cfg, args.steps)

    run_dir = make_run_dir(args.run_root, args.mode, args.name, args.seed)
    save_json(os.path.join(run_dir, "config.json"),
              {"config": cfg.to_dict(), "mode": args.mode, "seed": args.seed,
               "device": args.device, "source_config": args.config})

    trainer = Trainer(cfg, args.mode, args.seed, run_dir,
                      device=args.device, verbose=not args.quiet)
    summary = trainer.train()
    if not args.quiet:
        print("finished %s seed=%d in %.1f s (%.1f steps/s) -> %s"
              % (args.mode, args.seed, summary["wall_clock_s"],
                 summary["steps_per_second"], run_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
