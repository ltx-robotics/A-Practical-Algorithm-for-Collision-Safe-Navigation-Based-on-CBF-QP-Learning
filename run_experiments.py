

import argparse
import os
import sys
import time

import torch

from safe_alvik.config import Config, MODES, rescale_schedule
from safe_alvik.io_utils import make_run_dir, save_json
from safe_alvik.trainer import Trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--name", default="main")
    parser.add_argument("--evaluate", action="store_true",
                        help="run the held-out evaluation on best.pt after each run")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    index = {"group": args.name, "timestamp": stamp, "modes": args.modes,
             "seeds": args.seeds, "device": args.device, "runs": []}

    for seed in args.seeds:
        for mode in args.modes:
            cfg = Config.from_json(args.config)
            if args.steps is not None:
                cfg = rescale_schedule(cfg, args.steps)
            run_dir = make_run_dir(args.run_root, mode, args.name, seed, stamp)
            save_json(os.path.join(run_dir, "config.json"),
                      {"config": cfg.to_dict(), "mode": mode, "seed": seed,
                       "device": args.device, "source_config": args.config,
                       "group": args.name, "timestamp": stamp})

            trainer = Trainer(cfg, mode, seed, run_dir, device=args.device,
                              verbose=not args.quiet)
            summary = trainer.train()
            record = {"mode": mode, "seed": seed, "run_dir": run_dir,
                      "wall_clock_s": summary["wall_clock_s"],
                      "steps_per_second": summary["steps_per_second"]}

            if args.evaluate:
                import evaluate as evaluate_module
                checkpoint = os.path.join(run_dir, "checkpoints", "best.pt")
                if os.path.exists(checkpoint):
                    evaluate_module.main(["--checkpoint", checkpoint,
                                          "--device", args.device,
                                          "--seed", str(seed)])
                    record["evaluated"] = True
            index["runs"].append(record)
            save_json(os.path.join(args.run_root, "group_%s_%s.json" % (args.name, stamp)), index)

    print("completed %d runs -> %s" % (len(index["runs"]), args.run_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
