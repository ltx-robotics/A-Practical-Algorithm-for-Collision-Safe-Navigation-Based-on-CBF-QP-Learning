

import argparse
import csv
import glob
import os
import sys

MODES = ("vanilla", "external_qp", "diff_qp")


def load(root, mode, name, seed):
    pattern = os.path.join(root, mode, "%s_seed%d_*" % (name, seed), "validation.csv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return []
    with open(matches[-1], encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="runs")
    parser.add_argument("--name", default="main")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--last", type=int, default=2,
                        help="how many validation points to print (0 = all)")
    parser.add_argument("--total-steps", type=int, default=200000)
    args = parser.parse_args(argv)

    rows = {mode: load(args.root, mode, args.name, args.seed) for mode in MODES}
    available = [len(v) for v in rows.values()]
    if not available or min(available) == 0:
        print("no validation rows yet")
        return 1
    common = min(available)
    start = 0 if args.last <= 0 else max(0, common - args.last)

    header = ("%-12s %7s %6s %8s %10s %8s %8s %7s %10s %10s" %
              ("method", "step", "stage", "success", "collision", "timeout",
               "dist_m", "steps", "qp_interv", "qp_infeas"))
    print(header)
    for index in range(start, common):
        print("-" * len(header))
        for mode in MODES:
            r = rows[mode][index]
            print("%-12s %7s %6s %8.2f %10.2f %8.2f %8.3f %7.0f %10.3f %10.3f" % (
                mode, r["step"], r["stage"].split("_")[0], float(r["success"]),
                float(r["collision"]), float(r["timeout"]),
                float(r["final_distance_true"]), float(r["steps"]),
                float(r["qp_intervention_rate"]), float(r["qp_infeasible_rate"])))

    print()
    print("%-12s %10s %14s %14s" % ("method", "step", "elapsed_min", "eta_min"))
    for mode in MODES:
        r = rows[mode][-1]
        step, elapsed = int(r["step"]), float(r["wall_clock_s"]) / 60.0
        remaining = elapsed * (args.total_steps - step) / max(step, 1)
        print("%-12s %10d %14.1f %14.1f" % (mode, step, elapsed, remaining))
    return 0


if __name__ == "__main__":
    sys.exit(main())
