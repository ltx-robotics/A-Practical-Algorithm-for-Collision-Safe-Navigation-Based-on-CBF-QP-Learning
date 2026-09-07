"""Run the Webots validation over all three fixed layouts and aggregate (G6).

    python webots_alvik/run_webots_eval.py [--episodes 20] [--checkpoint PATH]

Gate (PLAN.MD 9): success >= 0.70 and zero collisions across the three layouts,
otherwise the policy does not go to the real robot.

Each layout runs in its own headless Webots process. The controller writes
outputs/layout_<X>.csv; this script merges them into outputs/summary.csv and
prints the gate verdict.
"""

import argparse
import csv
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LAYOUTS = ("A", "B", "C")

WEBOTS = os.environ.get("WEBOTS_EXE") or os.path.join(
    os.environ.get("WEBOTS_HOME", r"D:\Webots"), "msys64", "mingw64", "bin", "webots.exe")


def kill_stale_webots() -> None:
    """A Webots instance that has not fully exited makes the next one hang
    instead of failing, which looks identical to a slow run. Clear it first."""
    if os.name != "nt":
        return
    for image in ("webots-bin.exe", "webots.exe"):
        subprocess.call(["taskkill", "/F", "/IM", image],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_layout(layout: str, episodes: int, checkpoint, seed: int, gui: bool,
               timeout_s: float) -> int:
    world = os.path.join(HERE, "worlds", "alvik_layout_%s.wbt" % layout)
    command = [WEBOTS, "--batch", "--stdout", "--stderr", world]
    command[1:1] = ["--mode=realtime"] if gui else ["--mode=fast", "--no-rendering", "--minimize"]
    # Controller arguments come from the world file; episodes/checkpoint/seed are
    # passed through the environment so the generated worlds stay untouched.
    env = dict(os.environ)
    env["ALVIK_EPISODES"] = str(episodes)
    env["ALVIK_SEED"] = str(seed)
    if checkpoint:
        env["ALVIK_CHECKPOINT"] = checkpoint
    print("=" * 70, flush=True)
    print("layout %s: %s" % (layout, " ".join(command)), flush=True)
    kill_stale_webots()
    try:
        return subprocess.call(command, env=env, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print("layout %s timed out after %.0f s; killing Webots" % (layout, timeout_s),
              flush=True)
        kill_stale_webots()
        return 124


def load_rows(layout: str):
    path = os.path.join(HERE, "outputs", "layout_%s.csv" % layout)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layouts", nargs="+", default=list(LAYOUTS))
    parser.add_argument("--gui", action="store_true", help="watch it run instead of headless")
    parser.add_argument("--timeout", type=float, default=1200.0,
                        help="per-layout wall-clock limit in seconds")
    args = parser.parse_args(argv)

    if not os.path.exists(WEBOTS):
        raise SystemExit("Webots not found at %s (set WEBOTS_HOME or WEBOTS_EXE)" % WEBOTS)

    for layout in args.layouts:
        code = run_layout(layout, args.episodes, args.checkpoint, args.seed,
                          args.gui, args.timeout)
        if code != 0:
            print("layout %s exited with code %d" % (layout, code))

    merged, totals = [], {}
    for layout in args.layouts:
        rows = load_rows(layout)
        merged.extend(rows)
        if not rows:
            continue
        n = len(rows)
        totals[layout] = {
            "episodes": n,
            "success": sum(int(r["success"]) for r in rows) / n,
            "collision": sum(int(r["collision"]) for r in rows) / n,
            "timeout": sum(int(r["timeout"]) for r in rows) / n,
            "final_distance_true": sum(float(r["final_distance_true"]) for r in rows) / n,
            "detour_ratio": sum(float(r["detour_ratio"]) for r in rows) / n,
            "qp_intervention_rate": sum(float(r["qp_intervention_rate"]) for r in rows) / n,
            "qp_infeasible_rate": sum(float(r["qp_infeasible_rate"]) for r in rows) / n,
            "qp_solve_ms_max": max(float(r["qp_solve_ms_max"]) for r in rows),
            "min_clearance": min(float(r["min_clearance"]) for r in rows),
        }

    if merged:
        out = os.path.join(HERE, "outputs", "summary.csv")
        with open(out, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(merged[0].keys()))
            writer.writeheader()
            writer.writerows(merged)
        n = len(merged)
        overall = {
            "episodes": n,
            "success": sum(int(r["success"]) for r in merged) / n,
            "collision": sum(int(r["collision"]) for r in merged) / n,
            "timeout": sum(int(r["timeout"]) for r in merged) / n,
            "final_distance_true": sum(float(r["final_distance_true"]) for r in merged) / n,
            "qp_solve_ms_max": max(float(r["qp_solve_ms_max"]) for r in merged),
            "min_clearance": min(float(r["min_clearance"]) for r in merged),
        }
        totals["overall"] = overall
        with open(os.path.join(HERE, "outputs", "summary.json"), "w", encoding="utf-8") as handle:
            json.dump(totals, handle, indent=2, ensure_ascii=False)

        print("\n" + "=" * 70)
        print("%-9s %8s %9s %10s %8s %10s %11s %10s" % (
            "layout", "episodes", "success", "collision", "timeout", "dist_m",
            "qp_infeas", "qp_ms_max"))
        for layout in args.layouts:
            if layout not in totals:
                continue
            t = totals[layout]
            print("%-9s %8d %9.2f %10.2f %8.2f %10.3f %11.3f %10.2f" % (
                layout, t["episodes"], t["success"], t["collision"], t["timeout"],
                t["final_distance_true"], t["qp_infeasible_rate"], t["qp_solve_ms_max"]))
        print("%-9s %8d %9.2f %10.2f %8.2f %10.3f %11s %10.2f" % (
            "overall", overall["episodes"], overall["success"], overall["collision"],
            overall["timeout"], overall["final_distance_true"], "-", overall["qp_solve_ms_max"]))
        print("worst true clearance: %+.4f m" % overall["min_clearance"])

        passed = overall["success"] >= 0.70 and overall["collision"] == 0.0
        print("\nG6 gate (success >= 0.70 且 collision == 0): %s"
              % ("PASS" if passed else "FAIL"))
        return 0 if passed else 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
