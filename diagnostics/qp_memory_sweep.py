

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safe_alvik.config import Config
from safe_alvik.environment import AlvikEnv
from safe_alvik.evaluation import holdout_scenarios
from safe_alvik.sac import SACAgent


def run(cfg: Config, mode: str, scenarios, seed: int):
    env = AlvikEnv(cfg, seed=seed)
    agent = SACAgent(cfg, mode, device="cpu", seed=seed)
    agent.rng = np.random.default_rng(seed)      # identical random action stream
    collisions = 0
    episodes_with_violation = 0
    violation_steps = 0
    infeasible = 0
    steps_total = 0
    worst = float("inf")
    for scenario in scenarios:
        obs = env.reset(scenario)
        done = False
        saw_violation = False
        info = {}
        while not done:
            out = agent.act(obs, env.constraints, random_action=True)
            obs, _, done, info = env.step(out["action_real"])
            steps_total += 1
            if info["true_clearance"] < 0.0:
                violation_steps += 1
                saw_violation = True
            worst = min(worst, info["true_clearance"])
            if not out["qp"].get("feasible", True):
                infeasible += 1
        collisions += int(info["collision"])
        episodes_with_violation += int(saw_violation)
    n = len(scenarios)
    return {
        "collision_rate": collisions / n,
        "episode_violation_rate": episodes_with_violation / n,
        "violation_steps": violation_steps,
        "worst_clearance_m": worst,
        "infeasible_rate": infeasible / max(steps_total, 1),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ttls", nargs="+", type=float, default=[1.4, 3.0, 5.0, 8.0])
    args = parser.parse_args(argv)

    scenarios = holdout_scenarios(Config(), args.episodes)
    header = "%-18s %10s %12s %12s %13s %10s" % (
        "setting", "collision", "any_viol", "viol_steps", "worst_clear", "infeas")
    print(header)
    print("-" * len(header))

    result = run(Config(), "vanilla", scenarios, args.seed)
    print("%-18s %9.1f%% %11.1f%% %12d %12.4f %9s"
          % ("no safety layer", 100 * result["collision_rate"],
             100 * result["episode_violation_rate"], result["violation_steps"],
             result["worst_clearance_m"], "-"))

    for ttl in args.ttls:
        cfg = Config()
        cfg.memory.ttl_s = ttl
        result = run(cfg, "external_qp", scenarios, args.seed)
        print("%-18s %9.1f%% %11.1f%% %12d %12.4f %9.2f%%"
              % ("QP, ttl %.1f s" % ttl, 100 * result["collision_rate"],
                 100 * result["episode_violation_rate"], result["violation_steps"],
                 result["worst_clearance_m"], 100 * result["infeasible_rate"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
