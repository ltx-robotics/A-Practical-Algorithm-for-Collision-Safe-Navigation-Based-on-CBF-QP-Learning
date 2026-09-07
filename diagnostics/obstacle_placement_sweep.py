

import math
import sys

import numpy as np

from real_runtime_parity import PROJECT_ROOT, default_checkpoint  # noqa: F401,E402

sys.path.insert(0, PROJECT_ROOT)

from safe_alvik.config import Config  # noqa: E402
from safe_alvik.environment import AlvikEnv, Scenario  # noqa: E402
from safe_alvik.io_utils import load_checkpoint  # noqa: E402
from safe_alvik.sac import PolicyRunner  # noqa: E402

PLACEMENTS = [([[0.00, 0.00, 0.05]], 0.0, "A (0.00, 0.00)"),
              ([[0.00, 0.00, 0.05]], 45.0, "A, nose at the goal"),
              ([[-0.05, 0.06, 0.06]], 0.0, "B (-0.05, 0.06)"),
              ([[0.10, -0.10, 0.05]], 0.0, "on the driven leg (0.10,-0.10)"),
              ([[0.15, -0.05, 0.05]], 0.0, "on the driven leg (0.15,-0.05)"),
              ([[-0.127, 0.127, 0.05], [0.127, -0.127, 0.05]], 0.0, "C (+-0.127)")]


def batch(cfg, policy, obstacles, heading_deg, episodes):
    env = AlvikEnv(cfg, seed=1)
    start = np.array(cfg.map.start_pose)
    goal = np.array(cfg.map.goal_position)
    tally = {"success": 0, "collision": 0, "timeout": 0, "engaged": 0}
    clearances = []

    for _ in range(episodes):
        # S2 sampling: the fixed start with its jitter, which is what the
        # policy trained on and what the RESET packet reproduces.
        jitter = env.rng.uniform(-cfg.curriculum[-1].start_pose_jitter_m,
                                 cfg.curriculum[-1].start_pose_jitter_m, size=2)
        heading = math.radians(heading_deg) + env.rng.uniform(-1.0, 1.0) * math.radians(
            cfg.curriculum[-1].start_heading_jitter_deg)
        scenario = Scenario(start_pose=np.array([start[0] + jitter[0],
                                                 start[1] + jitter[1], heading]),
                            goal=goal,
                            obstacles=np.asarray(obstacles, float).reshape(-1, 3))
        obs = env.reset(scenario)
        engaged, clearance = False, 9.9
        while True:
            action = policy.act(obs, env.constraints, deterministic=True)["action_real"]
            obs, _, done, info = env.step(action)
            engaged = engaged or bool(env.constraints[2].sum() > 0)
            surface = (np.linalg.norm(env.true_pose[:2] - scenario.obstacles[:, :2], axis=1)
                       - scenario.obstacles[:, 2])
            clearance = min(clearance, float(surface.min()) - cfg.robot.body_radius_m)
            if done:
                break
        tally["engaged"] += engaged
        tally["success"] += bool(info["success"])
        tally["collision"] += bool(info["collision"])
        tally["timeout"] += not (info["success"] or info["collision"])
        clearances.append(clearance)
    return tally, float(np.median(clearances))


def main(argv):
    checkpoint = argv[1] if len(argv) > 1 else default_checkpoint()
    episodes = int(argv[2]) if len(argv) > 2 else 40
    state = load_checkpoint(checkpoint)
    cfg = Config.from_dict(state["config"])
    policy = PolicyRunner(cfg, "diff_qp", state["actor"])
    print("checkpoint %s, %d episodes per placement, randomization ON\n"
          % (checkpoint, episodes))
    print("%-31s %9s %8s %10s %8s %11s"
          % ("placement", "CBF used", "success", "collision", "timeout", "clearance"))
    for obstacles, heading, label in PLACEMENTS:
        tally, clearance = batch(cfg, policy, obstacles, heading, episodes)
        print("%-31s %6d/%-3d %6d/%-3d %8d %9d %9.3f m"
              % (label, tally["engaged"], episodes, tally["success"], episodes,
                 tally["collision"], tally["timeout"], clearance))
    print("\nPick a placement with CBF used on every episode for the demonstration "
          "video;\notherwise the safety layer may sit idle for the whole run.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
