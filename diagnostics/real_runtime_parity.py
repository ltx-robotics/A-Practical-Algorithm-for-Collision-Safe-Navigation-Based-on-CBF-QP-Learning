

import glob
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, PROJECT_ROOT)

from real_robot.pc.protocol import Telemetry  # noqa: E402
from real_robot.pc.runtime import RealRobotRuntime  # noqa: E402
from safe_alvik.config import Config  # noqa: E402
from safe_alvik.environment import AlvikEnv, Scenario  # noqa: E402
from safe_alvik.io_utils import load_checkpoint  # noqa: E402
from safe_alvik.sac import PolicyRunner  # noqa: E402

TOLERANCE = 1e-6
SCENARIOS = [(np.zeros((0, 3)), "empty floor"),
             ([[0.00, 0.00, 0.05]], "layout A"),
             ([[-0.05, 0.06, 0.06]], "layout B"),
             ([[-0.127, 0.127, 0.05], [0.127, -0.127, 0.05]], "layout C")]


def default_checkpoint():
    for root in ("runs_g4_ttl8", "runs_g4"):
        best, key = None, None
        for path in sorted(glob.glob(os.path.join(PROJECT_ROOT, root, "diff_qp",
                                                  "main_seed*", "checkpoints", "best.pt"))):
            summary = os.path.join(os.path.dirname(os.path.dirname(path)),
                                   "evaluation_summary.json")
            if not os.path.exists(summary):
                continue
            with open(summary, encoding="utf-8") as handle:
                data = json.load(handle)
            candidate = (data["success"], -data["collision"])
            if key is None or candidate > key:
                best, key = path, candidate
        if best is not None:
            return best
    raise SystemExit("no evaluated diff_qp checkpoint found")


def board_frame(env, step, start, gain):
    """What the board would put on the wire, given the simulator's state.

    The board reports its own uncorrected odometry - the PC divides the gain
    back out - and it has no median filter, so it reports the frame the PC is
    about to filter, not the filtered one.
    """
    reported = env.odom_pose.copy()
    reported[:2] = start[:2] + (reported[:2] - start[:2]) * gain
    return Telemetry(sequence=step + 1, board_ms=1000 + 200 * step, pose=reported,
                     channels=env.sensor._history[-1].copy(), c_front_m=0.0,
                     command_v=0.0, command_omega=0.0,
                     feedback_v=float(env._applied[0]),
                     feedback_omega=float(env._applied[1]),
                     state="COMMAND_ACTIVE", last_command_sequence=step)


def compare(checkpoint, obstacles, label):
    state = load_checkpoint(checkpoint)
    cfg = Config.from_dict(state["config"])
    cfg.randomization.enabled = False        # a shared, noiseless plant

    env = AlvikEnv(cfg, seed=0)
    policy = PolicyRunner(cfg, "diff_qp", state["actor"])
    runtime = RealRobotRuntime(cfg, state["actor"])

    start = np.array(cfg.map.start_pose, dtype=np.float64)
    scenario = Scenario(start_pose=start.copy(),
                        goal=np.array(cfg.map.goal_position, dtype=np.float64),
                        obstacles=np.asarray(obstacles, dtype=np.float64).reshape(-1, 3))
    obs = env.reset(scenario)
    runtime.reset(start)

    worst = np.zeros(3)
    step = 0
    while True:
        mirrored = runtime.step(board_frame(env, step, start, cfg.robot.odom_distance_gain))
        reference = policy.act(obs, env.constraints, deterministic=True)
        worst = np.maximum(worst, [
            np.max(np.abs(mirrored.observation - obs)),
            np.max(np.abs(mirrored.desired - reference["action_real"])),
            np.max(np.abs(mirrored.odom_pose - env.odom_pose))])
        obs, _, done, info = env.step(reference["action_real"])
        step += 1
        if done:
            break

    outcome = ("success" if info["success"]
               else "COLLISION" if info["collision"] else "timeout")
    print("%-14s %3d steps  %-9s  obs %.2e  action %.2e  odom %.2e"
          % (label, step, outcome, worst[0], worst[1], worst[2]))
    return float(worst[:2].max())


def main(argv):
    checkpoint = argv[1] if len(argv) > 1 else default_checkpoint()
    print("checkpoint %s\n" % checkpoint)
    worst = max(compare(checkpoint, obstacles, label) for obstacles, label in SCENARIOS)
    print("\nworst disagreement anywhere: %.3e" % worst)
    if worst < TOLERANCE:
        print("PARITY OK - the runtime sees what the simulator sees")
        return 0
    print("PARITY BROKEN - do not arm the robot")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
