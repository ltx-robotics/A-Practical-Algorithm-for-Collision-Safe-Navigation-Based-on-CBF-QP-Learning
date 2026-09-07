"""Drive the real Alvik with the trained diff_qp policy (PLAN.MD 10.4).

    # dry run: reads telemetry, runs the policy and the QP, sends NOTHING
    python real_robot/pc/run_diff_qp_real.py --board-ip 192.168.1.50 --layout A

    # for real, after the dry run has been checked
    python real_robot/pc/run_diff_qp_real.py --board-ip 192.168.1.50 --layout A --arm

Dry run is the default and it is not a formality: it is where you confirm the
19 numbers going into the actor are the ones it trained on. Check that the ToF
channels match a tape measure, that the goal-body components have the right
sign, and that the QP constraint count is zero on an empty floor.

Only diff_qp checkpoints are accepted (PLAN.MD 10.1).

Success is NOT decided here. The odometry is the thing being corrected, so it
cannot also be the judge; measure the final position on the printed grid or from
the overhead video (PLAN.MD 10.6). This script records where the robot thought
it was, and stops when that estimate reaches the goal radius.
"""

import argparse
import csv
import glob
import json
import math
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from safe_alvik.config import Config  # noqa: E402
from safe_alvik.io_utils import load_checkpoint, save_json  # noqa: E402

from real_robot.pc.link import BoardLink  # noqa: E402
from real_robot.pc.runtime import RealRobotRuntime  # noqa: E402

TRAINING_ROOTS = ("runs_g4_ttl8", "runs_g4")


def deployment_checkpoint() -> str:
    """Best diff_qp checkpoint, PLAN.MD 7.3 ordering: success up, collision down."""
    for root in TRAINING_ROOTS:
        best, key = None, None
        for path in sorted(glob.glob(os.path.join(PROJECT_ROOT, root, "diff_qp",
                                                  "main_seed*", "checkpoints", "best.pt"))):
            summary = os.path.join(os.path.dirname(os.path.dirname(path)),
                                   "evaluation_summary.json")
            if not os.path.exists(summary):
                continue
            data = json.load(open(summary, encoding="utf-8"))
            candidate = (data["success"], -data["collision"])
            if key is None or candidate > key:
                best, key = path, candidate
        if best is not None:
            return best
    raise SystemExit("no evaluated diff_qp checkpoint under %s" % " or ".join(TRAINING_ROOTS))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--board-ip", required=True, help="the Alvik's Wi-Fi address")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--layout", default="A", help="label recorded in the log")
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--arm", action="store_true",
                        help="actually send drive commands; without it nothing is sent")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--telemetry-timeout", type=float, default=1.2,
                        help="give up on the trial after this long without a frame; "
                        "the link stops driving after 0.4 s of silence regardless")
    parser.add_argument("--output", default=os.path.join(PROJECT_ROOT, "real_robot", "logs"))
    return parser


def preflight(runtime: RealRobotRuntime, checkpoint: str, state, armed: bool) -> None:
    facts = runtime.describe()
    print("=" * 72)
    print("checkpoint      %s" % checkpoint)
    print("  mode %s, step %s, semantics %s"
          % (state.get("mode"), state.get("step"), state.get("algorithm_semantics")))
    print("control         %.1f Hz, v <= %.3f m/s, omega <= %.1f deg/s"
          % (facts["control_hz"], facts["max_linear_mps"], facts["max_angular_deg_s"]))
    print("compensation    v / %.3f,  omega / %.3f (CCW) or %.3f (CW)"
          % (facts["gain_linear"], facts["gain_angular_ccw"], facts["gain_angular_cw"]))
    print("odometry        reported distance / %.3f   <-- omit this and the robot"
          % facts["odom_distance_gain"])
    print("                stops about 20 mm short every run")
    print("memory          ttl %.1f s, confirm %d frames"
          % (facts["memory_ttl_s"], facts["memory_confirm_frames"]))
    print("CBF             alpha %.2f, safety radius %.3f m"
          % (facts["cbf_alpha"], facts["cbf_safety_radius_m"]))
    print("stop / success  %.3f m odometry stop, %.3f m independently measured"
          % (facts["controller_goal_radius_m"], facts["success_radius_m"]))
    print("ARMED" if armed else "DRY RUN - no command will be sent")
    print("=" * 72, flush=True)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint = args.checkpoint or deployment_checkpoint()
    state = load_checkpoint(checkpoint)
    if state.get("mode") != "diff_qp":
        raise SystemExit("refusing mode %r: PLAN.MD 10.1 restricts the real robot "
                         "to diff_qp" % state.get("mode"))
    cfg = Config.from_dict(state["config"])
    runtime = RealRobotRuntime(cfg, state["actor"])
    preflight(runtime, checkpoint, state, args.arm)

    max_steps = args.max_steps or cfg.robot.max_episode_steps
    os.makedirs(args.output, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = "layout%s_trial%02d_%s%s" % (args.layout, args.trial, stamp,
                                       "" if args.arm else "_dryrun")
    csv_path = os.path.join(args.output, tag + ".csv")

    rows = []
    stop_reason = "unknown"
    started = time.time()
    try:
        with BoardLink(args.board_ip, armed=args.arm) as link:
            print("waiting for telemetry from %s ..." % args.board_ip, flush=True)
            first = link.receive_latest(5.0)
            if first is None:
                  print("no telemetry; check the board is running main.py, the Wi-Fi "
                        "credentials and that PC_IP in wifi_secrets.py is this machine",
                        flush=True)
                  return 2

            # Telemetry first, so the outgoing numbering can continue above what
            # the board already accepted. Sending anything before this - RESET
            # included - would be silently dropped for the rest of the session.
            started_at = link.sync_sequence(first.last_command_sequence)
            if first.last_command_sequence > 0:
                  print("board already accepted up to sequence %d; continuing from %d"
                        % (first.last_command_sequence, started_at), flush=True)
            if args.arm:
                  link.send_reset(cfg.map.start_pose)
                  time.sleep(0.3)
            runtime.reset(cfg.map.start_pose)

            # The policy decides at cfg.robot.control_dt_s because that is the
            # interval it trained at - both the applied-speed observation and the
            # three-frame median window are defined by it. Letting the 10 Hz
            # telemetry pace the loop instead would silently run the policy at
            # twice its training rate.
            period = cfg.robot.control_dt_s
            next_tick = time.time()
            skipped_frames = 0

            for step in range(max_steps):
                  now = time.time()
                  if next_tick > now:
                      time.sleep(next_tick - now)
                  next_tick += period

                  frames = link.receive_frames(args.telemetry_timeout)
                  if not frames:
                      link.brake()
                      stop_reason = "telemetry_timeout"
                      break
                  # Act on the newest, but let the mirror see every board sample.
                  for stale in frames[:-1]:
                      runtime.note_board_sample(stale)
                  skipped_frames += len(frames) - 1
                  telemetry = frames[-1]

                  result = runtime.step(telemetry)
                  link.send_command(result.command[0], result.command[1])

                  rows.append({
                      "step": step,
                      "board_ms": telemetry.board_ms,
                      "board_seq": telemetry.sequence,
                      "board_state": result.board_state,
                      "predicted_board_state": result.predicted_board_state,
                      "odom_x": result.odom_pose[0], "odom_y": result.odom_pose[1],
                      "odom_theta_deg": math.degrees(result.odom_pose[2]),
                      "board_x": telemetry.pose[0], "board_y": telemetry.pose[1],
                      "board_theta_deg": math.degrees(telemetry.pose[2]),
                      "L_mm": telemetry.channels[0] * 1000, "CL_mm": telemetry.channels[1] * 1000,
                      "C_mm": telemetry.channels[2] * 1000, "CR_mm": telemetry.channels[3] * 1000,
                      "R_mm": telemetry.channels[4] * 1000,
                      "L_filt_mm": result.channels[0] * 1000, "CL_filt_mm": result.channels[1] * 1000,
                      "C_filt_mm": result.channels[2] * 1000, "CR_filt_mm": result.channels[3] * 1000,
                      "R_filt_mm": result.channels[4] * 1000,
                      "z_nom_v": result.z_nominal[0], "z_nom_w": result.z_nominal[1],
                      "z_safe_v": result.z_safe[0], "z_safe_w": result.z_safe[1],
                      "desired_v_mps": result.desired[0], "desired_w_degs": math.degrees(result.desired[1]),
                      "command_v_cms": result.command[0] * 100,
                      "command_w_degs": math.degrees(result.command[1]),
                      "applied_v_mps": runtime.applied[0],
                      "applied_w_degs": math.degrees(runtime.applied[1]),
                      "n_constraints": result.n_constraints,
                      "qp_feasible": int(result.feasible),
                      "qp_solve_ms": result.solve_ms,
                      "goal_distance_odom": result.goal_distance,
                      "memory_points": len(result.memory_points),
                  })

                  if step % 5 == 0:
                      print("  %3d  odom (%+.3f,%+.3f) d=%.3f  ToF %3.0f/%3.0f/%3.0f  "
                            "cons %d  v %.3f->%.3f  %s"
                            % (step, result.odom_pose[0], result.odom_pose[1],
                               result.goal_distance, result.channels[1] * 1000,
                               result.channels[2] * 1000, result.channels[3] * 1000,
                               result.n_constraints,
                               result.z_nominal[0] * cfg.robot.max_linear_mps,
                               result.z_safe[0] * cfg.robot.max_linear_mps,
                               result.board_state), flush=True)

                  if result.reached:
                      link.brake()
                      stop_reason = "goal_reached_by_odometry"
                      break
            else:
                  link.brake()
                  stop_reason = "max_steps"

    except KeyboardInterrupt:
        # The interesting rows are usually the ones just before the stop.
        stop_reason = "interrupted"
        print("interrupted - braking and keeping the log", flush=True)
    elapsed = time.time() - started
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "layout": args.layout, "trial": args.trial, "armed": bool(args.arm),
        "checkpoint": os.path.abspath(checkpoint), "checkpoint_step": state.get("step"),
        "stop_reason": stop_reason, "steps": len(rows), "wall_clock_s": elapsed,
        "final_goal_distance_odom": rows[-1]["goal_distance_odom"] if rows else None,
        "board_forward_blocked_activations": runtime.board.blocked_activations,
        "board_critical_activations": runtime.board.critical_activations,
        "qp_infeasible_steps": sum(1 for r in rows if not r["qp_feasible"]),
        "qp_solve_ms_max": max((r["qp_solve_ms"] for r in rows), default=0.0),
        "control_hz_measured": (len(rows) / elapsed) if elapsed > 0 else None,
        "telemetry_frames_received": link.received_frames,
        "telemetry_frames_undecodable": link.dropped_frames,
        "telemetry_frames_superseded": skipped_frames,
        "heartbeats_zeroed_by_stale_telemetry": link.stale_heartbeats,
        "configuration": runtime.describe(),
        "note": ("Success must be measured independently on the printed grid or "
                 "from the overhead video; the odometry here is the thing being "
                 "corrected and cannot judge itself."),
    }
    save_json(os.path.join(args.output, tag + ".json"), summary)

    print("-" * 72)
    print("stop: %s after %d steps (%.1f s)" % (stop_reason, len(rows), elapsed))
    print("board safety: forward-blocked %d, critical %d   <-- each one is a step "
          "the CBF should have handled earlier"
          % (runtime.board.blocked_activations, runtime.board.critical_activations))
    print("QP: %d infeasible steps, worst solve %.2f ms"
          % (summary["qp_infeasible_steps"], summary["qp_solve_ms_max"]))
    print("rate: %.2f Hz measured against %.2f Hz trained"
          % (summary["control_hz_measured"] or 0.0, 1.0 / cfg.robot.control_dt_s))
    if rows:
        print("odometry says the goal is %.3f m away; MEASURE THE REAL POSITION"
              % rows[-1]["goal_distance_odom"])
    print("log: %s" % csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
