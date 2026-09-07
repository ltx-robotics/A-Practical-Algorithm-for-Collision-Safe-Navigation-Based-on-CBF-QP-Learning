"""Export a trained actor plus everything a deployment needs to reproduce its
observation and its safety layer.

Only `diff_qp` checkpoints are exported by default: PLAN.MD 10.1 restricts the
Webots and real-robot experiments to that method.

    python export_policy.py --checkpoint runs/diff_qp/RUN/checkpoints/best.pt
"""

import argparse
import os
import sys

import torch

from safe_alvik.config import (ALGORITHM_SEMANTICS, Config, GOAL_OBSERVATION_MODE,
                               CHANNEL_NAMES, OBS_NAMES, ZONE_NAMES)
from safe_alvik.io_utils import load_checkpoint, save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--allow-non-diff-qp", action="store_true",
                        help="export a vanilla/external_qp actor anyway (not for deployment)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    state = load_checkpoint(args.checkpoint)
    mode = state.get("mode")
    if mode != "diff_qp" and not args.allow_non_diff_qp:
        print("refusing to export mode %r: only diff_qp is deployed to Webots and the "
              "real robot (use --allow-non-diff-qp for a diagnostic export)" % mode)
        return 2

    cfg = Config.from_dict(state["config"])
    payload = {
        "actor": state["actor"],
        "mode": mode,
        "algorithm_semantics": ALGORITHM_SEMANTICS,
        "goal_observation_mode": GOAL_OBSERVATION_MODE,
        "observation_names": list(OBS_NAMES),
        "channel_names": list(CHANNEL_NAMES),
        "zone_names": list(ZONE_NAMES),
        "hidden_sizes": list(cfg.sac.hidden_sizes),
        "action": {
            "max_linear_mps": cfg.robot.max_linear_mps,
            "max_angular_rps": cfg.robot.max_angular_rps,
            "allow_reverse": cfg.robot.allow_reverse,
            "control_dt_s": cfg.robot.control_dt_s,
        },
        "compensation": {
            "gain_linear": cfg.robot.gain_linear,
            "gain_angular_ccw": cfg.robot.gain_angular_ccw,
            "gain_angular_cw": cfg.robot.gain_angular_cw,
            # The runtime MUST divide reported odometry distance by this before
            # computing the goal features, or it will stop ~20 mm short.
            "odom_distance_gain": cfg.robot.odom_distance_gain,
        },
        "cbf": {
            "alpha": cfg.cbf.alpha,
            "robust_velocity_mps": cfg.cbf.robust_velocity_mps,
            "look_ahead_m": cfg.robot.look_ahead_m,
            "safety_radius_m": cfg.robot.safety_radius_m,
            "weight_linear": cfg.cbf.weight_linear,
            "weight_angular": cfg.cbf.weight_angular,
            "max_rows": cfg.memory.max_tracks,
        },
        "memory": {
            "cluster_radius_m": cfg.memory.cluster_radius_m,
            "ttl_s": cfg.memory.ttl_s,
            "max_tracks": cfg.memory.max_tracks,
            "endpoint_accept_range_m": cfg.tof.endpoint_accept_range_m,
        },
        "tof": {
            "zone_centers_deg": list(cfg.tof.zone_centers_deg),
            "max_range_m": cfg.tof.max_range_m,
            "median_frames": cfg.tof.median_frames,
            "sensor_offset_x_m": cfg.robot.tof_offset_x_m,
        },
        "task": {
            "goal_position": list(cfg.map.goal_position),
            "controller_goal_radius_m": cfg.map.controller_goal_radius_m,
            "success_radius_m": cfg.map.success_radius_m,
            "goal_normalizer_m": 2.0 * cfg.map.half_extent_m,
        },
        "source_checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_step": state.get("step"),
        "config": state["config"],
    }

    output = args.output or os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)),
                                         "policy_%s.pt" % mode)
    torch.save(payload, output)
    save_json(os.path.splitext(output)[0] + "_metadata.json",
              {k: v for k, v in payload.items() if k not in ("actor", "config")})
    print("exported %s -> %s" % (mode, output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
