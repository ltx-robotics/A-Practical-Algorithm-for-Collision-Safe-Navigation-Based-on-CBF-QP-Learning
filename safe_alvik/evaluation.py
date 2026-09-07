

from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .config import Config, StageConfig
from .environment import AlvikEnv, Scenario, ScenarioGenerator


def make_scenarios(cfg: Config, stage: StageConfig, seeds: Sequence[int],
                   blocking_probability: Optional[float] = None,
                   progress: float = 1.0) -> List[Scenario]:
    generator = ScenarioGenerator(cfg)
    scenarios = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        scenarios.append(generator.sample(rng, stage, progress, blocking_probability))
    return scenarios


def holdout_scenarios(cfg: Config, count: Optional[int] = None) -> List[Scenario]:
    """The fixed target distribution: final stage, every episode blocking."""
    train = cfg.train
    stage = StageConfig(name="holdout",
                        until_step=cfg.train.total_steps,
                        n_obstacles_min=train.holdout_n_obstacles_min,
                        n_obstacles_max=train.holdout_n_obstacles_max,
                        randomize_start_goal=False,
                        start_pose_jitter_m=cfg.curriculum[-1].start_pose_jitter_m,
                        start_heading_jitter_deg=cfg.curriculum[-1].start_heading_jitter_deg)
    episodes = int(count if count is not None else train.eval_episodes)
    seeds = range(train.holdout_seed_start, train.holdout_seed_start + episodes)
    return make_scenarios(cfg, stage, list(seeds),
                          blocking_probability=train.holdout_blocking_probability)


def run_episode(env: AlvikEnv, agent, scenario: Scenario,
                deterministic: bool = True) -> Dict[str, Any]:
    obs = env.reset(scenario)
    straight = float(np.linalg.norm(np.asarray(scenario.goal) - scenario.start_pose[:2]))

    total_reward = 0.0
    steps = 0
    violations = 0
    interventions = 0
    infeasible = 0
    min_clearance = float("inf")
    min_channel = float("inf")
    delta_v: List[float] = []
    delta_w: List[float] = []
    solve_ms: List[float] = []
    info: Dict[str, Any] = {}

    done = False
    while not done:
        result = agent.act(obs, env.constraints, deterministic=deterministic)
        obs, reward, done, info = env.step(result["action_real"])
        steps += 1
        total_reward += reward

        dz = result["z_safe"] - result["z_nominal"]
        if abs(float(dz[0])) > 1e-4 or abs(float(dz[1])) > 1e-4:
            interventions += 1
        delta_v.append(abs(float(dz[0])) * env.cfg.robot.max_linear_mps)
        delta_w.append(abs(float(dz[1])) * env.cfg.robot.max_angular_rps)
        solve_ms.append(float(result["qp"].get("solve_ms", 0.0)))
        if not result["qp"].get("feasible", True):
            infeasible += 1

        clearance = info["true_clearance"]
        if clearance < min_clearance:
            min_clearance = clearance
        if clearance < 0.0:
            violations += 1
        if info["min_channel_m"] < min_channel:
            min_channel = info["min_channel_m"]

    return {
        "return": total_reward,
        "steps": steps,
        "success": bool(info.get("success", False)),
        "collision": bool(info.get("collision", False)),
        "timeout": bool(info.get("timeout", False)),
        "controller_stopped": bool(info.get("controller_stopped", False)),
        "final_distance_true": float(info.get("distance_true", float("nan"))),
        "final_distance_odom": float(info.get("distance_odom", float("nan"))),
        "odom_error_m": float(info.get("odom_error_m", float("nan"))),
        "path_length_m": float(info.get("path_length_m", 0.0)),
        "detour_ratio": float(info.get("path_length_m", 0.0) / straight) if straight > 1e-9 else float("nan"),
        "straight_distance_m": straight,
        "min_true_clearance_m": float(min_clearance),
        "cbf_violation_steps": violations,
        # Per-step rate is confounded by early termination: a method that
        # collides ends the episode and stops accumulating violation steps, so
        # a *lower* rate can mean a *worse* policy. Always read it next to the
        # per-episode flag and the collision rate.
        "cbf_violation_rate": violations / max(steps, 1),
        "any_cbf_violation": bool(violations > 0),
        "min_channel_m": float(min_channel),
        "qp_intervention_steps": interventions,
        "qp_intervention_rate": interventions / max(steps, 1),
        "qp_infeasible_steps": infeasible,
        "qp_infeasible_rate": infeasible / max(steps, 1),
        "mean_abs_delta_v": float(np.mean(delta_v)) if delta_v else 0.0,
        "p95_abs_delta_v": float(np.percentile(delta_v, 95)) if delta_v else 0.0,
        "mean_abs_delta_omega": float(np.mean(delta_w)) if delta_w else 0.0,
        "p95_abs_delta_omega": float(np.percentile(delta_w, 95)) if delta_w else 0.0,
        "qp_solve_ms_mean": float(np.mean(solve_ms)) if solve_ms else 0.0,
        "qp_solve_ms_max": float(np.max(solve_ms)) if solve_ms else 0.0,
        "n_obstacles": scenario.n_obstacles,
        "direct_path_blocked": bool(scenario.direct_path_blocked),
        "straight_path_gap_m": float(scenario.straight_path_gap_m)
        if np.isfinite(scenario.straight_path_gap_m) else float("nan"),
    }


def evaluate(env: AlvikEnv, agent, scenarios: Sequence[Scenario],
             deterministic: bool = True):
    episodes = [run_episode(env, agent, scenario, deterministic) for scenario in scenarios]
    return summarize(episodes), episodes


def summarize(episodes: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not episodes:
        return {}
    keys = [
        "return", "steps", "success", "collision", "timeout", "controller_stopped",
        "final_distance_true", "final_distance_odom", "odom_error_m", "path_length_m",
        "detour_ratio", "min_true_clearance_m", "cbf_violation_rate",
        "any_cbf_violation", "cbf_violation_steps", "min_channel_m",
        "qp_intervention_rate", "qp_infeasible_rate", "mean_abs_delta_v",
        "mean_abs_delta_omega", "qp_solve_ms_mean", "qp_solve_ms_max",
    ]
    summary: Dict[str, float] = {"episodes": float(len(episodes))}
    for key in keys:
        values = np.array([float(episode[key]) for episode in episodes], dtype=np.float64)
        finite = values[np.isfinite(values)]
        summary[key] = float(np.mean(finite)) if finite.size else float("nan")

    blocked = [e for e in episodes if e["direct_path_blocked"]]
    summary["blocked_fraction"] = len(blocked) / len(episodes)
    summary["blocked_success"] = (float(np.mean([e["success"] for e in blocked]))
                                  if blocked else float("nan"))
    successes = [e for e in episodes if e["success"]]
    summary["success_mean_steps"] = (float(np.mean([e["steps"] for e in successes]))
                                     if successes else float("nan"))
    summary["success_mean_detour"] = (float(np.mean([e["detour_ratio"] for e in successes]))
                                      if successes else float("nan"))
    worst = np.array([e["min_true_clearance_m"] for e in episodes], dtype=np.float64)
    worst = worst[np.isfinite(worst)]
    summary["worst_true_clearance_m"] = float(np.min(worst)) if worst.size else float("nan")
    solve = np.array([e["qp_solve_ms_max"] for e in episodes], dtype=np.float64)
    summary["qp_solve_ms_p95"] = float(np.percentile(solve, 95)) if solve.size else 0.0
    return summary


def better_than(candidate: Dict[str, float], incumbent: Optional[Dict[str, float]]) -> bool:
    """Lexicographic model selection on the held-out target distribution.

    success up, collision down, final true distance down, return up. The order
    matters: judging on return or collision first lets a policy that never moves
    win, which is exactly what happened in the earlier project.
    """
    if incumbent is None:
        return True
    order = [("success", 1), ("collision", -1), ("final_distance_true", -1), ("return", 1)]
    for key, direction in order:
        a = candidate.get(key, float("nan"))
        b = incumbent.get(key, float("nan"))
        if not np.isfinite(a) or not np.isfinite(b):
            continue
        if abs(a - b) <= 1e-12:
            continue
        return (a > b) if direction > 0 else (a < b)
    return False
