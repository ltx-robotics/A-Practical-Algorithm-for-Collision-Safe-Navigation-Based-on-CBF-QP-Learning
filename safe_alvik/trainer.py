

import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .config import ALGORITHM_SEMANTICS, Config, ACT_DIM, OBS_DIM
from .environment import AlvikEnv
from .evaluation import better_than, evaluate, holdout_scenarios, make_scenarios
from .io_utils import CsvLogger, save_checkpoint, save_json, set_global_seed
from .replay import ReplayBuffer
from .sac import SACAgent


class Trainer:
    def __init__(self, cfg: Config, mode: str, seed: int, run_dir: str,
                 device: str = "cpu", verbose: bool = True):
        self.cfg = cfg.validate()
        self.mode = mode
        self.seed = int(seed)
        self.run_dir = run_dir
        self.device = device
        self.verbose = verbose

        set_global_seed(seed)
        self.env = AlvikEnv(cfg, seed=seed)
        # A separate environment and seed offset so validation never consumes
        # the training RNG stream.
        self.eval_env = AlvikEnv(cfg, seed=seed + 100000)
        self.agent = SACAgent(cfg, mode, device=device, seed=seed)
        self.buffer = ReplayBuffer(cfg.sac.replay_capacity, OBS_DIM, ACT_DIM,
                                   cfg.memory.max_tracks)
        self.sample_rng = np.random.default_rng(seed + 777)

        self.holdout = holdout_scenarios(cfg)
        self.stage_index = 0
        self.stage_start_step = 0
        self.best_summary: Optional[Dict[str, float]] = None
        self.history: List[Dict[str, Any]] = []

        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
        self.train_log = CsvLogger(os.path.join(run_dir, "training.csv"))
        self.valid_log = CsvLogger(os.path.join(run_dir, "validation.csv"))
        self.stage_log = CsvLogger(os.path.join(run_dir, "validation_current_stage.csv"))

    # ------------------------------------------------------------------ stage
    @property
    def stage(self):
        return self.cfg.curriculum[self.stage_index]

    def stage_progress(self, step: int) -> float:
        span = max(self.stage.until_step - self.stage_start_step, 1)
        return float(np.clip((step - self.stage_start_step) / span, 0.0, 1.0))

    def maybe_promote(self, step: int, stage_summary: Dict[str, float]) -> bool:
        if self.stage_index >= len(self.cfg.curriculum) - 1:
            return False
        if step < self.stage.until_step:
            return False
        success = stage_summary.get("success", 0.0)
        collision = stage_summary.get("collision", 1.0)
        if success < self.stage.promote_min_success:
            return False
        if collision > self.stage.promote_max_collision:
            return False
        self.stage_index += 1
        self.stage_start_step = step
        return True

    # ------------------------------------------------------------------ train
    def train(self, total_steps: Optional[int] = None) -> Dict[str, Any]:
        cfg = self.cfg
        total = int(total_steps if total_steps is not None else cfg.train.total_steps)
        started = time.time()

        scenario = self.env.sample_scenario(self.stage, self.stage_progress(0))
        obs = self.env.reset(scenario)
        constraints = self.env.constraints
        episode = self._new_episode_record(0, scenario)
        episode_index = 0
        last_metrics: Dict[str, float] = {}

        for step in range(1, total + 1):
            random_action = step <= cfg.sac.start_steps
            out = self.agent.act(obs, constraints, deterministic=False,
                                 random_action=random_action)
            next_obs, reward, done, info = self.env.step(out["action_real"])
            next_constraints = self.env.constraints

            # A timeout is not a terminal state of the MDP: bootstrapping must
            # continue through it, otherwise the value function learns that the
            # world ends after 300 steps.
            bootstrap_done = bool(info["collision"] or info["controller_stopped"])
            self.buffer.add(obs, out["z_nominal"], out["z_safe"], reward, next_obs,
                            bootstrap_done, constraints, next_constraints)

            self._accumulate(episode, out, info, reward)
            obs, constraints = next_obs, next_constraints

            if step > cfg.sac.start_steps and len(self.buffer) >= cfg.sac.batch_size:
                for _ in range(cfg.sac.updates_per_step):
                    batch = self.buffer.sample(cfg.sac.batch_size,
                                               self.agent.device, self.sample_rng)
                    last_metrics = self.agent.update(batch)

            if done:
                episode_index += 1
                self._log_episode(step, episode_index, episode, last_metrics)
                scenario = self.env.sample_scenario(self.stage, self.stage_progress(step))
                obs = self.env.reset(scenario)
                constraints = self.env.constraints
                episode = self._new_episode_record(step, scenario)

            if step % cfg.train.eval_interval == 0 or step == total:
                self._validate(step, total, started)

        self.train_log.close()
        self.valid_log.close()
        self.stage_log.close()

        summary = {
            "mode": self.mode,
            "seed": self.seed,
            "algorithm_semantics": ALGORITHM_SEMANTICS,
            "total_steps": total,
            "episodes": episode_index,
            "wall_clock_s": time.time() - started,
            "steps_per_second": total / max(time.time() - started, 1e-9),
            "final_stage": self.stage.name,
            "best": self.best_summary,
            "qp_calls_interaction": self.agent.qp_calls_interaction,
            "qp_calls_update": self.agent.qp_calls_update,
        }
        save_json(os.path.join(self.run_dir, "run_summary.json"), summary)
        return summary

    # ------------------------------------------------------------- validation
    def _validate(self, step: int, total: int, started: float) -> None:
        cfg = self.cfg
        stage_scenarios = make_scenarios(
            cfg, self.stage,
            seeds=range(500000 + step, 500000 + step + cfg.train.stage_eval_episodes),
            progress=self.stage_progress(step))
        runner = self.agent.eval_runner()
        stage_summary, _ = evaluate(self.eval_env, runner, stage_scenarios)
        holdout_summary, _ = evaluate(self.eval_env, runner, self.holdout)

        row_common = {"step": step, "stage_index": self.stage_index,
                      "stage": self.stage.name,
                      "wall_clock_s": time.time() - started}
        self.stage_log.write(dict(row_common, **stage_summary))
        self.valid_log.write(dict(row_common, **holdout_summary))
        self.history.append(dict(row_common, **holdout_summary))

        payload = self.agent.state_dict()
        payload["step"] = step
        payload["holdout_summary"] = holdout_summary
        save_checkpoint(os.path.join(self.run_dir, "checkpoints", "latest.pt"), payload)
        if step % cfg.train.checkpoint_interval == 0 or step == total:
            save_checkpoint(os.path.join(self.run_dir, "checkpoints",
                                         "step_%07d.pt" % step), payload)
        if better_than(holdout_summary, self.best_summary):
            self.best_summary = dict(holdout_summary, step=step)
            save_checkpoint(os.path.join(self.run_dir, "checkpoints", "best.pt"), payload)

        evaluated_stage = self.stage.name
        promoted = self.maybe_promote(step, stage_summary)
        if self.verbose:
            print("[%s seed=%d] step %d/%d stage=%s success=%.3f collision=%.3f "
                  "dist=%.3f return=%.1f%s"
                  % (self.mode, self.seed, step, total, evaluated_stage,
                     holdout_summary.get("success", float("nan")),
                     holdout_summary.get("collision", float("nan")),
                     holdout_summary.get("final_distance_true", float("nan")),
                     holdout_summary.get("return", float("nan")),
                     "  -> promoted to %s" % self.stage.name if promoted else ""), flush=True)

    # ---------------------------------------------------------------- logging
    def _new_episode_record(self, step: int, scenario) -> Dict[str, Any]:
        return {
            "start_step": step,
            "return": 0.0,
            "steps": 0,
            "interventions": 0,
            "infeasible": 0,
            "violations": 0,
            "min_true_clearance_m": float("inf"),
            "min_channel_m": float("inf"),
            "delta_v": 0.0,
            "delta_omega": 0.0,
            "n_obstacles": scenario.n_obstacles,
            "direct_path_blocked": bool(scenario.direct_path_blocked),
            "straight_distance_m": float(np.linalg.norm(
                np.asarray(scenario.goal) - scenario.start_pose[:2])),
            "info": {},
        }

    def _accumulate(self, episode: Dict[str, Any], out: Dict[str, Any],
                    info: Dict[str, Any], reward: float) -> None:
        episode["return"] += reward
        episode["steps"] += 1
        dz = out["z_safe"] - out["z_nominal"]
        if abs(float(dz[0])) > 1e-4 or abs(float(dz[1])) > 1e-4:
            episode["interventions"] += 1
        episode["delta_v"] += abs(float(dz[0])) * self.cfg.robot.max_linear_mps
        episode["delta_omega"] += abs(float(dz[1])) * self.cfg.robot.max_angular_rps
        if not out["qp"].get("feasible", True):
            episode["infeasible"] += 1
        if info["true_clearance"] < 0.0:
            episode["violations"] += 1
        episode["min_true_clearance_m"] = min(episode["min_true_clearance_m"],
                                              info["true_clearance"])
        episode["min_channel_m"] = min(episode["min_channel_m"], info["min_channel_m"])
        episode["info"] = info

    def _log_episode(self, step: int, index: int, episode: Dict[str, Any],
                     metrics: Dict[str, float]) -> None:
        info = episode["info"]
        steps = max(episode["steps"], 1)
        straight = episode["straight_distance_m"]
        path = float(info.get("path_length_m", 0.0))
        row = {
            "step": step,
            "episode": index,
            "stage_index": self.stage_index,
            "stage": self.stage.name,
            "return": episode["return"],
            "steps": episode["steps"],
            "success": info.get("success", False),
            "collision": info.get("collision", False),
            "timeout": info.get("timeout", False),
            "final_distance_true": info.get("distance_true", float("nan")),
            "final_distance_odom": info.get("distance_odom", float("nan")),
            "path_length_m": path,
            "detour_ratio": path / straight if straight > 1e-9 else float("nan"),
            "min_true_clearance_m": episode["min_true_clearance_m"],
            "min_channel_m": episode["min_channel_m"],
            "cbf_violation_steps": episode["violations"],
            "qp_intervention_rate": episode["interventions"] / steps,
            "qp_infeasible_steps": episode["infeasible"],
            "mean_abs_delta_v": episode["delta_v"] / steps,
            "mean_abs_delta_omega": episode["delta_omega"] / steps,
            "n_obstacles": episode["n_obstacles"],
            "direct_path_blocked": episode["direct_path_blocked"],
            "critic_loss": metrics.get("critic_loss", float("nan")),
            "actor_loss": metrics.get("actor_loss", float("nan")),
            "alpha": metrics.get("alpha", float("nan")),
            "q_mean": metrics.get("q_mean", float("nan")),
        }
        self.train_log.write(row)
