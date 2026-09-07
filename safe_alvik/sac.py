

import copy
import time
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .cbf_qp import CBFQPLayer
from .config import (ACT_DIM, ALGORITHM_SEMANTICS, Config, GOAL_OBSERVATION_MODE,
                     MODES, OBS_DIM)
from .geometry import tanh_to_normalized
from .networks import GaussianActor, TwinCritic


def single_step_action(actor, qp, obs, constraints, max_linear: float,
                       max_angular: float, rng, device, deterministic: bool,
                       random_action: bool) -> Dict[str, Any]:
    """One nominal action, optionally corrected by the safety layer.

    Shared by `SACAgent.act` and `PolicyRunner.act` so training rollouts and
    validation rollouts can never drift apart.
    """
    A, b, mask = constraints
    if random_action:
        z_nom = np.array([rng.uniform(0.0, 1.0), rng.uniform(-1.0, 1.0)])
    else:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action, _ = actor.sample(obs_t, deterministic=deterministic)
            z_nom = tanh_to_normalized(action)[0].cpu().numpy().astype(np.float64)

    if qp is not None:
        started = time.perf_counter()
        z_safe, qp_info = qp.solve_numpy(z_nom, A, b, mask)
        qp_info["solve_ms"] = (time.perf_counter() - started) * 1000.0
    else:
        z_safe = z_nom.copy()
        qp_info = {"feasible": True, "violation": 0.0, "solve_ms": 0.0,
                   "n_constraints": int(np.sum(mask)), "intervention": 0.0}

    return {
        "z_nominal": z_nom,
        "z_safe": z_safe,
        "action_real": np.array([z_safe[0] * max_linear, z_safe[1] * max_angular]),
        "qp": qp_info,
    }


class PolicyRunner:
    """Frozen CPU policy with the same `act` interface as `SACAgent`.

    Used for validation and deployment-style rollouts. It holds no critics and
    no optimizers, so building one per validation is cheap.
    """

    def __init__(self, cfg: Config, mode: str, actor_state: Dict[str, Any],
                 seed: int = 0):
        if mode not in MODES:
            raise ValueError("unknown mode %r" % (mode,))
        self.cfg = cfg
        self.mode = mode
        self.device = torch.device("cpu")
        self.rng = np.random.default_rng(seed)
        self.actor = GaussianActor(OBS_DIM, ACT_DIM, list(cfg.sac.hidden_sizes),
                                   cfg.sac.log_std_min, cfg.sac.log_std_max)
        self.actor.load_state_dict({k: v.detach().cpu() for k, v in actor_state.items()})
        self.actor.eval()
        self.uses_safety_layer = mode in ("external_qp", "diff_qp")
        self.qp = (CBFQPLayer(cfg.robot, cfg.cbf, max_rows=cfg.memory.max_tracks)
                   if self.uses_safety_layer else None)
        self.max_linear = cfg.robot.max_linear_mps
        self.max_angular = cfg.robot.max_angular_rps
        self.qp_calls_interaction = 0

    def act(self, obs: np.ndarray, constraints, deterministic: bool = True,
            random_action: bool = False) -> Dict[str, Any]:
        if self.uses_safety_layer:
            self.qp_calls_interaction += 1
        return single_step_action(self.actor, self.qp, obs, constraints,
                                  self.max_linear, self.max_angular, self.rng,
                                  self.device, deterministic, random_action)


class SACAgent:
    def __init__(self, cfg: Config, mode: str, device: str = "cpu", seed: int = 0):
        if mode not in MODES:
            raise ValueError("unknown mode %r, expected one of %s" % (mode, MODES))
        self.cfg = cfg
        self.mode = mode
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        hidden = list(cfg.sac.hidden_sizes)
        self.actor = GaussianActor(OBS_DIM, ACT_DIM, hidden,
                                   cfg.sac.log_std_min, cfg.sac.log_std_max).to(self.device)
        self.critic = TwinCritic(OBS_DIM, ACT_DIM, hidden).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for parameter in self.critic_target.parameters():
            parameter.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.sac.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.sac.lr_critic)
        self.log_alpha = torch.tensor(float(np.log(cfg.sac.alpha_init)),
                                      device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.sac.lr_alpha)

        # Two instances of the same stateless layer: the batched one lives on
        # the training device, the rollout one always on CPU because a
        # single-sample solve is dominated by kernel-launch overhead on GPU.
        self.qp = CBFQPLayer(cfg.robot, cfg.cbf, max_rows=cfg.memory.max_tracks).to(self.device)
        self.qp_rollout = (self.qp if self.device.type == "cpu"
                           else CBFQPLayer(cfg.robot, cfg.cbf,
                                           max_rows=cfg.memory.max_tracks))
        self.max_linear = cfg.robot.max_linear_mps
        self.max_angular = cfg.robot.max_angular_rps

        # Instrumentation used by the semantics tests and by the run logs.
        self.qp_calls_interaction = 0
        self.qp_calls_update = 0
        self.updates = 0

    # ------------------------------------------------------------- properties
    @property
    def uses_safety_layer(self) -> bool:
        return self.mode in ("external_qp", "diff_qp")

    @property
    def differentiates_through_qp(self) -> bool:
        return self.mode == "diff_qp"

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp().detach().clamp(min=self.cfg.sac.alpha_min)

    # ------------------------------------------------------------ interaction
    def act(self, obs: np.ndarray, constraints, deterministic: bool = False,
            random_action: bool = False) -> Dict[str, Any]:
        """One interaction step. Returns nominal, safe and real-unit actions."""
        result = single_step_action(
            self.actor, self.qp_rollout if self.uses_safety_layer else None,
            obs, constraints, self.max_linear, self.max_angular, self.rng,
            self.device, deterministic, random_action)
        if self.uses_safety_layer:
            self.qp_calls_interaction += 1
        return result

    def eval_runner(self) -> "PolicyRunner":
        """A frozen CPU copy of the policy for validation rollouts.

        Single-sample inference is launch-bound: on this machine one actor
        forward costs 0.157 ms on CPU against 0.554 ms on CUDA, and one QP solve
        0.64 ms against 2.30 ms. Validation runs tens of thousands of such steps
        with the weights held fixed, so copying the actor to CPU once per
        validation is worth roughly a 3x speed-up of the dominant cost.
        """
        return PolicyRunner(self.cfg, self.mode, self.actor.state_dict())

    # ----------------------------------------------------------------- update
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        cfg = self.cfg.sac
        obs = batch["obs"]
        next_obs = batch["next_obs"]
        reward = batch["reward"]
        done = batch["done"]
        alpha = self.alpha

        # ---- critic -------------------------------------------------------
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_obs)
            z_next = tanh_to_normalized(next_action)
            if self.differentiates_through_qp:
                z_next, _ = self.qp(z_next, batch["next_cons_a"], batch["next_cons_b"],
                                    batch["next_cons_mask"])
                self.qp_calls_update += 1
            target_q = self.critic_target.q_min(next_obs, z_next) - alpha * next_log_prob
            backup = reward + self.cfg.sac.gamma * (1.0 - done) * target_q

        current_action = (batch["action_nominal"] if self.mode == "vanilla"
                          else batch["action_executed"])
        q1, q2 = self.critic(obs, current_action)
        critic_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---- actor --------------------------------------------------------
        action, log_prob = self.actor.sample(obs)
        z_actor = tanh_to_normalized(action)
        intervention = 0.0
        infeasible = 0.0
        if self.differentiates_through_qp:
            z_before = z_actor
            z_actor, qp_info = self.qp(z_actor, batch["cons_a"], batch["cons_b"],
                                       batch["cons_mask"])
            self.qp_calls_update += 1
            intervention = float(torch.linalg.norm(z_actor - z_before, dim=-1).mean())
            infeasible = float((~qp_info["feasible"]).float().mean())

        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)
        actor_loss = (alpha * log_prob - self.critic.q_min(obs, z_actor)).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        for parameter in self.critic.parameters():
            parameter.requires_grad_(True)

        # ---- entropy coefficient ------------------------------------------
        alpha_loss = -(self.log_alpha * (log_prob.detach() + cfg.target_entropy)).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            for parameter, target in zip(self.critic.parameters(),
                                         self.critic_target.parameters()):
                target.data.mul_(1.0 - cfg.tau)
                target.data.add_(cfg.tau * parameter.data)

        self.updates += 1
        return {
            "critic_loss": float(critic_loss),
            "actor_loss": float(actor_loss),
            "alpha_loss": float(alpha_loss),
            "alpha": float(self.alpha),
            "q_mean": float(q1.mean()),
            "log_prob": float(log_prob.mean()),
            "qp_intervention": intervention,
            "qp_infeasible": infeasible,
        }

    # ------------------------------------------------------------ checkpoints
    def state_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "algorithm_semantics": ALGORITHM_SEMANTICS,
            "goal_observation_mode": GOAL_OBSERVATION_MODE,
            "obs_dim": OBS_DIM,
            "act_dim": ACT_DIM,
            "controller_goal_radius": self.cfg.map.controller_goal_radius_m,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "updates": self.updates,
            "config": self.cfg.to_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any], strict: bool = True) -> None:
        if strict:
            if state.get("obs_dim") != OBS_DIM:
                raise ValueError(
                    "checkpoint observation dimension %s does not match the current "
                    "%d-dim layout; 9/10-dim policies cannot be loaded"
                    % (state.get("obs_dim"), OBS_DIM))
            if state.get("algorithm_semantics") != ALGORITHM_SEMANTICS:
                raise ValueError(
                    "checkpoint was produced with algorithm semantics %r, current "
                    "code is %r; results from the two must not be mixed"
                    % (state.get("algorithm_semantics"), ALGORITHM_SEMANTICS))
            if state.get("goal_observation_mode") != GOAL_OBSERVATION_MODE:
                raise ValueError("checkpoint uses a different goal observation mode")
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"].to(self.device))
        if "actor_optimizer" in state:
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
            self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        self.updates = int(state.get("updates", 0))
