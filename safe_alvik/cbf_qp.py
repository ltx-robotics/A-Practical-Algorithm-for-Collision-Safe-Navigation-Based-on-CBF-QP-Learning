

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .config import CBFConfig, RobotConfig
from .geometry import look_ahead_point

INACTIVE_B = -1.0e6


def build_constraint_rows(odom_pose: Sequence[float],
                          points: Optional[np.ndarray],
                          robot: RobotConfig,
                          cbf: CBFConfig,
                          max_rows: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Constraint rows for the current step, in real action units (v, omega).

    Returns (A, b, mask) with shapes (max_rows, 2), (max_rows,), (max_rows,).
    Rows whose mask is 0 are padding and are always satisfied.
    """
    A = np.zeros((max_rows, 2), dtype=np.float64)
    b = np.full(max_rows, INACTIVE_B, dtype=np.float64)
    mask = np.zeros(max_rows, dtype=np.float64)
    if points is None or len(points) == 0 or not cbf.enabled:
        return A, b, mask

    look_ahead = float(robot.look_ahead_m)
    safety_radius = float(robot.safety_radius_m)
    theta = float(odom_pose[2])
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    control_point = look_ahead_point(odom_pose, look_ahead)

    array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    delta = control_point[None, :] - array
    distance = np.linalg.norm(delta, axis=1)
    order = np.argsort(distance)          # nearest surfaces first, deterministic

    row = 0
    for index in order:
        if row >= max_rows:
            break
        dist = float(distance[index])
        if dist < 1e-9:
            continue
        normal = delta[index] / dist
        h = dist - safety_radius
        # Rotate the world-frame normal into the body frame.
        mx = cos_t * normal[0] + sin_t * normal[1]
        my = -sin_t * normal[0] + cos_t * normal[1]
        A[row, 0] = mx
        A[row, 1] = look_ahead * my
        b[row] = -cbf.alpha * h + cbf.robust_velocity_mps
        mask[row] = 1.0
        row += 1
    return A, b, mask


def barrier_values(odom_pose: Sequence[float], points: Optional[np.ndarray],
                   robot: RobotConfig) -> np.ndarray:
    """h_i for the current memory points, as the QP sees them."""
    if points is None or len(points) == 0:
        return np.zeros((0,), dtype=np.float64)
    control_point = look_ahead_point(odom_pose, robot.look_ahead_m)
    array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return np.linalg.norm(control_point[None, :] - array, axis=1) - robot.safety_radius_m


class CBFQPLayer(torch.nn.Module):
    """Batched, differentiable solver for the two-variable safety QP."""

    def __init__(self, robot: RobotConfig, cbf: CBFConfig, max_rows: int = 4):
        super().__init__()
        self.max_linear = float(robot.max_linear_mps)
        self.max_angular = float(robot.max_angular_rps)
        self.weight_linear = float(cbf.weight_linear)
        self.weight_angular = float(cbf.weight_angular)
        self.tol = float(cbf.feasibility_tol)
        self.max_rows = int(max_rows)
        self.n_rows = self.max_rows + 4          # obstacle rows + box rows

        pairs_i: List[int] = []
        pairs_j: List[int] = []
        for i in range(self.n_rows):
            for j in range(i + 1, self.n_rows):
                pairs_i.append(i)
                pairs_j.append(j)
        self.register_buffer("_pair_i", torch.tensor(pairs_i, dtype=torch.long), persistent=False)
        self.register_buffer("_pair_j", torch.tensor(pairs_j, dtype=torch.long), persistent=False)
        self.register_buffer("_action_scale",
                             torch.tensor([self.max_linear, self.max_angular], dtype=torch.float32),
                             persistent=False)
        self.register_buffer("_weights",
                             torch.tensor([self.weight_linear, self.weight_angular], dtype=torch.float32),
                             persistent=False)
        box_a = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype=torch.float32)
        box_b = torch.tensor([0.0, -1.0, -1.0, -1.0], dtype=torch.float32)
        self.register_buffer("_box_a", box_a, persistent=False)
        self.register_buffer("_box_b", box_b, persistent=False)

    # ------------------------------------------------------------------ solve
    def forward(self, z_nom: torch.Tensor, A: torch.Tensor, b: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """z_nom (B, 2) normalized; A (B, M, 2) and b (B, M) in real units."""
        batch = z_nom.shape[0]
        dtype = z_nom.dtype
        device = z_nom.device
        eps = 1e-9

        scale = self._action_scale.to(device=device, dtype=dtype)
        weights = self._weights.to(device=device, dtype=dtype)
        A = A.to(dtype)
        b = b.to(dtype)
        active = mask.to(dtype) > 0.5

        # Real-unit rows act on (v, omega); rewrite them to act on z, then
        # normalize each row so the feasibility tolerance means the same thing
        # for every constraint.
        A_z = A * scale.view(1, 1, 2)
        norms = torch.linalg.norm(A_z, dim=-1)
        usable = active & (norms > eps)
        norms_safe = torch.where(usable, norms, torch.ones_like(norms))
        A_z = A_z / norms_safe.unsqueeze(-1)
        b_z = b / norms_safe
        A_z = torch.where(usable.unsqueeze(-1), A_z, torch.zeros_like(A_z))
        b_z = torch.where(usable, b_z, torch.full_like(b_z, INACTIVE_B))

        box_a = self._box_a.to(device=device, dtype=dtype).expand(batch, 4, 2)
        box_b = self._box_b.to(device=device, dtype=dtype).expand(batch, 4)
        A_all = torch.cat([A_z, box_a], dim=1)                    # (B, N, 2)
        b_all = torch.cat([b_z, box_b], dim=1)                    # (B, N)
        active_all = torch.cat([usable, torch.ones(batch, 4, dtype=torch.bool, device=device)], dim=1)

        # --- candidate 1: unconstrained minimum -------------------------------
        cand_free = z_nom.unsqueeze(1)                            # (B, 1, 2)
        valid_free = torch.ones(batch, 1, dtype=torch.bool, device=device)

        # --- candidates 2..N+1: exactly one constraint active -----------------
        inv_w = 1.0 / weights
        denom = ((A_all * A_all) * inv_w.view(1, 1, 2)).sum(-1)   # (B, N)
        resid = b_all - (A_all * z_nom.unsqueeze(1)).sum(-1)      # (B, N)
        valid_single = active_all & (denom > eps)
        denom_safe = torch.where(valid_single, denom, torch.ones_like(denom))
        step = (A_all * inv_w.view(1, 1, 2)) * (resid / denom_safe).unsqueeze(-1)
        cand_single = z_nom.unsqueeze(1) + step                   # (B, N, 2)

        # --- candidates: two constraints active -------------------------------
        idx_i = self._pair_i.to(device)
        idx_j = self._pair_j.to(device)
        Ai = A_all.index_select(1, idx_i)
        Aj = A_all.index_select(1, idx_j)
        bi = b_all.index_select(1, idx_i)
        bj = b_all.index_select(1, idx_j)
        det = Ai[..., 0] * Aj[..., 1] - Ai[..., 1] * Aj[..., 0]
        valid_pair = (active_all.index_select(1, idx_i)
                      & active_all.index_select(1, idx_j)
                      & (det.abs() > 1e-7))
        det_safe = torch.where(valid_pair, det, torch.ones_like(det))
        pair_x = (bi * Aj[..., 1] - Ai[..., 1] * bj) / det_safe
        pair_y = (Ai[..., 0] * bj - bi * Aj[..., 0]) / det_safe
        cand_pair = torch.stack([pair_x, pair_y], dim=-1)         # (B, P, 2)

        candidates = torch.cat([cand_free, cand_single, cand_pair], dim=1)
        valid = torch.cat([valid_free, valid_single, valid_pair], dim=1)

        # --- pick the feasible candidate with the smallest objective ----------
        residuals = torch.einsum("bnk,bck->bcn", A_all, candidates) - b_all.unsqueeze(1)
        feasible = (residuals >= -self.tol).all(dim=-1) & valid
        objective = (weights.view(1, 1, 2) * (candidates - z_nom.unsqueeze(1)) ** 2).sum(-1)
        masked = torch.where(feasible, objective, torch.full_like(objective, float("inf")))
        best_value, best_index = masked.min(dim=1)
        any_feasible = torch.isfinite(best_value)

        gather_index = best_index.view(batch, 1, 1).expand(batch, 1, 2)
        z_best = candidates.gather(1, gather_index).squeeze(1)

        # Infeasible hard problem: stop translating, keep the requested turn.
        z_fallback = torch.cat([torch.zeros(batch, 1, dtype=dtype, device=device),
                                z_nom[:, 1:2].clamp(-1.0, 1.0)], dim=-1)
        z_safe = torch.where(any_feasible.unsqueeze(-1), z_best, z_fallback)
        z_safe = torch.cat([z_safe[:, 0:1].clamp(0.0, 1.0),
                            z_safe[:, 1:2].clamp(-1.0, 1.0)], dim=-1)

        with torch.no_grad():
            final_residual = (torch.einsum("bnk,bk->bn", A_all, z_safe) - b_all)
            violation = torch.relu(-final_residual)
            violation = torch.where(active_all, violation, torch.zeros_like(violation))
            info = {
                "feasible": any_feasible,
                "violation": violation.max(dim=-1).values,
                "n_constraints": usable.sum(dim=-1),
                "intervention": torch.linalg.norm(z_safe - z_nom, dim=-1),
                "active_candidate": best_index,
            }
        return z_safe, info

    # ------------------------------------------------------------ convenience
    @torch.no_grad()
    def solve_numpy(self, z_nom: np.ndarray, A: np.ndarray, b: np.ndarray,
                    mask: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Single-sample solve used during environment rollouts."""
        device = self._action_scale.device
        z_t = torch.as_tensor(np.asarray(z_nom, dtype=np.float32).reshape(1, 2), device=device)
        A_t = torch.as_tensor(np.asarray(A, dtype=np.float32).reshape(1, -1, 2), device=device)
        b_t = torch.as_tensor(np.asarray(b, dtype=np.float32).reshape(1, -1), device=device)
        m_t = torch.as_tensor(np.asarray(mask, dtype=np.float32).reshape(1, -1), device=device)
        z_safe, info = self.forward(z_t, A_t, b_t, m_t)
        summary = {
            "feasible": bool(info["feasible"][0].item()),
            "violation": float(info["violation"][0].item()),
            "n_constraints": int(info["n_constraints"][0].item()),
            "intervention": float(info["intervention"][0].item()),
        }
        return z_safe[0].cpu().numpy().astype(np.float64), summary
