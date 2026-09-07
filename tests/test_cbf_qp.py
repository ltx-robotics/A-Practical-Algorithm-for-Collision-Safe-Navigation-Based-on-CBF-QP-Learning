import math
import unittest

import numpy as np
import torch

from safe_alvik.cbf_qp import CBFQPLayer, barrier_values, build_constraint_rows
from safe_alvik.config import Config


def make_layer(cfg=None):
    cfg = cfg or Config()
    return CBFQPLayer(cfg.robot, cfg.cbf, max_rows=cfg.memory.max_tracks)


def solve(layer, z_nom, A, b, mask, dtype=torch.float32):
    z = torch.tensor(np.asarray(z_nom, dtype=np.float64)[None, :], dtype=dtype)
    A_t = torch.tensor(np.asarray(A, dtype=np.float64)[None, ...], dtype=dtype)
    b_t = torch.tensor(np.asarray(b, dtype=np.float64)[None, :], dtype=dtype)
    m_t = torch.tensor(np.asarray(mask, dtype=np.float64)[None, :], dtype=dtype)
    z_safe, info = layer(z, A_t, b_t, m_t)
    return z_safe[0].detach().numpy(), info


def empty_rows(rows=4):
    return (np.zeros((rows, 2)), np.full(rows, -1e6), np.zeros(rows))


class TestConstraintConstruction(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()

    def test_head_on_row(self):
        A, b, mask = build_constraint_rows((0.0, 0.0, 0.0), np.array([[0.15, 0.0]]),
                                           self.cfg.robot, self.cfg.cbf, 4)
        self.assertAlmostEqual(mask[0], 1.0)
        np.testing.assert_allclose(A[0], [-1.0, 0.0], atol=1e-12)
        # h = 0.12 - 0.096 = 0.024 -> b = -alpha h + delta_v
        self.assertAlmostEqual(b[0], -0.024 + 0.006, places=12)
        self.assertAlmostEqual(mask[1], 0.0)

    def test_safety_radius_excludes_look_ahead(self):
        points = np.array([[0.15, 0.0]])
        h = barrier_values((0.0, 0.0, 0.0), points, self.cfg.robot)
        # distance from the look-ahead point (0.03, 0) minus body+margin only
        self.assertAlmostEqual(float(h[0]), 0.12 - 0.096, places=12)

    def test_angular_column_uses_look_ahead(self):
        # A surface directly to the left: the normal has no forward component,
        # so only the angular column can act on it.
        A, b, mask = build_constraint_rows((0.0, 0.0, 0.0), np.array([[0.03, 0.20]]),
                                           self.cfg.robot, self.cfg.cbf, 4)
        self.assertAlmostEqual(A[0, 0], 0.0, places=9)
        self.assertAlmostEqual(A[0, 1], -self.cfg.robot.look_ahead_m, places=9)

    def test_rows_are_sorted_nearest_first(self):
        points = np.array([[0.60, 0.0], [0.15, 0.0], [0.35, 0.0]])
        A, b, mask = build_constraint_rows((0.0, 0.0, 0.0), points,
                                           self.cfg.robot, self.cfg.cbf, 4)
        self.assertAlmostEqual(float(np.sum(mask)), 3.0)
        # b = -alpha*h + delta_v, so the nearest surface (smallest h) has the
        # largest right-hand side, i.e. the tightest constraint.
        self.assertGreater(b[0], b[1])
        self.assertGreater(b[1], b[2])

    def test_more_points_than_rows_keeps_the_nearest(self):
        points = np.array([[0.6, 0.0], [0.5, 0.0], [0.4, 0.0], [0.3, 0.0], [0.2, 0.0]])
        A, b, mask = build_constraint_rows((0.0, 0.0, 0.0), points,
                                           self.cfg.robot, self.cfg.cbf, 4)
        self.assertAlmostEqual(float(np.sum(mask)), 4.0)
        self.assertAlmostEqual(b[0], -(0.2 - 0.03 - 0.096) + 0.006, places=12)

    def test_disabled_cbf_produces_no_rows(self):
        cfg = Config()
        cfg.cbf.enabled = False
        A, b, mask = build_constraint_rows((0.0, 0.0, 0.0), np.array([[0.15, 0.0]]),
                                           cfg.robot, cfg.cbf, 4)
        self.assertAlmostEqual(float(np.sum(mask)), 0.0)


class TestQPBehaviour(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.layer = make_layer(self.cfg)

    def test_no_constraints_passes_the_action_through(self):
        A, b, mask = empty_rows()
        z_safe, info = solve(self.layer, [0.8, -0.4], A, b, mask)
        np.testing.assert_allclose(z_safe, [0.8, -0.4], atol=1e-6)
        self.assertTrue(bool(info["feasible"][0]))
        self.assertAlmostEqual(float(info["intervention"][0]), 0.0, places=6)

    def test_far_obstacle_does_not_intervene(self):
        A, b, mask = build_constraint_rows((0.0, 0.0, 0.0), np.array([[0.50, 0.0]]),
                                           self.cfg.robot, self.cfg.cbf, 4)
        z_safe, info = solve(self.layer, [1.0, 0.0], A, b, mask)
        np.testing.assert_allclose(z_safe, [1.0, 0.0], atol=1e-6)

    def test_close_head_on_obstacle_cuts_the_speed(self):
        A, b, mask = build_constraint_rows((0.0, 0.0, 0.0), np.array([[0.15, 0.0]]),
                                           self.cfg.robot, self.cfg.cbf, 4)
        z_safe, _ = solve(self.layer, [1.0, 0.0], A, b, mask)
        # v <= alpha*h - delta_v = 0.018 m/s -> z_v = 0.6
        self.assertAlmostEqual(float(z_safe[0]), 0.6, places=4)
        self.assertAlmostEqual(float(z_safe[1]), 0.0, places=5)

    def test_turning_is_preferred_over_braking(self):
        """Deviation weights are linear:angular = 2:1, so the QP steers first."""
        A, b, mask = build_constraint_rows((0.0, 0.0, 0.0), np.array([[0.16, 0.02]]),
                                           self.cfg.robot, self.cfg.cbf, 4)
        z_safe, _ = solve(self.layer, [1.0, 0.0], A, b, mask)
        self.assertGreater(abs(float(z_safe[1])), 0.0)

    def test_infeasible_stops_but_keeps_turning(self):
        # h = 0.004 < delta_v / alpha, so even v = 0 violates the constraint.
        A, b, mask = build_constraint_rows((0.0, 0.0, 0.0), np.array([[0.13, 0.0]]),
                                           self.cfg.robot, self.cfg.cbf, 4)
        z_safe, info = solve(self.layer, [1.0, 0.7], A, b, mask)
        self.assertFalse(bool(info["feasible"][0]))
        self.assertAlmostEqual(float(z_safe[0]), 0.0, places=6)
        self.assertAlmostEqual(float(z_safe[1]), 0.7, places=6)
        self.assertGreater(float(info["violation"][0]), 0.0)

    def test_masked_rows_are_ignored(self):
        A, b, mask = build_constraint_rows((0.0, 0.0, 0.0), np.array([[0.15, 0.0]]),
                                           self.cfg.robot, self.cfg.cbf, 4)
        mask = np.zeros_like(mask)
        z_safe, _ = solve(self.layer, [1.0, 0.0], A, b, mask)
        np.testing.assert_allclose(z_safe, [1.0, 0.0], atol=1e-6)

    def test_solution_stays_inside_the_action_box(self):
        rng = np.random.default_rng(3)
        for _ in range(200):
            A, b, mask = empty_rows()
            n = int(rng.integers(0, 5))
            for row in range(n):
                A[row] = rng.normal(size=2) * [1.0, 0.05]
                b[row] = rng.normal() * 0.02
                mask[row] = 1.0
            z_nom = np.array([rng.uniform(0, 1), rng.uniform(-1, 1)])
            z_safe, _ = solve(self.layer, z_nom, A, b, mask)
            self.assertGreaterEqual(float(z_safe[0]), -1e-5)
            self.assertLessEqual(float(z_safe[0]), 1.0 + 1e-5)
            self.assertGreaterEqual(float(z_safe[1]), -1.0 - 1e-5)
            self.assertLessEqual(float(z_safe[1]), 1.0 + 1e-5)

    def test_batched_solve_matches_individual_solves(self):
        rng = np.random.default_rng(11)
        batch = 16
        A = rng.normal(size=(batch, 4, 2)) * np.array([1.0, 0.05])
        b = rng.normal(size=(batch, 4)) * 0.02
        mask = (rng.random((batch, 4)) < 0.6).astype(np.float64)
        z_nom = np.stack([rng.uniform(0, 1, batch), rng.uniform(-1, 1, batch)], axis=1)
        z_batch, _ = self.layer(torch.tensor(z_nom, dtype=torch.float32),
                                torch.tensor(A, dtype=torch.float32),
                                torch.tensor(b, dtype=torch.float32),
                                torch.tensor(mask, dtype=torch.float32))
        for index in range(batch):
            single, _ = solve(self.layer, z_nom[index], A[index], b[index], mask[index])
            np.testing.assert_allclose(z_batch[index].detach().numpy(), single, atol=1e-5)

    def test_numpy_helper_matches_the_tensor_path(self):
        A, b, mask = build_constraint_rows((0.0, 0.0, 0.0), np.array([[0.15, 0.0]]),
                                           self.cfg.robot, self.cfg.cbf, 4)
        z_np, summary = self.layer.solve_numpy(np.array([1.0, 0.0]), A, b, mask)
        z_t, _ = solve(self.layer, [1.0, 0.0], A, b, mask)
        np.testing.assert_allclose(z_np, z_t, atol=1e-6)
        self.assertTrue(summary["feasible"])
        self.assertEqual(summary["n_constraints"], 1)


class TestQPOptimality(unittest.TestCase):
    """The enumerated solution must be the true constrained optimum."""

    def setUp(self):
        self.cfg = Config()
        self.layer = make_layer(self.cfg)
        self.scale = np.array([self.cfg.robot.max_linear_mps, self.cfg.robot.max_angular_rps])
        self.weights = np.array([self.cfg.cbf.weight_linear, self.cfg.cbf.weight_angular])
        grid_v = np.linspace(0.0, 1.0, 161)
        grid_w = np.linspace(-1.0, 1.0, 321)
        self.grid = np.stack(np.meshgrid(grid_v, grid_w, indexing="ij"), axis=-1).reshape(-1, 2)

    def brute_force(self, A, b, mask):
        rows = mask > 0.5
        if not np.any(rows):
            feasible = np.ones(len(self.grid), dtype=bool)
        else:
            A_z = (A[rows] * self.scale[None, :])
            feasible = np.all(self.grid @ A_z.T >= b[rows][None, :] - 1e-9, axis=1)
        return feasible

    def test_matches_grid_search(self):
        rng = np.random.default_rng(5)
        checked = 0
        for _ in range(300):
            A = np.zeros((4, 2))
            b = np.full(4, -1e6)
            mask = np.zeros(4)
            for row in range(int(rng.integers(1, 5))):
                angle = rng.uniform(-math.pi, math.pi)
                A[row] = [math.cos(angle), math.sin(angle) * 0.03]
                b[row] = rng.uniform(-0.03, 0.01)
                mask[row] = 1.0
            z_nom = np.array([rng.uniform(0, 1), rng.uniform(-1, 1)])

            z_safe, info = solve(self.layer, z_nom, A, b, mask, dtype=torch.float64)
            feasible = self.brute_force(A, b, mask)
            if not np.any(feasible):
                self.assertFalse(bool(info["feasible"][0]))
                continue
            checked += 1
            costs = (self.weights[None, :] * (self.grid - z_nom[None, :]) ** 2).sum(axis=1)
            grid_best = float(np.min(costs[feasible]))
            qp_cost = float((self.weights * (z_safe - z_nom) ** 2).sum())
            self.assertTrue(bool(info["feasible"][0]))
            self.assertLessEqual(qp_cost, grid_best + 1e-6)
        self.assertGreater(checked, 50)

    def test_solution_satisfies_every_active_constraint(self):
        rng = np.random.default_rng(9)
        for _ in range(300):
            A = np.zeros((4, 2))
            b = np.full(4, -1e6)
            mask = np.zeros(4)
            for row in range(int(rng.integers(1, 5))):
                angle = rng.uniform(-math.pi, math.pi)
                A[row] = [math.cos(angle), math.sin(angle) * 0.03]
                b[row] = rng.uniform(-0.03, -0.001)
                mask[row] = 1.0
            z_nom = np.array([rng.uniform(0, 1), rng.uniform(-1, 1)])
            z_safe, info = solve(self.layer, z_nom, A, b, mask, dtype=torch.float64)
            if not bool(info["feasible"][0]):
                continue
            rows = mask > 0.5
            residual = (A[rows] * self.scale[None, :]) @ z_safe - b[rows]
            self.assertGreaterEqual(float(np.min(residual)), -1e-7)


class TestQPGradients(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.layer = make_layer(self.cfg)

    def _fixed_rows(self, dtype=torch.float64):
        # v <= 0.01 m/s, expressed as -v >= -0.01
        A = torch.tensor([[[-1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]], dtype=dtype)
        b = torch.tensor([[-0.01, -1e6, -1e6, -1e6]], dtype=dtype)
        mask = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype)
        return A, b, mask

    def test_gradcheck_single_active_constraint(self):
        A, b, mask = self._fixed_rows()
        z_nom = torch.tensor([[0.9, 0.2]], dtype=torch.float64, requires_grad=True)

        def fn(z):
            return self.layer(z, A, b, mask)[0]

        self.assertTrue(torch.autograd.gradcheck(fn, (z_nom,), eps=1e-6, atol=1e-7))

    def test_gradcheck_diagonal_constraint(self):
        dtype = torch.float64
        A = torch.tensor([[[-0.8, 0.02], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]], dtype=dtype)
        b = torch.tensor([[-0.012, -1e6, -1e6, -1e6]], dtype=dtype)
        mask = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype)
        z_nom = torch.tensor([[0.85, 0.3]], dtype=dtype, requires_grad=True)

        def fn(z):
            return self.layer(z, A, b, mask)[0]

        self.assertTrue(torch.autograd.gradcheck(fn, (z_nom,), eps=1e-6, atol=1e-7))

    def test_gradient_is_non_trivial_when_the_qp_is_active(self):
        dtype = torch.float64
        A = torch.tensor([[[-0.8, 0.02], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]], dtype=dtype)
        b = torch.tensor([[-0.012, -1e6, -1e6, -1e6]], dtype=dtype)
        mask = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype)
        z_nom = torch.tensor([[0.85, 0.3]], dtype=dtype, requires_grad=True)
        z_safe, _ = self.layer(z_nom, A, b, mask)
        z_safe.sum().backward()
        # The active constraint couples v and omega, so a change in the nominal
        # action must move the safe action in both components.
        self.assertGreater(float(z_nom.grad.abs().sum()), 1e-6)

    def test_gradient_is_identity_when_inactive(self):
        A, b, mask = self._fixed_rows()
        mask = torch.zeros_like(mask)
        z_nom = torch.tensor([[0.5, 0.2]], dtype=torch.float64, requires_grad=True)
        z_safe, _ = self.layer(z_nom, A, b, mask)
        z_safe.sum().backward()
        np.testing.assert_allclose(z_nom.grad.numpy(), [[1.0, 1.0]], atol=1e-9)

    def test_no_nans_in_gradients_with_padded_rows(self):
        A, b, mask = self._fixed_rows(dtype=torch.float32)
        z_nom = torch.tensor([[0.9, 0.2]], dtype=torch.float32, requires_grad=True)
        z_safe, _ = self.layer(z_nom, A, b, mask)
        z_safe.sum().backward()
        self.assertFalse(bool(torch.isnan(z_nom.grad).any()))
        self.assertFalse(bool(torch.isinf(z_nom.grad).any()))


if __name__ == "__main__":
    unittest.main()
