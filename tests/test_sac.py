"""The three methods must differ in exactly the way PLAN.MD section 1 states,
and in no other way. These tests are the guard against silent drift between
`external_qp` and `diff_qp`.
"""

import unittest

import numpy as np
import torch

from safe_alvik.config import ACT_DIM, ALGORITHM_SEMANTICS, Config, MODES, OBS_DIM
from safe_alvik.sac import SACAgent


def small_config():
    cfg = Config()
    cfg.sac.hidden_sizes = (32, 32)
    cfg.sac.batch_size = 16
    return cfg


def tight_rows(rows=4, batch=1):
    """A constraint that caps the forward speed at 10% of the limit."""
    A = np.zeros((batch, rows, 2), dtype=np.float32)
    b = np.full((batch, rows), -1e6, dtype=np.float32)
    mask = np.zeros((batch, rows), dtype=np.float32)
    A[:, 0, 0] = -1.0
    b[:, 0] = -0.003
    mask[:, 0] = 1.0
    return A, b, mask


def make_batch(cfg, batch=16, seed=0, tight=True):
    rng = np.random.default_rng(seed)
    rows = cfg.memory.max_tracks
    if tight:
        A, b, mask = tight_rows(rows, batch)
    else:
        A = np.zeros((batch, rows, 2), dtype=np.float32)
        b = np.full((batch, rows), -1e6, dtype=np.float32)
        mask = np.zeros((batch, rows), dtype=np.float32)
    return {
        "obs": torch.tensor(rng.normal(size=(batch, OBS_DIM)), dtype=torch.float32),
        "next_obs": torch.tensor(rng.normal(size=(batch, OBS_DIM)), dtype=torch.float32),
        "action_nominal": torch.tensor(rng.uniform(0.0, 1.0, size=(batch, ACT_DIM)),
                                       dtype=torch.float32),
        "action_executed": torch.tensor(rng.uniform(0.0, 0.1, size=(batch, ACT_DIM)),
                                        dtype=torch.float32),
        "reward": torch.tensor(rng.normal(size=batch), dtype=torch.float32),
        "done": torch.zeros(batch, dtype=torch.float32),
        "cons_a": torch.tensor(A), "cons_b": torch.tensor(b), "cons_mask": torch.tensor(mask),
        "next_cons_a": torch.tensor(A), "next_cons_b": torch.tensor(b),
        "next_cons_mask": torch.tensor(mask),
    }


class TestModeSemantics(unittest.TestCase):
    def test_only_diff_qp_calls_the_qp_during_updates(self):
        for mode in MODES:
            cfg = small_config()
            agent = SACAgent(cfg, mode, seed=0)
            agent.update(make_batch(cfg))
            if mode == "diff_qp":
                # once for the Bellman next action, once for the actor objective
                self.assertEqual(agent.qp_calls_update, 2, mode)
            else:
                self.assertEqual(agent.qp_calls_update, 0, mode)

    def test_critic_scores_the_action_that_produced_the_transition(self):
        for mode in MODES:
            cfg = small_config()
            agent = SACAgent(cfg, mode, seed=0)
            batch = make_batch(cfg)
            captured = []

            def hook(module, args):
                captured.append(args[1].detach().clone())

            handle = agent.critic.register_forward_pre_hook(hook)
            agent.update(batch)
            handle.remove()

            expected = batch["action_nominal"] if mode == "vanilla" else batch["action_executed"]
            torch.testing.assert_close(captured[0], expected)

    def test_intervention_is_only_measured_inside_the_diff_qp_actor_path(self):
        cfg = small_config()
        metrics = {mode: SACAgent(cfg, mode, seed=0).update(make_batch(cfg)) for mode in MODES}
        self.assertGreater(metrics["diff_qp"]["qp_intervention"], 0.0)
        self.assertEqual(metrics["vanilla"]["qp_intervention"], 0.0)
        self.assertEqual(metrics["external_qp"]["qp_intervention"], 0.0)

    def test_external_and_diff_diverge_only_because_of_the_qp(self):
        cfg = small_config()
        batch = make_batch(cfg, tight=True)
        external = SACAgent(cfg, "external_qp", seed=3)
        differentiable = SACAgent(cfg, "diff_qp", seed=3)
        # Same seed, same initial weights.
        for a, b in zip(external.actor.parameters(), differentiable.actor.parameters()):
            torch.testing.assert_close(a, b)

        # The actor samples noise from the global torch RNG, so both updates
        # must start from the same state; otherwise the comparison measures
        # sampling noise instead of the safety-layer coupling.
        torch.manual_seed(123)
        external.update(batch)
        torch.manual_seed(123)
        differentiable.update(batch)
        difference = max(float((a - b).abs().max())
                         for a, b in zip(external.actor.parameters(),
                                         differentiable.actor.parameters()))
        self.assertGreater(difference, 0.0)

    def test_without_active_constraints_the_two_qp_methods_agree(self):
        cfg = small_config()
        batch = make_batch(cfg, tight=False)
        external = SACAgent(cfg, "external_qp", seed=4)
        differentiable = SACAgent(cfg, "diff_qp", seed=4)
        torch.manual_seed(321)
        external.update(batch)
        torch.manual_seed(321)
        differentiable.update(batch)
        for a, b in zip(external.actor.parameters(), differentiable.actor.parameters()):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


class TestInteraction(unittest.TestCase):
    def setUp(self):
        self.cfg = small_config()
        self.constraints = tuple(x[0] for x in tight_rows())

    def test_vanilla_never_touches_the_safety_layer(self):
        agent = SACAgent(self.cfg, "vanilla", seed=0)
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        for _ in range(5):
            out = agent.act(obs, self.constraints)
            np.testing.assert_allclose(out["z_safe"], out["z_nominal"], atol=1e-12)
        self.assertEqual(agent.qp_calls_interaction, 0)

    def test_qp_methods_correct_the_action(self):
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        for mode in ("external_qp", "diff_qp"):
            agent = SACAgent(self.cfg, mode, seed=0)
            out = agent.act(obs, self.constraints, deterministic=True)
            self.assertGreater(float(out["z_nominal"][0]), float(out["z_safe"][0]))
            self.assertLessEqual(float(out["z_safe"][0]), 0.1 + 1e-6)
            self.assertEqual(agent.qp_calls_interaction, 1)

    def test_real_action_respects_the_speed_limits(self):
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        agent = SACAgent(self.cfg, "diff_qp", seed=0)
        for _ in range(50):
            out = agent.act(obs, self.constraints)
            v, w = out["action_real"]
            self.assertGreaterEqual(v, -1e-9)
            self.assertLessEqual(v, self.cfg.robot.max_linear_mps + 1e-9)
            self.assertLessEqual(abs(w), self.cfg.robot.max_angular_rps + 1e-9)

    def test_random_actions_cover_the_box(self):
        agent = SACAgent(self.cfg, "vanilla", seed=0)
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        samples = np.array([agent.act(obs, self.constraints, random_action=True)["z_nominal"]
                            for _ in range(200)])
        self.assertGreaterEqual(samples[:, 0].min(), 0.0)
        self.assertLessEqual(samples[:, 0].max(), 1.0)
        self.assertLess(samples[:, 1].min(), -0.8)
        self.assertGreater(samples[:, 1].max(), 0.8)

    def test_deterministic_action_is_reproducible(self):
        agent = SACAgent(self.cfg, "vanilla", seed=0)
        obs = np.random.default_rng(0).normal(size=OBS_DIM).astype(np.float32)
        first = agent.act(obs, self.constraints, deterministic=True)["z_nominal"]
        second = agent.act(obs, self.constraints, deterministic=True)["z_nominal"]
        np.testing.assert_allclose(first, second, atol=1e-12)

    def test_eval_runner_reproduces_the_agent_exactly(self):
        """Validation must score the same policy the trainer is training."""
        for mode in MODES:
            agent = SACAgent(self.cfg, mode, seed=0)
            agent.update(make_batch(self.cfg))
            runner = agent.eval_runner()
            self.assertEqual(runner.uses_safety_layer, agent.uses_safety_layer)
            rng = np.random.default_rng(0)
            for _ in range(10):
                obs = rng.normal(size=OBS_DIM).astype(np.float32)
                a = agent.act(obs, self.constraints, deterministic=True)
                b = runner.act(obs, self.constraints, deterministic=True)
                np.testing.assert_allclose(a["z_nominal"], b["z_nominal"], atol=1e-6)
                np.testing.assert_allclose(a["z_safe"], b["z_safe"], atol=1e-6)

    def test_rollout_qp_stays_on_cpu(self):
        """Single-sample solves are launch-bound; they must not run on the GPU."""
        agent = SACAgent(self.cfg, "diff_qp", seed=0)
        self.assertEqual(agent.qp_rollout._action_scale.device.type, "cpu")

    def test_qp_solve_time_is_reported(self):
        agent = SACAgent(self.cfg, "diff_qp", seed=0)
        out = agent.act(np.zeros(OBS_DIM, dtype=np.float32), self.constraints)
        self.assertIn("solve_ms", out["qp"])
        self.assertLess(out["qp"]["solve_ms"], 200.0)


class TestOptimisation(unittest.TestCase):
    def test_entropy_coefficient_respects_its_floor(self):
        cfg = small_config()
        cfg.sac.alpha_init = 0.001
        agent = SACAgent(cfg, "vanilla", seed=0)
        # log_alpha is float32, so the clamped value can sit one ulp below the
        # floor (0.01 -> 0.00999999978). Compare with a tolerance.
        tol = 1e-6
        self.assertGreaterEqual(float(agent.alpha), cfg.sac.alpha_min - tol)
        metrics = agent.update(make_batch(cfg))
        self.assertGreaterEqual(metrics["alpha"], cfg.sac.alpha_min - tol)

    def test_entropy_floor_is_not_binding_by_default(self):
        """A floor that pins alpha turns automatic entropy tuning off without
        saying so. Measured on seed 42: with 0.05 the coefficient pinned within
        12k steps in all three methods; given room it settles near 0.025."""
        self.assertLessEqual(Config().sac.alpha_min, 0.02)

    def test_updates_produce_finite_losses(self):
        for mode in MODES:
            cfg = small_config()
            agent = SACAgent(cfg, mode, seed=0)
            for index in range(5):
                metrics = agent.update(make_batch(cfg, seed=index))
            for key, value in metrics.items():
                self.assertTrue(np.isfinite(value), "%s/%s" % (mode, key))

    def test_target_network_moves_towards_the_critic(self):
        cfg = small_config()
        agent = SACAgent(cfg, "vanilla", seed=0)
        before = [p.clone() for p in agent.critic_target.parameters()]
        for _ in range(5):
            agent.update(make_batch(cfg))
        moved = max(float((a - b).abs().max())
                    for a, b in zip(before, agent.critic_target.parameters()))
        self.assertGreater(moved, 0.0)


class TestCheckpoints(unittest.TestCase):
    def test_roundtrip(self):
        cfg = small_config()
        agent = SACAgent(cfg, "diff_qp", seed=0)
        agent.update(make_batch(cfg))
        state = agent.state_dict()
        restored = SACAgent(cfg, "diff_qp", seed=1)
        restored.load_state_dict(state)
        obs = np.random.default_rng(0).normal(size=OBS_DIM).astype(np.float32)
        constraints = tuple(x[0] for x in tight_rows())
        np.testing.assert_allclose(
            agent.act(obs, constraints, deterministic=True)["z_nominal"],
            restored.act(obs, constraints, deterministic=True)["z_nominal"], atol=1e-6)

    def test_checkpoint_records_the_semantics_and_layout(self):
        cfg = small_config()
        state = SACAgent(cfg, "diff_qp", seed=0).state_dict()
        self.assertEqual(state["algorithm_semantics"], ALGORITHM_SEMANTICS)
        self.assertEqual(state["goal_observation_mode"], "body_vector")
        self.assertEqual(state["obs_dim"], OBS_DIM)
        self.assertEqual(state["mode"], "diff_qp")
        self.assertAlmostEqual(state["controller_goal_radius"], 0.05)

    def test_old_layout_is_refused(self):
        cfg = small_config()
        agent = SACAgent(cfg, "diff_qp", seed=0)
        state = agent.state_dict()
        state["obs_dim"] = 9
        with self.assertRaises(ValueError):
            SACAgent(cfg, "diff_qp", seed=0).load_state_dict(state)

    def test_foreign_semantics_are_refused(self):
        cfg = small_config()
        state = SACAgent(cfg, "diff_qp", seed=0).state_dict()
        state["algorithm_semantics"] = "some-older-project"
        with self.assertRaises(ValueError):
            SACAgent(cfg, "diff_qp", seed=0).load_state_dict(state)

    def test_unknown_mode_is_refused(self):
        with self.assertRaises(ValueError):
            SACAgent(small_config(), "not_a_mode", seed=0)


if __name__ == "__main__":
    unittest.main()
