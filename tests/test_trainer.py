import csv
import math
import os
import shutil
import tempfile
import unittest

import numpy as np

from safe_alvik.config import Config, MODES, rescale_schedule
from safe_alvik.evaluation import better_than, holdout_scenarios
from safe_alvik.trainer import Trainer


def tiny_config():
    cfg = Config()
    cfg.sac.hidden_sizes = (32, 32)
    cfg.sac.batch_size = 16
    cfg.sac.replay_capacity = 1000
    cfg.sac.start_steps = 40
    cfg.robot.max_episode_steps = 25
    cfg.train.eval_episodes = 2
    cfg.train.stage_eval_episodes = 2
    return rescale_schedule(cfg, 200)


class TestScheduleRescaling(unittest.TestCase):
    def test_boundaries_stay_ordered_and_end_on_the_budget(self):
        cfg = rescale_schedule(Config(), 1000)
        self.assertEqual(cfg.train.total_steps, 1000)
        self.assertEqual(cfg.curriculum[-1].until_step, 1000)
        boundaries = [stage.until_step for stage in cfg.curriculum]
        self.assertEqual(boundaries, sorted(set(boundaries)))
        self.assertEqual(cfg.curriculum[0].until_step, 200)   # 40k of 200k

    def test_pre_fill_and_validation_scale_too(self):
        cfg = rescale_schedule(Config(), 20000)
        self.assertEqual(cfg.sac.start_steps, 500)
        self.assertEqual(cfg.train.eval_interval, 2500)

    def test_identity_when_the_budget_is_unchanged(self):
        cfg = rescale_schedule(Config(), Config().train.total_steps)
        self.assertEqual(cfg.curriculum[0].until_step, 40000)

    def test_too_few_steps_is_rejected(self):
        with self.assertRaises(ValueError):
            rescale_schedule(Config(), 2)


class TestModelSelection(unittest.TestCase):
    def test_a_motionless_policy_cannot_win_on_low_collisions(self):
        mover = {"success": 0.4, "collision": 0.2, "final_distance_true": 0.1, "return": -5.0}
        frozen = {"success": 0.0, "collision": 0.0, "final_distance_true": 0.7, "return": 20.0}
        self.assertTrue(better_than(mover, frozen))
        self.assertFalse(better_than(frozen, mover))

    def test_collision_breaks_a_success_tie(self):
        safe = {"success": 0.5, "collision": 0.05, "final_distance_true": 0.2, "return": 1.0}
        risky = {"success": 0.5, "collision": 0.30, "final_distance_true": 0.1, "return": 9.0}
        self.assertTrue(better_than(safe, risky))

    def test_first_candidate_always_wins(self):
        self.assertTrue(better_than({"success": 0.0}, None))


class TestHoldout(unittest.TestCase):
    def test_holdout_is_identical_for_every_call(self):
        cfg = Config()
        first = holdout_scenarios(cfg, 10)
        second = holdout_scenarios(cfg, 10)
        for a, b in zip(first, second):
            np.testing.assert_allclose(a.start_pose, b.start_pose, atol=1e-12)
            np.testing.assert_allclose(a.obstacles, b.obstacles, atol=1e-12)

    def test_every_holdout_episode_blocks_the_direct_path(self):
        cfg = Config()
        for scenario in holdout_scenarios(cfg, 20):
            self.assertGreaterEqual(scenario.n_obstacles, 1)
            self.assertTrue(scenario.direct_path_blocked)

    def test_holdout_uses_the_fixed_goal(self):
        cfg = Config()
        for scenario in holdout_scenarios(cfg, 10):
            np.testing.assert_allclose(scenario.goal, cfg.map.goal_position, atol=1e-12)


class TestTrainingIntegration(unittest.TestCase):
    """Gate G1: all three methods must run end to end and produce identical
    output structures."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="alvik_trainer_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, mode):
        cfg = tiny_config()
        run_dir = os.path.join(self.tmp, mode)
        os.makedirs(run_dir, exist_ok=True)
        trainer = Trainer(cfg, mode, seed=0, run_dir=run_dir, device="cpu", verbose=False)
        return trainer.train(), run_dir

    def test_all_modes_complete_and_write_the_same_files(self):
        columns = {}
        for mode in MODES:
            summary, run_dir = self._run(mode)
            self.assertEqual(summary["total_steps"], 200)
            for name in ("training.csv", "validation.csv",
                         "validation_current_stage.csv", "run_summary.json"):
                self.assertTrue(os.path.exists(os.path.join(run_dir, name)), name)
            for name in ("best.pt", "latest.pt"):
                self.assertTrue(os.path.exists(os.path.join(run_dir, "checkpoints", name)))
            with open(os.path.join(run_dir, "validation.csv"), encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertGreater(len(rows), 0)
            columns[mode] = sorted(rows[0].keys())
            for row in rows:
                for key, value in row.items():
                    if value in ("", None):
                        continue
                    try:
                        number = float(value)
                    except ValueError:
                        continue
                    self.assertFalse(math.isnan(number), "%s/%s" % (mode, key))
        self.assertEqual(columns["vanilla"], columns["external_qp"])
        self.assertEqual(columns["vanilla"], columns["diff_qp"])

    def test_only_the_qp_modes_touch_the_safety_layer(self):
        summaries = {mode: self._run(mode)[0] for mode in MODES}
        self.assertEqual(summaries["vanilla"]["qp_calls_interaction"], 0)
        self.assertEqual(summaries["vanilla"]["qp_calls_update"], 0)
        self.assertGreater(summaries["external_qp"]["qp_calls_interaction"], 0)
        self.assertEqual(summaries["external_qp"]["qp_calls_update"], 0)
        self.assertGreater(summaries["diff_qp"]["qp_calls_interaction"], 0)
        self.assertGreater(summaries["diff_qp"]["qp_calls_update"], 0)

    def test_curriculum_promotion_is_gated_not_automatic(self):
        cfg = tiny_config()
        cfg.curriculum[0].promote_min_success = 1.01     # unreachable
        run_dir = os.path.join(self.tmp, "gated")
        os.makedirs(run_dir, exist_ok=True)
        trainer = Trainer(cfg, "diff_qp", seed=0, run_dir=run_dir,
                          device="cpu", verbose=False)
        summary = trainer.train()
        self.assertEqual(summary["final_stage"], cfg.curriculum[0].name)


if __name__ == "__main__":
    unittest.main()
