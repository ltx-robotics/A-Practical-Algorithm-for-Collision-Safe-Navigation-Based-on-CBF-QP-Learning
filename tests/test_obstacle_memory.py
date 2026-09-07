import math
import unittest

import numpy as np

from safe_alvik.config import Config
from safe_alvik.obstacle_memory import ObstacleMemory


def make_memory(cfg=None, confirm_frames=1):
    """Bookkeeping tests use confirm_frames=1 so they exercise clustering,
    expiry and the track cap without the confirmation rule getting in the way.
    The confirmation rule has its own test class below."""
    cfg = cfg or Config()
    cfg.memory.confirm_frames = confirm_frames
    return ObstacleMemory(cfg.memory, cfg.tof.endpoint_accept_range_m)


class TestBookkeeping(unittest.TestCase):
    def test_starts_empty(self):
        memory = make_memory()
        self.assertEqual(len(memory), 0)
        self.assertEqual(memory.points().shape, (0, 2))

    def test_nearby_endpoints_merge_into_one_track(self):
        memory = make_memory()
        memory.update(np.array([[0.20, 0.00]]), 0.0)
        memory.update(np.array([[0.24, 0.02]]), 0.2)   # 4.5 cm away, within 8 cm
        self.assertEqual(len(memory), 1)

    def test_distant_endpoints_create_separate_tracks(self):
        memory = make_memory()
        memory.update(np.array([[0.20, 0.00]]), 0.0)
        memory.update(np.array([[0.20, 0.15]]), 0.2)   # 15 cm away
        self.assertEqual(len(memory), 2)

    def test_track_position_is_smoothed_towards_new_observations(self):
        cfg = Config()
        cfg.memory.update_alpha = 0.5
        memory = make_memory(cfg)
        memory.update(np.array([[0.20, 0.00]]), 0.0)
        memory.update(np.array([[0.24, 0.00]]), 0.2)
        np.testing.assert_allclose(memory.points()[0], [0.22, 0.0], atol=1e-12)

    def test_tracks_expire_after_the_ttl(self):
        cfg = Config()
        cfg.memory.ttl_s = 1.4
        memory = make_memory(cfg)
        memory.update(np.array([[0.20, 0.00]]), 0.0)
        memory.update(None, 1.2)
        self.assertEqual(len(memory), 1)
        memory.update(None, 1.5)
        self.assertEqual(len(memory), 0)

    def test_re_observation_refreshes_the_ttl(self):
        cfg = Config()
        cfg.memory.ttl_s = 1.4
        memory = make_memory(cfg)
        memory.update(np.array([[0.20, 0.00]]), 0.0)
        memory.update(np.array([[0.20, 0.00]]), 1.0)
        memory.update(None, 1.5)
        self.assertEqual(len(memory), 1)

    def test_default_ttl_covers_driving_past_an_obstacle(self):
        """A surface must stay remembered until the robot is clear of it.

        Going around an obstacle puts it outside the 60 deg forward FoV for most
        of the manoeuvre, so the track is the only thing holding it. At 5.0 s
        (0.15 m of travel) tracks expired while the surface was still 0.12 m
        away, leaving the robot blind to something beside it: 47% collisions on
        the corridor layout and 11% on the held-out set. 8.0 s (0.24 m) gives 0%
        and 3%. See PLAN.MD 11.2.7.
        """
        cfg = Config()
        travel = cfg.memory.ttl_s * cfg.robot.max_linear_mps
        self.assertGreater(travel, cfg.robot.body_radius_m)
        self.assertGreaterEqual(travel, 0.20)

    def test_track_count_is_capped_and_keeps_the_newest(self):
        memory = make_memory()
        for index in range(6):
            memory.update(np.array([[0.2 * index, 0.0]]), 0.1 * index)
        self.assertEqual(len(memory), 4)
        xs = sorted(float(p[0]) for p in memory.points())
        self.assertAlmostEqual(xs[0], 0.4, places=9)     # the two oldest dropped
        self.assertAlmostEqual(xs[-1], 1.0, places=9)

    def test_reset_clears_everything(self):
        memory = make_memory()
        memory.update(np.array([[0.2, 0.0]]), 0.0)
        memory.reset()
        self.assertEqual(len(memory), 0)


class TestConfirmationRule(unittest.TestCase):
    """A single ToF false low must not create an obstacle the QP has to obey."""

    def test_one_sighting_does_not_constrain_anything(self):
        memory = make_memory(confirm_frames=2)
        memory.update(np.array([[0.20, 0.00]]), 0.0)
        self.assertEqual(len(memory), 0)
        self.assertEqual(memory.points().shape, (0, 2))
        np.testing.assert_allclose(memory.quadrant_features((0.0, 0.0, 0.0)), 0.0)

    def test_a_second_sighting_confirms_it(self):
        memory = make_memory(confirm_frames=2)
        memory.update(np.array([[0.20, 0.00]]), 0.0)
        memory.update(np.array([[0.20, 0.00]]), 0.2)
        self.assertEqual(len(memory), 1)
        self.assertGreater(memory.quadrant_features((0.0, 0.0, 0.0))[0], 0.0)

    def test_several_zones_in_one_frame_do_not_confirm(self):
        """CL and CR can hit the same surface in the same frame; that is one
        observation, not two."""
        memory = make_memory(confirm_frames=2)
        memory.update(np.array([[0.20, 0.01], [0.20, -0.01]]), 0.0)
        self.assertEqual(len(memory), 0)

    def test_three_frame_setting_needs_three(self):
        memory = make_memory(confirm_frames=3)
        for index in range(2):
            memory.update(np.array([[0.20, 0.00]]), 0.2 * index)
        self.assertEqual(len(memory), 0)
        memory.update(np.array([[0.20, 0.00]]), 0.4)
        self.assertEqual(len(memory), 1)

    def test_unconfirmed_tracks_cannot_evict_confirmed_ones(self):
        memory = make_memory(confirm_frames=2)
        memory.update(np.array([[0.20, 0.00]]), 0.0)
        memory.update(np.array([[0.20, 0.00]]), 0.2)      # confirmed
        self.assertEqual(len(memory), 1)
        for index in range(5):                            # five one-off phantoms
            memory.update(np.array([[-0.2 * (index + 1), 0.30]]), 0.4)
        self.assertEqual(len(memory), 1)
        np.testing.assert_allclose(memory.points()[0], [0.20, 0.0], atol=1e-9)

    def test_default_requires_more_than_one_frame(self):
        self.assertGreater(Config().memory.confirm_frames, 1)


class TestQuadrantFeatures(unittest.TestCase):
    def setUp(self):
        self.memory = make_memory()

    def test_front_left(self):
        self.memory.update(np.array([[0.30, 0.20]]), 0.0)
        features = self.memory.quadrant_features((0.0, 0.0, 0.0))
        self.assertGreater(features[0], 0.0)
        self.assertEqual(float(np.sum(features[1:])), 0.0)

    def test_all_four_slots(self):
        self.memory.update(np.array([[0.2, 0.2], [0.2, -0.2], [-0.2, 0.2], [-0.2, -0.2]]), 0.0)
        self.assertTrue(np.all(self.memory.quadrant_features((0.0, 0.0, 0.0)) > 0.0))

    def test_features_rotate_with_the_robot(self):
        self.memory.update(np.array([[0.30, 0.00]]), 0.0)
        ahead = self.memory.quadrant_features((0.0, 0.0, 0.0))
        turned = self.memory.quadrant_features((0.0, 0.0, math.pi))
        self.assertGreater(ahead[0] + ahead[1], 0.0)
        self.assertGreater(turned[2] + turned[3], 0.0)
        self.assertAlmostEqual(float(ahead[2] + ahead[3]), 0.0, places=12)

    def test_proximity_grows_as_the_surface_gets_closer(self):
        self.memory.update(np.array([[0.60, 0.00]]), 0.0)
        far = self.memory.quadrant_features((0.0, 0.0, 0.0))[0]
        self.memory.reset()
        self.memory.update(np.array([[0.10, 0.00]]), 0.0)
        near = self.memory.quadrant_features((0.0, 0.0, 0.0))[0]
        self.assertGreater(near, far)
        self.assertLessEqual(near, 1.0)

    def test_zero_beyond_the_acceptance_range(self):
        self.memory.update(np.array([[1.50, 0.00]]), 0.0)
        np.testing.assert_allclose(self.memory.quadrant_features((0.0, 0.0, 0.0)), 0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
