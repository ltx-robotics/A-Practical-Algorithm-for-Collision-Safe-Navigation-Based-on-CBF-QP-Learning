import unittest

from safe_alvik.config import (ACT_DIM, Config, OBS_DIM, OBS_NAMES, StageConfig,
                               default_curriculum)


class TestConfig(unittest.TestCase):
    def test_observation_layout_is_frozen(self):
        self.assertEqual(OBS_DIM, 19)
        self.assertEqual(ACT_DIM, 2)
        self.assertEqual(OBS_NAMES[0], "goal_body_x_norm")
        self.assertEqual(OBS_NAMES[-1], "applied_omega_norm")
        self.assertEqual(len(set(OBS_NAMES)), OBS_DIM)

    def test_defaults_validate(self):
        cfg = Config().validate()
        self.assertAlmostEqual(cfg.robot.max_angular_rps, 0.436332, places=6)
        self.assertAlmostEqual(cfg.map.goal_position[0], 0.26)

    def test_safety_radius_excludes_look_ahead(self):
        cfg = Config()
        self.assertAlmostEqual(cfg.robot.safety_radius_m, 0.096, places=9)
        self.assertNotAlmostEqual(
            cfg.robot.safety_radius_m,
            cfg.robot.body_radius_m + cfg.robot.obstacle_margin_m + cfg.robot.look_ahead_m)

    def test_map_boundary_must_stay_disabled(self):
        cfg = Config()
        cfg.map.consider_map_boundary = True
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_reverse_is_rejected(self):
        cfg = Config()
        cfg.robot.allow_reverse = True
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_curriculum_must_end_at_total_steps(self):
        cfg = Config()
        cfg.train.total_steps = 123456
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_curriculum_boundaries_increase(self):
        cfg = Config()
        cfg.curriculum = [StageConfig(name="a", until_step=100),
                          StageConfig(name="b", until_step=50)]
        cfg.train.total_steps = 50
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_success_radius_not_tighter_than_stop_radius(self):
        cfg = Config()
        cfg.map.success_radius_m = 0.02
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_zone_order_is_left_to_right(self):
        cfg = Config()
        cfg.tof.zone_centers_deg = (-22.5, -7.5, 7.5, 22.5)
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_override_roundtrip(self):
        cfg = Config.from_dict({"cbf": {"alpha": 2.0},
                                "tof": {"rays_per_zone": 5}})
        self.assertAlmostEqual(cfg.cbf.alpha, 2.0)
        self.assertEqual(cfg.tof.rays_per_zone, 5)
        restored = Config.from_dict(cfg.to_dict())
        self.assertAlmostEqual(restored.cbf.alpha, 2.0)
        self.assertEqual(len(restored.curriculum), len(default_curriculum()))
        self.assertEqual(restored.curriculum[0].name, "S0_goal_only")

    def test_unknown_key_is_rejected(self):
        with self.assertRaises(ValueError):
            Config.from_dict({"cbf": {"not_a_key": 1}})


if __name__ == "__main__":
    unittest.main()
