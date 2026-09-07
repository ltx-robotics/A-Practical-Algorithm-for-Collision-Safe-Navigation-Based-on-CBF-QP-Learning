import math
import unittest

import numpy as np

from safe_alvik import geometry as g
from safe_alvik.config import Config


class TestGeometry(unittest.TestCase):
    def test_body_frame_roundtrip(self):
        pose = (0.13, -0.21, 0.7)
        point = (-0.05, 0.32)
        body = g.world_to_body(pose, point)
        back = g.body_to_world(pose, body)
        np.testing.assert_allclose(back, point, atol=1e-12)

    def test_left_is_positive_body_y(self):
        # Facing +x, a point at +y is on the robot's left.
        body = g.world_to_body((0.0, 0.0, 0.0), (0.0, 0.5))
        self.assertGreater(body[1], 0.0)
        self.assertAlmostEqual(body[0], 0.0, places=12)
        # Facing +y, a point at -x is on the robot's left.
        body = g.world_to_body((0.0, 0.0, math.pi / 2), (-0.5, 0.0))
        self.assertGreater(body[1], 0.0)

    def test_goal_features_have_no_pi_discontinuity(self):
        goal = (0.26, 0.26)
        previous = None
        for theta in np.linspace(-math.pi, math.pi, 721):
            features = np.array(g.goal_body_features((0.0, 0.0, theta), goal))
            if previous is not None:
                self.assertLess(np.max(np.abs(features - previous)), 0.02)
            previous = features

    def test_goal_distance_matches_euclidean(self):
        bx, by, dist = g.goal_body_features((-0.26, -0.26, 0.0), (0.26, 0.26))
        self.assertAlmostEqual(dist, math.hypot(0.52, 0.52), places=12)
        self.assertAlmostEqual(math.hypot(bx, by), dist, places=12)
        self.assertAlmostEqual(dist, 0.735391, places=5)

    def test_unicycle_straight(self):
        pose = g.unicycle_step((0.0, 0.0, 0.0), 0.03, 0.0, 0.2)
        np.testing.assert_allclose(pose, [0.006, 0.0, 0.0], atol=1e-12)

    def test_unicycle_full_circle_returns_to_start(self):
        pose = np.array([0.1, -0.2, 0.3])
        v, omega, dt = 0.03, 0.436332, 0.01
        steps = int(round(2 * math.pi / omega / dt))
        for _ in range(steps):
            pose = g.unicycle_step(pose, v, omega, dt)
        np.testing.assert_allclose(pose[:2], [0.1, -0.2], atol=2e-4)

    def test_unicycle_turn_in_place(self):
        pose = g.unicycle_step((0.0, 0.0, 0.0), 0.0, 0.436332, 0.2)
        np.testing.assert_allclose(pose[:2], [0.0, 0.0], atol=1e-12)
        self.assertAlmostEqual(pose[2], 0.0872664, places=6)

    def test_wrap_angle(self):
        # The interval is [-pi, pi), so +/-3 pi both land on -pi.
        self.assertAlmostEqual(float(g.wrap_angle(3 * math.pi)), -math.pi, places=12)
        self.assertAlmostEqual(float(g.wrap_angle(-3 * math.pi)), -math.pi, places=12)
        self.assertAlmostEqual(float(g.wrap_angle(0.5)), 0.5, places=12)
        self.assertAlmostEqual(float(g.wrap_angle(-0.5)), -0.5, places=12)
        for angle in (0.0, 1.0, -2.0, 7.0, -9.0):
            wrapped = float(g.wrap_angle(angle))
            self.assertAlmostEqual(math.cos(wrapped), math.cos(angle), places=12)
            self.assertAlmostEqual(math.sin(wrapped), math.sin(angle), places=12)

    def test_action_mapping_forbids_reverse(self):
        cfg = Config()
        z = g.tanh_to_normalized(np.array([-1.0, -1.0]))
        np.testing.assert_allclose(z, [0.0, -1.0], atol=1e-12)
        u = g.normalized_to_real(z, cfg.robot.max_linear_mps, cfg.robot.max_angular_rps)
        self.assertAlmostEqual(u[0], 0.0)
        self.assertAlmostEqual(u[1], -cfg.robot.max_angular_rps)

        z = g.tanh_to_normalized(np.array([1.0, 1.0]))
        u = g.normalized_to_real(z, cfg.robot.max_linear_mps, cfg.robot.max_angular_rps)
        np.testing.assert_allclose(u, [0.03, 0.436332], atol=1e-6)

    def test_action_mapping_roundtrip(self):
        cfg = Config()
        u = np.array([0.017, -0.2])
        z = g.real_to_normalized(u, cfg.robot.max_linear_mps, cfg.robot.max_angular_rps)
        back = g.normalized_to_real(z, cfg.robot.max_linear_mps, cfg.robot.max_angular_rps)
        np.testing.assert_allclose(back, u, atol=1e-12)

    def test_inverse_gain_compensation(self):
        cfg = Config().robot
        v_cmd, w_cmd = g.compensate_command(0.03, 0.436332, cfg.gain_linear,
                                            cfg.gain_angular_ccw, cfg.gain_angular_cw)
        self.assertAlmostEqual(v_cmd, 0.03 / 0.908, places=9)
        self.assertAlmostEqual(w_cmd, 0.436332 / 0.882, places=9)
        # Command stays under the board caps of 3.4 cm/s and 30 deg/s.
        self.assertLess(v_cmd, 0.034)
        self.assertLess(math.degrees(w_cmd), 30.0)

        _, w_cw = g.compensate_command(0.03, -0.436332, cfg.gain_linear,
                                       cfg.gain_angular_ccw, cfg.gain_angular_cw)
        self.assertAlmostEqual(w_cw, -0.436332 / 0.907, places=9)
        self.assertGreater(math.degrees(w_cw), -30.0)

    def test_collision_and_clearance(self):
        cfg = Config()
        obstacles = g.as_obstacle_array([[0.2, 0.0, 0.05]])
        self.assertFalse(g.is_collision((0.0, 0.0, 0.0), obstacles, cfg.robot.body_radius_m))
        self.assertTrue(g.is_collision((0.1, 0.0, 0.0), obstacles, cfg.robot.body_radius_m))
        clearance = g.true_clearance((0.0, 0.0, 0.0), obstacles,
                                     cfg.robot.body_radius_m, cfg.robot.obstacle_margin_m)
        self.assertAlmostEqual(clearance, 0.2 - 0.05 - 0.066 - 0.030, places=12)

    def test_clearance_without_obstacles_is_infinite(self):
        cfg = Config()
        empty = g.as_obstacle_array(None)
        self.assertEqual(g.true_clearance((0.0, 0.0, 0.0), empty,
                                          cfg.robot.body_radius_m,
                                          cfg.robot.obstacle_margin_m), float("inf"))

    def test_segment_circle_distance(self):
        d = g.segment_circle_min_distance((-1.0, 0.0), (1.0, 0.0), (0.0, 0.3))
        self.assertAlmostEqual(d, 0.3, places=12)
        d = g.segment_circle_min_distance((-1.0, 0.0), (-0.5, 0.0), (1.0, 0.0))
        self.assertAlmostEqual(d, 1.5, places=12)


if __name__ == "__main__":
    unittest.main()
