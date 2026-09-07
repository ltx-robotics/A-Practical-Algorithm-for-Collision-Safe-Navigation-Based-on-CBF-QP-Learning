import math
import unittest

import numpy as np

from safe_alvik import tof_model as tof
from safe_alvik.config import Config
from safe_alvik.geometry import as_obstacle_array

DEG = math.pi / 180.0


def place(angle_deg, distance_m, radius_m, cfg):
    """Obstacle whose centre sits at a given bearing/distance from the ToF window."""
    a = angle_deg * DEG
    ox = cfg.robot.tof_offset_x_m + distance_m * math.cos(a)
    oy = distance_m * math.sin(a)
    return as_obstacle_array([[ox, oy, radius_m]])


def make_sensor(cfg, randomize=False, seed=0):
    cfg.randomization.enabled = randomize
    sensor = tof.ToFSensor(cfg.tof, cfg.randomization, cfg.robot.tof_offset_x_m,
                           cfg.robot.tof_to_bumper_m, np.random.default_rng(seed))
    sensor.reset()
    return sensor


class TestToFGeometry(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.pose = (0.0, 0.0, 0.0)
        self.max_range = self.cfg.tof.max_range_m

    def test_empty_scene_saturates_every_channel(self):
        sensor = make_sensor(self.cfg)
        channels = sensor.measure(self.pose, as_obstacle_array(None))
        np.testing.assert_allclose(channels, self.max_range, atol=1e-12)

    def test_front_left_obstacle_lowers_L(self):
        """Regression guard for the single easiest sign error in this project."""
        sensor = make_sensor(self.cfg)
        obstacles = place(22.5, 0.30, 0.04, self.cfg)
        L, CL, C, CR, R = sensor.measure(self.pose, obstacles)
        self.assertLess(L, self.max_range)
        self.assertAlmostEqual(L, 0.26, places=2)
        self.assertAlmostEqual(R, self.max_range, places=9)
        self.assertAlmostEqual(CR, self.max_range, places=9)

    def test_front_right_obstacle_lowers_R(self):
        sensor = make_sensor(self.cfg)
        obstacles = place(-22.5, 0.30, 0.04, self.cfg)
        L, CL, C, CR, R = sensor.measure(self.pose, obstacles)
        self.assertLess(R, self.max_range)
        self.assertAlmostEqual(R, 0.26, places=2)
        self.assertAlmostEqual(L, self.max_range, places=9)
        self.assertAlmostEqual(CL, self.max_range, places=9)

    def test_wide_left_obstacle_lowers_both_left_zones(self):
        sensor = make_sensor(self.cfg)
        obstacles = place(12.0, 0.30, 0.05, self.cfg)
        L, CL, C, CR, R = sensor.measure(self.pose, obstacles)
        self.assertLess(L, self.max_range)
        self.assertLess(CL, self.max_range)
        self.assertAlmostEqual(R, self.max_range, places=9)
        self.assertAlmostEqual(CR, self.max_range, places=9)

    def test_centre_obstacle_lowers_both_middle_zones(self):
        sensor = make_sensor(self.cfg)
        obstacles = place(0.0, 0.30, 0.05, self.cfg)
        L, CL, C, CR, R = sensor.measure(self.pose, obstacles)
        self.assertLess(CL, self.max_range)
        self.assertLess(CR, self.max_range)
        self.assertAlmostEqual(CL, CR, places=9)

    def test_C_is_the_average_of_CL_and_CR(self):
        sensor = make_sensor(self.cfg)
        obstacles = place(6.0, 0.25, 0.05, self.cfg)
        L, CL, C, CR, R = sensor.measure(self.pose, obstacles)
        self.assertAlmostEqual(C, 0.5 * (CL + CR), places=9)
        self.assertNotAlmostEqual(CL, CR, places=6)

    def test_rays_cover_the_whole_zone(self):
        """An obstacle at the far edge of a zone must still be seen."""
        sensor = make_sensor(self.cfg)
        obstacles = place(29.5, 0.30, 0.005, self.cfg)
        L = sensor.measure(self.pose, obstacles)[0]
        self.assertLess(L, self.max_range)

    def test_outside_the_fov_is_invisible(self):
        sensor = make_sensor(self.cfg)
        obstacles = place(45.0, 0.30, 0.02, self.cfg)
        channels = sensor.measure(self.pose, obstacles)
        np.testing.assert_allclose(channels, self.max_range, atol=1e-12)

    def test_measurement_follows_the_robot_heading(self):
        sensor = make_sensor(self.cfg)
        # Same obstacle, robot rotated by +90 deg: what was on the left when
        # facing +x is straight ahead when facing +y... build it explicitly.
        obstacles = as_obstacle_array([[0.0, 0.30, 0.05]])
        channels = sensor.measure((0.0, 0.0, math.pi / 2), obstacles)
        self.assertLess(channels[2], self.max_range)
        sensor.reset()
        channels = sensor.measure((0.0, 0.0, 0.0), obstacles)
        np.testing.assert_allclose(channels, self.max_range, atol=1e-12)

    def test_ray_angles_span_each_zone(self):
        angles = tof.zone_ray_angles(self.cfg.tof) / DEG
        self.assertEqual(angles.shape, (4, 7))
        np.testing.assert_allclose(angles[0], np.linspace(15.0, 30.0, 7), atol=1e-9)
        np.testing.assert_allclose(angles[3], np.linspace(-30.0, -15.0, 7), atol=1e-9)


class TestToFErrorModel(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()

    def test_bias_and_std_interpolation(self):
        sensor = make_sensor(self.cfg)
        # A sensor reading of 108 mm corresponds to a 100 mm bumper truth.
        bias, std = sensor.bias_std_m(np.array([0.108]))
        self.assertAlmostEqual(bias[0] * 1000.0, 3.39, places=6)
        self.assertAlmostEqual(std[0] * 1000.0, 0.77, places=6)
        # Midpoint between 200 and 250 mm truth.
        bias, std = sensor.bias_std_m(np.array([0.233]))
        self.assertAlmostEqual(bias[0] * 1000.0, 0.5 * (1.93 - 0.61), places=6)

    def test_bias_extrapolates_flat_outside_the_table(self):
        sensor = make_sensor(self.cfg)
        near, _ = sensor.bias_std_m(np.array([0.020]))
        far, _ = sensor.bias_std_m(np.array([1.100]))
        self.assertAlmostEqual(near[0] * 1000.0, 5.12, places=6)
        self.assertAlmostEqual(far[0] * 1000.0, 0.67, places=6)

    def test_randomization_disabled_is_exactly_geometric(self):
        sensor = make_sensor(self.cfg, randomize=False)
        obstacles = place(0.0, 0.30, 0.05, self.cfg)
        ideal = tof.ideal_zone_distances((0.0, 0.0, 0.0), obstacles, self.cfg.tof,
                                         self.cfg.robot.tof_offset_x_m)
        channels = sensor.measure((0.0, 0.0, 0.0), obstacles)
        np.testing.assert_allclose(channels, tof.to_channels(ideal), atol=1e-12)

    def test_randomization_perturbs_but_stays_bounded(self):
        sensor = make_sensor(self.cfg, randomize=True, seed=7)
        obstacles = place(0.0, 0.30, 0.05, self.cfg)
        samples = [sensor.measure((0.0, 0.0, 0.0), obstacles) for _ in range(50)]
        stacked = np.stack(samples)
        self.assertTrue(np.all(stacked >= 0.01 - 1e-9))
        self.assertTrue(np.all(stacked <= self.cfg.tof.max_range_m + 1e-9))
        self.assertGreater(float(np.std(stacked[:, 2])), 0.0)

    def test_median_filter_rejects_a_single_false_low(self):
        sensor = make_sensor(self.cfg)
        good = np.full(5, 0.50)
        sensor.push(good)
        sensor.push(good)
        filtered = sensor.push(np.full(5, 0.08))
        np.testing.assert_allclose(filtered, good, atol=1e-12)

    def test_median_filter_follows_a_persistent_drop(self):
        sensor = make_sensor(self.cfg)
        sensor.push(np.full(5, 0.50))
        sensor.push(np.full(5, 0.08))
        filtered = sensor.push(np.full(5, 0.08))
        np.testing.assert_allclose(filtered, 0.08, atol=1e-12)

    def test_repeat_last_returns_the_current_frame(self):
        sensor = make_sensor(self.cfg)
        sensor.push(np.full(5, 0.40))
        np.testing.assert_allclose(sensor.repeat_last(), 0.40, atol=1e-12)


class TestEndpoints(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()

    def test_endpoint_distance_is_exactly_the_measured_range(self):
        sensor = make_sensor(self.cfg)
        obstacles = place(0.0, 0.30, 0.05, self.cfg)
        pose = (0.0, 0.0, 0.0)
        channels = sensor.measure(pose, obstacles)
        points = sensor.endpoints(pose, channels)
        self.assertGreaterEqual(len(points), 2)
        origin = np.array([self.cfg.robot.tof_offset_x_m, 0.0])
        ranges = np.linalg.norm(points - origin[None, :], axis=1)
        expected = channels[[1, 3]]          # CL and CR are the two that fire
        np.testing.assert_allclose(np.sort(ranges), np.sort(expected), atol=1e-12)

    def test_endpoint_bearing_error_is_within_half_a_zone(self):
        """A zone reports a range but not a bearing, so the reconstructed point
        sits on the zone centre line. The resulting error is bounded by the
        chord subtended by half the zone width, and the 30 mm CBF margin is
        sized to absorb it."""
        sensor = make_sensor(self.cfg)
        obstacles = place(0.0, 0.30, 0.05, self.cfg)
        pose = (0.0, 0.0, 0.0)
        channels = sensor.measure(pose, obstacles)
        points = sensor.endpoints(pose, channels)
        centre, radius = obstacles[0, :2], obstacles[0, 2]
        origin = np.array([self.cfg.robot.tof_offset_x_m, 0.0])
        for point in points:
            measured = float(np.linalg.norm(point - origin))
            bound = 2.0 * measured * math.sin(0.5 * self.cfg.tof.zone_width_deg * DEG / 2.0)
            surface_error = abs(float(np.linalg.norm(point - centre)) - radius)
            self.assertLessEqual(surface_error, bound + 1e-9)
            self.assertLess(surface_error, self.cfg.robot.obstacle_margin_m)

    def test_far_readings_are_not_turned_into_endpoints(self):
        sensor = make_sensor(self.cfg)
        obstacles = place(0.0, 0.90, 0.05, self.cfg)
        channels = sensor.measure((0.0, 0.0, 0.0), obstacles)
        self.assertLess(channels[2], self.cfg.tof.max_range_m)
        self.assertEqual(len(sensor.endpoints((0.0, 0.0, 0.0), channels)), 0)

    def test_endpoints_use_the_odometry_pose(self):
        sensor = make_sensor(self.cfg)
        obstacles = place(0.0, 0.30, 0.05, self.cfg)
        channels = sensor.measure((0.0, 0.0, 0.0), obstacles)
        shifted = sensor.endpoints((1.0, 2.0, 0.0), channels)
        base = sensor.endpoints((0.0, 0.0, 0.0), channels)
        np.testing.assert_allclose(shifted - base, [[1.0, 2.0]] * len(base), atol=1e-12)

    def test_c_channel_does_not_create_a_duplicate_endpoint(self):
        sensor = make_sensor(self.cfg)
        obstacles = place(0.0, 0.30, 0.05, self.cfg)
        channels = sensor.measure((0.0, 0.0, 0.0), obstacles)
        # Only the four independent zones may contribute.
        self.assertLessEqual(len(sensor.endpoints((0.0, 0.0, 0.0), channels)), 4)


if __name__ == "__main__":
    unittest.main()
