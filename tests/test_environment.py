import math
import unittest

import numpy as np

from safe_alvik.config import Config, OBS_DIM, StageConfig
from safe_alvik.environment import AlvikEnv, Scenario, ScenarioGenerator
from safe_alvik.geometry import as_obstacle_array, segment_circle_min_distance


def quiet_config():
    cfg = Config()
    cfg.randomization.enabled = False
    return cfg


def simple_scenario(cfg, start=None, goal=None, obstacles=None):
    return Scenario(
        np.asarray(start if start is not None else cfg.map.start_pose, dtype=np.float64),
        np.asarray(goal if goal is not None else cfg.map.goal_position, dtype=np.float64),
        as_obstacle_array(obstacles))


class TestObservation(unittest.TestCase):
    def setUp(self):
        self.cfg = quiet_config()
        self.env = AlvikEnv(self.cfg, seed=0)

    def test_observation_shape_and_range(self):
        obs = self.env.reset(simple_scenario(self.cfg))
        self.assertEqual(obs.shape, (OBS_DIM,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(obs)))
        self.assertTrue(np.all(obs[3:13] >= -1e-6))
        self.assertTrue(np.all(obs[3:13] <= 1.0 + 1e-6))

    def test_empty_scene_saturates_the_tof_block(self):
        obs = self.env.reset(simple_scenario(self.cfg))
        np.testing.assert_allclose(obs[3:8], 1.0, atol=1e-6)
        np.testing.assert_allclose(obs[8:13], 0.0, atol=1e-6)
        np.testing.assert_allclose(obs[13:17], 0.0, atol=1e-6)

    def test_goal_features_come_from_odometry_not_truth(self):
        env = self.env
        env.reset(simple_scenario(self.cfg))
        env.odom_reported = np.array([0.0, 0.0, 0.0])
        env.true_pose = np.array([-0.3, -0.3, 0.0])
        obs = env.observation()
        goal = np.asarray(self.cfg.map.goal_position)
        self.assertAlmostEqual(float(obs[2]) * env.goal_norm,
                               float(np.linalg.norm(goal)), places=5)

    def test_applied_action_is_reported_back(self):
        env = self.env
        env.reset(simple_scenario(self.cfg))
        obs, _, _, info = env.step([0.03, 0.0])
        self.assertAlmostEqual(float(obs[17]), 1.0, places=5)
        self.assertAlmostEqual(info["applied_v"], 0.03, places=9)

    def test_reverse_commands_are_clipped_to_zero(self):
        env = self.env
        env.reset(simple_scenario(self.cfg))
        before = env.true_pose.copy()
        env.step([-0.05, 0.0])
        np.testing.assert_allclose(env.true_pose[:2], before[:2], atol=1e-12)


class TestDynamicsAndTermination(unittest.TestCase):
    def setUp(self):
        self.cfg = quiet_config()
        self.env = AlvikEnv(self.cfg, seed=1)

    def test_straight_motion_matches_the_speed_limit(self):
        env = self.env
        env.reset(simple_scenario(self.cfg, start=[-0.35, -0.35, 0.0]))
        env.step([0.03, 0.0])
        self.assertAlmostEqual(env.true_pose[0], -0.35 + 0.03 * 0.2, places=9)

    def test_timeout_terminates_without_success(self):
        cfg = quiet_config()
        cfg.robot.max_episode_steps = 20
        env = AlvikEnv(cfg, seed=2)
        env.reset(simple_scenario(cfg, start=[-0.35, -0.35, math.pi]))
        done = False
        steps = 0
        while not done:
            _, _, done, info = env.step([0.0, 0.0])
            steps += 1
            self.assertLessEqual(steps, 20)
        self.assertTrue(info["timeout"])
        self.assertFalse(info["success"])

    def test_collision_uses_the_true_pose(self):
        env = self.env
        scenario = simple_scenario(self.cfg, start=[-0.20, 0.0, 0.0],
                                   goal=[0.35, 0.0], obstacles=[[0.0, 0.0, 0.05]])
        env.reset(scenario)
        done = False
        info = {}
        for _ in range(60):
            _, _, done, info = env.step([0.03, 0.0])
            if done:
                break
        self.assertTrue(info["collision"])
        self.assertLess(info["true_clearance"], 0.0)

    def test_controller_stops_on_odometry_but_success_needs_the_truth(self):
        env = self.env
        goal = np.asarray(self.cfg.map.goal_position)
        env.reset(simple_scenario(self.cfg))
        env.odom_pose = np.array([goal[0], goal[1], 0.0])
        env.true_pose = np.array([goal[0] - 0.30, goal[1], 0.0])
        _, _, done, info = env.step([0.0, 0.0])
        self.assertTrue(done)
        self.assertTrue(info["controller_stopped"])
        self.assertFalse(info["success"])

    def test_success_requires_both_criteria(self):
        env = self.env
        goal = np.asarray(self.cfg.map.goal_position)
        env.reset(simple_scenario(self.cfg))
        env.odom_pose = np.array([goal[0], goal[1], 0.0])
        env.true_pose = np.array([goal[0] - 0.04, goal[1], 0.0])
        _, reward, done, info = env.step([0.0, 0.0])
        self.assertTrue(done)
        self.assertTrue(info["success"])
        self.assertGreater(reward, self.cfg.reward.goal_bonus - 1.0)

    def test_stop_radius_is_tighter_than_the_success_radius(self):
        # 50 mm control stop, 70 mm independent success: 20 mm of odometry slack
        # without loosening the reported success criterion.
        self.assertLess(self.cfg.map.controller_goal_radius_m,
                        self.cfg.map.success_radius_m)


class TestMapBoundaryIsInvisible(unittest.TestCase):
    """The printed border has no wall and no ToF echo, and nothing refers to it."""

    def setUp(self):
        self.cfg = quiet_config()
        self.env = AlvikEnv(self.cfg, seed=3)

    def test_driving_off_the_map_neither_terminates_nor_is_penalised(self):
        env = self.env
        env.reset(simple_scenario(self.cfg, start=[0.35, -0.35, 0.0], goal=[0.26, 0.26]))
        for _ in range(60):
            obs, reward, done, info = env.step([0.03, 0.0])
            self.assertFalse(done)
            self.assertFalse(info["collision"])
            np.testing.assert_allclose(obs[3:8], 1.0, atol=1e-6)
        self.assertGreater(env.true_pose[0], self.cfg.map.half_extent_m)

    def test_no_constraint_rows_are_created_outside_the_map(self):
        env = self.env
        env.reset(simple_scenario(self.cfg, start=[0.39, 0.39, 0.0]))
        A, b, mask = env.constraints
        self.assertAlmostEqual(float(np.sum(mask)), 0.0)


class TestRandomization(unittest.TestCase):
    def test_disabled_randomization_is_deterministic(self):
        cfg = quiet_config()
        first = AlvikEnv(cfg, seed=5)
        second = AlvikEnv(cfg, seed=9999)
        scenario = simple_scenario(cfg, obstacles=[[0.0, 0.0, 0.05]])
        obs_a = first.reset(scenario)
        obs_b = second.reset(scenario)
        np.testing.assert_allclose(obs_a, obs_b, atol=1e-12)
        for _ in range(10):
            obs_a, _, _, _ = first.step([0.02, 0.1])
            obs_b, _, _, _ = second.step([0.02, 0.1])
        np.testing.assert_allclose(obs_a, obs_b, atol=1e-12)

    def test_enabled_randomization_changes_the_rollout(self):
        cfg = Config()
        env = AlvikEnv(cfg, seed=5)
        scenario = simple_scenario(cfg, obstacles=[[0.0, 0.0, 0.05]])
        env.reset(scenario)
        for _ in range(20):
            env.step([0.03, 0.0])
        first = env.true_pose.copy()
        env.reset(scenario)
        for _ in range(20):
            env.step([0.03, 0.0])
        self.assertGreater(float(np.linalg.norm(env.true_pose - first)), 1e-6)

    def _straight_run_odometry_errors(self, count=100, steps=123):
        """Odometry error after driving the 0.735 m task distance straight."""
        cfg = Config()
        absolute, signed = [], []
        for seed in range(count):
            env = AlvikEnv(cfg, seed=seed)
            env.reset(simple_scenario(cfg, start=[-0.35, -0.35, 0.0], goal=[0.35, -0.35]))
            for _ in range(steps):
                env.step([0.03, 0.0])
            delta = env.odom_reported[:2] - env.true_pose[:2]
            absolute.append(float(np.linalg.norm(delta)))
            signed.append(float(delta[0]))
        return np.array(absolute), np.array(signed)

    def test_calibrated_odometry_has_no_systematic_along_track_bias(self):
        """The measured +2.7% overestimate is corrected on the PC side.

        Left uncorrected it costs 0.735 * 0.027 = 19.8 mm in one direction,
        which is the entire budget between the 50 mm stop radius and the 70 mm
        success radius, and it caused most of the G2 held-out failures.
        """
        _, signed = self._straight_run_odometry_errors()
        self.assertLess(abs(float(np.mean(signed))), 0.005)

    def test_odometry_error_fits_the_task_budget(self):
        cfg = Config()
        budget = cfg.map.success_radius_m - cfg.map.controller_goal_radius_m
        absolute, _ = self._straight_run_odometry_errors()
        self.assertLess(float(np.mean(absolute)), budget)

    def test_disabled_randomization_gives_perfect_odometry(self):
        cfg = quiet_config()
        env = AlvikEnv(cfg, seed=0)
        env.reset(simple_scenario(cfg, start=[-0.35, -0.35, 0.0]))
        for _ in range(50):
            env.step([0.03, 0.05])
        np.testing.assert_allclose(env.odom_reported, env.true_pose, atol=1e-9)

    def test_odometry_drifts_away_from_the_truth(self):
        cfg = Config()
        env = AlvikEnv(cfg, seed=11)
        env.reset(simple_scenario(cfg, start=[-0.35, -0.35, 0.0]))
        for _ in range(100):
            _, _, done, info = env.step([0.03, 0.05])
            if done:
                break
        self.assertGreater(info["odom_error_m"], 0.0)


class TestScenarioGenerator(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.generator = ScenarioGenerator(self.cfg)
        self.stage = StageConfig(name="t", until_step=1, n_obstacles_min=1,
                                 n_obstacles_max=2, randomize_start_goal=False)

    def test_blocking_scenarios_really_block_the_corridor(self):
        rng = np.random.default_rng(0)
        blocked = 0
        for _ in range(30):
            scenario = self.generator.sample(rng, self.stage, blocking_probability=1.0)
            if scenario.n_obstacles == 0:
                continue
            blocked += 1
            gaps = [segment_circle_min_distance(scenario.start_pose[:2], scenario.goal,
                                                row[:2]) - row[2] - self.cfg.robot.body_radius_m
                    for row in scenario.obstacles]
            self.assertLess(min(gaps), 0.0)
            self.assertTrue(scenario.direct_path_blocked)
        self.assertGreater(blocked, 20)

    def test_non_blocking_scenarios_leave_the_corridor_open(self):
        rng = np.random.default_rng(1)
        for _ in range(30):
            scenario = self.generator.sample(rng, self.stage, blocking_probability=0.0)
            if scenario.n_obstacles == 0:
                continue
            self.assertFalse(scenario.direct_path_blocked)
            self.assertGreater(scenario.straight_path_gap_m, 0.0)

    def test_start_and_goal_stay_clear_of_obstacles(self):
        rng = np.random.default_rng(2)
        needed = self.cfg.robot.body_radius_m + self.cfg.obstacles.start_goal_clearance_m
        for _ in range(40):
            scenario = self.generator.sample(rng, self.stage)
            for row in scenario.obstacles:
                self.assertGreaterEqual(
                    float(np.linalg.norm(row[:2] - scenario.start_pose[:2])) - row[2],
                    needed - 1e-9)
                self.assertGreaterEqual(
                    float(np.linalg.norm(row[:2] - scenario.goal)) - row[2], needed - 1e-9)

    def test_obstacles_do_not_overlap(self):
        rng = np.random.default_rng(3)
        for _ in range(40):
            scenario = self.generator.sample(rng, self.stage)
            rows = scenario.obstacles
            for i in range(rows.shape[0]):
                for j in range(i + 1, rows.shape[0]):
                    gap = (float(np.linalg.norm(rows[i, :2] - rows[j, :2]))
                           - rows[i, 2] - rows[j, 2])
                    self.assertGreaterEqual(gap, self.cfg.obstacles.min_surface_gap_m - 1e-9)

    def test_a_feasible_path_always_exists(self):
        rng = np.random.default_rng(4)
        for _ in range(30):
            scenario = self.generator.sample(rng, self.stage)
            self.assertTrue(self.generator._path_exists(scenario.start_pose,
                                                        scenario.goal,
                                                        scenario.obstacles))

    def test_obstacle_free_stage_produces_no_obstacles(self):
        stage = StageConfig(name="s0", until_step=1, n_obstacles_min=0,
                            n_obstacles_max=0, randomize_start_goal=True)
        rng = np.random.default_rng(5)
        for _ in range(20):
            scenario = self.generator.sample(rng, stage, progress=0.5)
            self.assertEqual(scenario.n_obstacles, 0)

    def test_randomized_endpoints_respect_the_distance_curriculum(self):
        stage = StageConfig(name="s0", until_step=1, n_obstacles_min=0,
                            n_obstacles_max=0, randomize_start_goal=True,
                            start_distance_range_m=(0.30, 0.735))
        rng = np.random.default_rng(6)
        distances = []
        for _ in range(40):
            scenario = self.generator.sample(rng, stage, progress=0.0)
            distances.append(float(np.linalg.norm(scenario.goal - scenario.start_pose[:2])))
        self.assertLess(max(distances), 0.31)
        distances = []
        for _ in range(40):
            scenario = self.generator.sample(rng, stage, progress=1.0)
            distances.append(float(np.linalg.norm(scenario.goal - scenario.start_pose[:2])))
        self.assertGreater(max(distances), 0.60)

    def test_fixed_stage_keeps_the_paper_endpoints(self):
        rng = np.random.default_rng(7)
        scenario = self.generator.sample(rng, self.stage)
        np.testing.assert_allclose(scenario.goal, self.cfg.map.goal_position, atol=1e-12)
        offset = np.linalg.norm(scenario.start_pose[:2]
                                - np.asarray(self.cfg.map.start_pose[:2]))
        self.assertLessEqual(float(offset), math.hypot(0.02, 0.02) + 1e-9)


if __name__ == "__main__":
    unittest.main()
