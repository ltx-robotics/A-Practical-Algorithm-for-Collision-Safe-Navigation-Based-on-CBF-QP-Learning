
import math
import socket
import time
import unittest

import numpy as np

from real_robot.pc import protocol
from real_robot.pc.link import BoardLink
from real_robot.pc.runtime import RealRobotRuntime
from safe_alvik.config import Config, OBS_DIM
from safe_alvik.sac import SACAgent


def small_config():
    cfg = Config()
    cfg.sac.hidden_sizes = (32, 32)
    return cfg


def actor_state(cfg):
    return SACAgent(cfg, "diff_qp", seed=0).actor.state_dict()


def telemetry_packet(sequence=1, board_ms=1000, pose=(0.0, 0.0, 0.0),
                     channels_m=(1.2, 1.2, 1.2, 1.2, 1.2),
                     command=(0.0, 0.0), feedback=(0.0, 0.0),
                     state="COMMAND_ACTIVE", last_sequence=0):
    """One board frame, built the way the board builds it (mm, deg, cm/s)."""
    millimetres = [value * 1000.0 for value in channels_m]
    fields = ["TEL", str(sequence), str(board_ms),
              "%.3f" % (pose[0] * 1000.0), "%.3f" % (pose[1] * 1000.0),
              "%.4f" % math.degrees(pose[2])]
    fields += ["%.2f" % value for value in millimetres]
    fields.append("%.2f" % (millimetres[2] - 8.0))
    fields += ["%.4f" % (command[0] * 100.0), "%.4f" % math.degrees(command[1])]
    fields += ["%.4f" % (feedback[0] * 100.0), "%.4f" % math.degrees(feedback[1])]
    fields += [state, str(last_sequence)]
    return ",".join(fields).encode("ascii")


def frame(**kwargs):
    decoded = protocol.decode_telemetry(telemetry_packet(**kwargs))
    assert decoded is not None
    return decoded


# --------------------------------------------------------------------- wire

class TestProtocol(unittest.TestCase):
    def test_decode_converts_the_board_units_to_si(self):
        telemetry = frame(pose=(-0.26, 0.13, math.radians(45.0)),
                          channels_m=(0.5, 0.4, 0.3, 0.35, 0.6),
                          command=(0.033, math.radians(20.0)),
                          feedback=(0.030, math.radians(18.0)))
        np.testing.assert_allclose(telemetry.pose, [-0.26, 0.13, math.radians(45.0)],
                                   atol=1e-6)
        np.testing.assert_allclose(telemetry.channels, [0.5, 0.4, 0.3, 0.35, 0.6],
                                   atol=1e-6)
        self.assertAlmostEqual(telemetry.command_v, 0.033, places=6)
        self.assertAlmostEqual(telemetry.feedback_omega, math.radians(18.0), places=6)
        # C_front is the display-only bumper-referenced value, 8 mm shorter.
        self.assertAlmostEqual(telemetry.c_front_m, 0.292, places=6)

    def test_a_malformed_frame_is_dropped_not_guessed(self):
        self.assertIsNone(protocol.decode_telemetry(b"TEL,1,2,3"))
        self.assertIsNone(protocol.decode_telemetry(b"NOTTEL," + b",".join([b"0"] * 17)))
        self.assertIsNone(protocol.decode_telemetry(b""))
        truncated = telemetry_packet().rsplit(b",", 1)[0]
        self.assertIsNone(protocol.decode_telemetry(truncated))

    def test_a_corrupt_number_does_not_raise(self):
        fields = telemetry_packet().split(b",")
        fields[7] = b"nan-ish"
        self.assertIsNone(protocol.decode_telemetry(b",".join(fields)))

    def test_command_encoding_uses_board_units_and_caps(self):
        payload = protocol.encode_command(4, 0.030, math.radians(20.0))
        parts = payload.decode().split(",")
        self.assertEqual(parts[0], "CMD")
        self.assertEqual(int(parts[1]), 4)
        self.assertAlmostEqual(float(parts[2]), 3.0, places=3)     # cm/s
        self.assertAlmostEqual(float(parts[3]), 20.0, places=3)    # deg/s

    def test_the_caps_are_a_backstop_against_a_compensation_bug(self):
        parts = protocol.encode_command(1, 5.0, math.radians(400.0)).decode().split(",")
        self.assertAlmostEqual(float(parts[2]), protocol.BOARD_MAX_LINEAR_CM_S, places=3)
        self.assertAlmostEqual(float(parts[3]), protocol.BOARD_MAX_ANGULAR_DEG_S, places=3)
        # Reverse is not in the action space and the board clamps it away too.
        back = protocol.encode_command(2, -0.02, 0.0).decode().split(",")
        self.assertAlmostEqual(float(back[2]), 0.0, places=6)

    def test_reset_is_sent_in_millimetres_and_degrees(self):
        parts = protocol.encode_reset(9, (-0.26, -0.26, math.pi / 2)).decode().split(",")
        self.assertEqual(parts[0], "RESET")
        self.assertAlmostEqual(float(parts[2]), -260.0, places=3)
        self.assertAlmostEqual(float(parts[4]), 90.0, places=3)

    def test_board_blocking_lists_pass_through_states_not_blocking_ones(self):
        self.assertFalse(frame(state="COMMAND_ACTIVE").board_blocking)
        self.assertFalse(frame(state="TOF_TRANSIENT_LOW_IGNORED").board_blocking)
        self.assertTrue(frame(state="TOF_FORWARD_BLOCKED").board_blocking)
        self.assertTrue(frame(state="HARD_TOF_STOP_CRITICAL").board_blocking)
        self.assertTrue(frame(state="COMMAND_TIMEOUT_STOP").board_blocking)
        # An unrecognised state must read as blocking, never as fine.
        self.assertTrue(frame(state="SOMETHING_NEW").board_blocking)


# ----------------------------------------------------------------- odometry

class TestOdometryCorrection(unittest.TestCase):
    def setUp(self):
        self.cfg = small_config()
        self.runtime = RealRobotRuntime(self.cfg, actor_state(self.cfg))
        self.gain = self.cfg.robot.odom_distance_gain

    def test_the_gain_is_applied_to_the_increment_not_the_absolute_pose(self):
        """The distinguishing case: a start pose away from the origin.

        The board reports its own frame; scaling that absolute pose would move
        the start point itself. Here the robot begins at (-0.26, -0.26) and the
        board reports 102.7 mm of travel, so the corrected pose must be
        -0.26 + 0.100, not (-0.26 + 0.1027) / 1.027.
        """
        start = self.cfg.map.start_pose
        self.runtime.reset(start)
        self.runtime.step(frame(pose=start, board_ms=1000))
        result = self.runtime.step(frame(pose=(start[0] + 0.1027, start[1], 0.0),
                                         board_ms=1200))
        self.assertAlmostEqual(result.odom_pose[0], start[0] + 0.1, places=6)
        self.assertNotAlmostEqual(result.odom_pose[0], (start[0] + 0.1027) / self.gain,
                                  places=4)

    def test_a_turning_path_is_corrected_leg_by_leg(self):
        self.runtime.reset((0.0, 0.0, 0.0))
        self.runtime.step(frame(pose=(0.0, 0.0, 0.0), board_ms=1000))
        self.runtime.step(frame(pose=(0.1027, 0.0, 0.0), board_ms=1200))
        result = self.runtime.step(frame(pose=(0.1027, 0.2054, math.pi / 2),
                                         board_ms=1400))
        np.testing.assert_allclose(result.odom_pose[:2], [0.1, 0.2], atol=1e-6)

    def test_backtracking_cancels_out(self):
        self.runtime.reset((0.0, 0.0, 0.0))
        for index, x in enumerate([0.0, 0.1027, 0.0]):
            result = self.runtime.step(frame(pose=(x, 0.0, 0.0), board_ms=1000 + 200 * index))
        np.testing.assert_allclose(result.odom_pose[:2], [0.0, 0.0], atol=1e-9)

    def test_heading_is_not_divided_by_a_distance_gain(self):
        """1.027 was measured on straight-line travel; it says nothing about yaw."""
        self.runtime.reset((0.0, 0.0, 0.0))
        self.runtime.step(frame(pose=(0.0, 0.0, 0.0), board_ms=1000))
        result = self.runtime.step(frame(pose=(0.0, 0.0, math.radians(90.0)),
                                         board_ms=1200))
        self.assertAlmostEqual(result.odom_pose[2], math.radians(90.0), places=6)

    def test_heading_increments_wrap(self):
        self.runtime.reset((0.0, 0.0, math.radians(170.0)))
        self.runtime.step(frame(pose=(0.0, 0.0, math.radians(170.0)), board_ms=1000))
        result = self.runtime.step(frame(pose=(0.0, 0.0, math.radians(-170.0)),
                                         board_ms=1200))
        # +20 deg across the branch cut, not -340.
        self.assertAlmostEqual(result.odom_pose[2], math.radians(-170.0), places=6)


# -------------------------------------------------------------- observation

class TestObservation(unittest.TestCase):
    def setUp(self):
        self.cfg = small_config()
        self.runtime = RealRobotRuntime(self.cfg, actor_state(self.cfg))

    def test_layout_matches_the_frozen_order(self):
        start = self.cfg.map.start_pose
        self.runtime.reset(start)
        channels = (0.90, 0.80, 0.70, 0.60, 0.50)
        for index in range(3):
            result = self.runtime.step(frame(pose=start, channels_m=channels,
                                             feedback=(0.015, math.radians(10.0)),
                                             board_ms=1000 + 200 * index))
        obs = result.observation
        self.assertEqual(obs.shape, (OBS_DIM,))
        self.assertEqual(obs.dtype, np.float32)

        goal = np.asarray(self.cfg.map.goal_position)
        distance = float(np.linalg.norm(goal - np.asarray(start[:2])))
        norm = 2.0 * self.cfg.map.half_extent_m
        self.assertAlmostEqual(float(obs[2]), distance / norm, places=5)
        # Heading 0 with the goal up and to the right: ahead and to the left.
        self.assertGreater(obs[0], 0.0)
        self.assertGreater(obs[1], 0.0)

        expected = np.asarray(channels) / self.cfg.tof.max_range_m
        np.testing.assert_allclose(obs[3:8], expected, atol=1e-5)
        np.testing.assert_allclose(obs[8:13], 1.0 - expected, atol=1e-5)
        np.testing.assert_allclose(obs[3:8] + obs[8:13], np.ones(5), atol=1e-5)

        self.assertAlmostEqual(float(obs[17]), 0.015 / self.cfg.robot.max_linear_mps,
                               places=5)
        self.assertAlmostEqual(float(obs[18]),
                               math.radians(10.0) / self.cfg.robot.max_angular_rps,
                               places=5)

    def test_out_of_range_returns_saturate_rather_than_exceed_one(self):
        self.runtime.reset((0.0, 0.0, 0.0))
        result = self.runtime.step(frame(channels_m=(2.0, 2.0, 2.0, 2.0, 2.0)))
        np.testing.assert_allclose(result.observation[3:8], np.ones(5), atol=1e-6)

    def test_the_simulated_tof_error_model_is_not_applied_to_a_real_sensor(self):
        """Only the median runs here. `corrupt` would add bias and noise twice."""
        self.runtime.reset((0.0, 0.0, 0.0))
        channels = (0.55, 0.45, 0.35, 0.40, 0.60)
        for index in range(3):
            result = self.runtime.step(frame(channels_m=channels, board_ms=1000 + 200 * index))
        np.testing.assert_allclose(result.channels, channels, atol=1e-9)

    def test_the_median_rejects_a_single_bad_frame(self):
        self.runtime.reset((0.0, 0.0, 0.0))
        self.runtime.step(frame(channels_m=(0.5,) * 5, board_ms=1000))
        self.runtime.step(frame(channels_m=(0.5,) * 5, board_ms=1200))
        result = self.runtime.step(frame(channels_m=(0.05,) * 5, board_ms=1400))
        np.testing.assert_allclose(result.channels, [0.5] * 5, atol=1e-9)


# ------------------------------------------------------- policy and command

class TestControlChain(unittest.TestCase):
    def setUp(self):
        self.cfg = small_config()
        self.runtime = RealRobotRuntime(self.cfg, actor_state(self.cfg))

    def test_only_diff_qp_is_allowed_on_hardware(self):
        for mode in ("vanilla", "external_qp"):
            with self.assertRaises(ValueError):
                RealRobotRuntime(self.cfg, actor_state(self.cfg), mode=mode)

    def test_the_command_is_the_inverse_gain_compensated_desired_speed(self):
        self.runtime.reset(self.cfg.map.start_pose)
        result = self.runtime.step(frame(pose=self.cfg.map.start_pose))
        robot = self.cfg.robot
        self.assertAlmostEqual(result.command[0], result.desired[0] / robot.gain_linear,
                               places=9)
        gain_w = (robot.gain_angular_ccw if result.desired[1] >= 0
                  else robot.gain_angular_cw)
        self.assertAlmostEqual(result.command[1], result.desired[1] / gain_w, places=9)
        self.assertGreaterEqual(abs(result.command[1]), abs(result.desired[1]))

    def test_the_command_stays_inside_what_the_board_accepts(self):
        """Compensation scales the command up; it must not exceed the board cap."""
        self.runtime.reset(self.cfg.map.start_pose)
        result = self.runtime.step(frame(pose=self.cfg.map.start_pose))
        self.assertLessEqual(result.command[0] * 100.0, protocol.BOARD_MAX_LINEAR_CM_S + 1e-9)
        self.assertLessEqual(abs(math.degrees(result.command[1])),
                             protocol.BOARD_MAX_ANGULAR_DEG_S + 1e-9)

    def test_an_empty_floor_produces_no_constraints(self):
        self.runtime.reset(self.cfg.map.start_pose)
        for index in range(4):
            result = self.runtime.step(frame(pose=self.cfg.map.start_pose,
                                             board_ms=1000 + 200 * index))
        self.assertEqual(result.n_constraints, 0)
        np.testing.assert_allclose(result.z_safe, result.z_nominal, atol=1e-6)

    def test_an_obstacle_ahead_becomes_a_constraint_and_slows_the_robot(self):
        self.runtime.reset((0.0, 0.0, 0.0))
        close = (1.2, 0.14, 0.13, 0.14, 1.2)
        for index in range(1 + self.cfg.memory.confirm_frames + 1):
            result = self.runtime.step(frame(pose=(0.0, 0.0, 0.0), channels_m=close,
                                             board_ms=1000 + 200 * index))
        self.assertGreater(result.n_constraints, 0)
        self.assertLessEqual(result.z_safe[0], result.z_nominal[0] + 1e-6)
        self.assertTrue(result.feasible)

    def test_a_track_needs_confirm_frames_before_it_constrains(self):
        """One frame of a phantom return must not brake the robot (PLAN.MD 8.6)."""
        self.assertGreaterEqual(self.cfg.memory.confirm_frames, 2)
        self.runtime.reset((0.0, 0.0, 0.0))
        result = self.runtime.step(frame(pose=(0.0, 0.0, 0.0),
                                         channels_m=(1.2, 0.14, 0.13, 0.14, 1.2)))
        self.assertEqual(result.n_constraints, 0)

    def test_reached_uses_the_controller_radius_not_the_success_radius(self):
        goal = self.cfg.map.goal_position
        self.runtime.reset((goal[0], goal[1], 0.0))
        result = self.runtime.step(frame(pose=(goal[0], goal[1], 0.0)))
        self.assertTrue(result.reached)

        just_outside = self.cfg.map.controller_goal_radius_m + 0.005
        self.runtime.reset((goal[0] - just_outside, goal[1], 0.0))
        result = self.runtime.step(frame(pose=(goal[0] - just_outside, goal[1], 0.0)))
        self.assertFalse(result.reached)
        # The 20 mm between the two radii is the odometry error budget, so a
        # pose inside the stop radius can still be a failed trial.
        self.assertGreater(self.cfg.map.success_radius_m,
                           self.cfg.map.controller_goal_radius_m)

    def test_the_board_state_is_mirrored_and_counted(self):
        self.runtime.reset((0.0, 0.0, 0.0))
        for index in range(3):
            result = self.runtime.step(frame(pose=(0.0, 0.0, 0.0),
                                             channels_m=(1.2, 0.09, 0.09, 0.09, 1.2),
                                             board_ms=1000 + 200 * index))
        self.assertEqual(result.predicted_board_state, "TOF_FORWARD_BLOCKED")
        self.assertEqual(self.runtime.board.blocked_activations, 1)

    def test_the_mirror_reads_raw_board_samples_not_the_median(self):
        """The board never sees the PC's filter, so the mirror must not either."""
        self.runtime.reset((0.0, 0.0, 0.0))
        for index in range(2):
            self.runtime.step(frame(channels_m=(1.2,) * 5, board_ms=1000 + 200 * index))
        result = self.runtime.step(frame(channels_m=(1.2, 1.2, 0.05, 1.2, 1.2),
                                         board_ms=1400))
        # The median still reports 1.2 m, but the board has already braked.
        self.assertAlmostEqual(result.channels[2], 1.2, places=6)
        self.assertEqual(result.predicted_board_state, "HARD_TOF_STOP_CRITICAL")

    def test_the_step_time_follows_the_board_clock_when_plausible(self):
        self.runtime.reset((0.0, 0.0, 0.0))
        self.runtime.step(frame(board_ms=1000))
        self.runtime.step(frame(board_ms=1300))
        self.assertAlmostEqual(self.runtime.time_s, self.cfg.robot.control_dt_s + 0.3,
                               places=6)

    def test_an_implausible_board_interval_falls_back_to_the_nominal_period(self):
        """ticks_ms wraps and packets are reordered; neither may stretch time."""
        self.runtime.reset((0.0, 0.0, 0.0))
        self.runtime.step(frame(board_ms=1000))
        self.runtime.step(frame(board_ms=999999))
        self.assertAlmostEqual(self.runtime.time_s, 2 * self.cfg.robot.control_dt_s,
                               places=6)

    def test_applied_speed_prefers_board_feedback(self):
        self.runtime.reset((0.0, 0.0, 0.0))
        self.runtime.step(frame(feedback=(0.021, math.radians(-7.0))))
        np.testing.assert_allclose(self.runtime.applied,
                                   [0.021, math.radians(-7.0)], atol=1e-6)

    def test_describe_records_every_constant_the_run_depended_on(self):
        facts = self.runtime.describe()
        for key in ("odom_distance_gain", "gain_linear", "gain_angular_ccw",
                    "gain_angular_cw", "memory_ttl_s", "memory_confirm_frames",
                    "cbf_alpha", "cbf_safety_radius_m", "success_radius_m"):
            self.assertIn(key, facts)
        self.assertAlmostEqual(facts["odom_distance_gain"],
                               self.cfg.robot.odom_distance_gain)
        self.assertAlmostEqual(facts["control_hz"], 5.0)

    def test_reset_clears_the_filter_and_the_memory(self):
        self.runtime.reset((0.0, 0.0, 0.0))
        for index in range(4):
            self.runtime.step(frame(channels_m=(1.2, 0.14, 0.13, 0.14, 1.2),
                                    board_ms=1000 + 200 * index))
        self.assertGreater(len(self.runtime.memory), 0)
        self.runtime.reset(self.cfg.map.start_pose)
        self.assertEqual(len(self.runtime.memory), 0)
        self.assertEqual(self.runtime.board.state, "COMMAND_ACTIVE")
        np.testing.assert_allclose(self.runtime.odom_pose, self.cfg.map.start_pose)


# --------------------------------------------------------------------- link

class TestBoardLink(unittest.TestCase):
    """Loopback only. The point is that a dry run cannot move the robot."""

    def setUp(self):
        self.board = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.board.bind(("127.0.0.1", 0))
        self.board.settimeout(0.3)
        self.port = self.board.getsockname()[1]
        self.addCleanup(self.board.close)

    def make_link(self, armed):
        link = BoardLink("127.0.0.1", command_port=self.port, telemetry_port=0,
                         armed=armed)
        self.addCleanup(link.close)
        return link

    def test_a_disarmed_link_sends_absolutely_nothing(self):
        link = self.make_link(armed=False)
        self.assertFalse(link.send_command(0.03, 0.4))
        self.assertFalse(link.send_reset((0.0, 0.0, 0.0)))
        link.brake()
        with self.assertRaises(socket.timeout):
            self.board.recvfrom(256)
        self.assertEqual(link.sent_commands, 0)

    def test_an_armed_link_sends_a_command_the_board_would_accept(self):
        link = self.make_link(armed=True)
        self.assertTrue(link.send_command(0.030, math.radians(20.0)))
        payload, _ = self.board.recvfrom(256)
        parts = payload.decode().split(",")
        self.assertEqual(parts[0], "CMD")
        self.assertAlmostEqual(float(parts[2]), 3.0, places=3)

    def test_sequences_increase_strictly_across_every_packet_type(self):
        """The board drops anything not strictly greater than the last accepted."""
        link = self.make_link(armed=True)
        link.send_reset((0.0, 0.0, 0.0))
        for _ in range(5):
            link.send_command(0.01, 0.0)
        link.brake()
        sequences = []
        while True:
            try:
                payload, _ = self.board.recvfrom(256)
            except socket.timeout:
                break
            sequences.append(int(payload.decode().split(",")[1]))
        self.assertGreaterEqual(len(sequences), 7)
        self.assertEqual(sequences, sorted(set(sequences)))

    def test_receive_latest_discards_a_backlog(self):
        """A queued stale frame is worse than waiting; only the newest is used."""
        link = self.make_link(armed=False)
        port = link._recv_socket.getsockname()[1]
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addCleanup(sender.close)
        for sequence in range(1, 6):
            sender.sendto(telemetry_packet(sequence=sequence, board_ms=1000 * sequence),
                          ("127.0.0.1", port))
        newest = link.receive_latest(0.5)
        self.assertIsNotNone(newest)
        self.assertEqual(newest.sequence, 5)
        self.assertIsNone(link.receive_latest(0.05))

    def test_a_junk_datagram_is_counted_and_ignored(self):
        link = self.make_link(armed=False)
        port = link._recv_socket.getsockname()[1]
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addCleanup(sender.close)
        sender.sendto(b"garbage", ("127.0.0.1", port))
        sender.sendto(telemetry_packet(sequence=42), ("127.0.0.1", port))
        newest = link.receive_latest(0.5)
        self.assertEqual(newest.sequence, 42)
        self.assertEqual(link.dropped_frames, 1)

    def test_no_telemetry_returns_none_rather_than_blocking_forever(self):
        link = self.make_link(armed=False)
        self.assertIsNone(link.receive_latest(0.05))

    def test_numbering_resumes_above_what_the_board_already_accepted(self):
        """A second run in the same board session must not restart at 1.

        The board keeps `last_command_sequence` until it reboots and drops
        anything not strictly greater. Restarting the count silently disables
        every command for the whole session - the robot sits in
        COMMAND_TIMEOUT_STOP and looks exactly like a dead link. Cost one
        hardware trial on 2026-08-24 before it was spotted.
        """
        link = self.make_link(armed=True)
        self.assertEqual(link.sync_sequence(857), 858)
        link.send_command(0.01, 0.0)
        payload, _ = self.board.recvfrom(256)
        self.assertGreater(int(payload.decode().split(",")[1]), 857)

    def test_the_heartbeat_stops_driving_when_telemetry_goes_silent(self):
        """Repeating the last command into a dead link means driving blind.

        The heartbeat exists so the board's 300 ms dead-man switch does not fire
        between 5 Hz decisions. If telemetry dies, that same heartbeat would
        happily keep a 3 cm/s command alive for as long as the loop waits, so it
        has to decay to zero on its own.
        """
        link = BoardLink("127.0.0.1", command_port=self.port, telemetry_port=0,
                         armed=True, telemetry_grace_s=0.0)
        self.addCleanup(link.close)
        with link:
            link.send_command(0.03, 0.4)
            self.board.recvfrom(256)              # the commanded one
            time.sleep(0.25)                      # let a few heartbeats fire
        speeds = []
        while True:
            try:
                payload, _ = self.board.recvfrom(256)
            except socket.timeout:
                break
            parts = payload.decode().split(",")
            if parts[0] == "CMD":
                speeds.append((float(parts[2]), float(parts[3])))
        self.assertTrue(speeds, "no heartbeat was sent at all")
        self.assertTrue(all(s == (0.0, 0.0) for s in speeds),
                        "heartbeat kept driving on stale telemetry: %s" % speeds)
        self.assertGreater(link.stale_heartbeats, 0)

    def test_a_fresh_frame_re_enables_the_heartbeat(self):
        link = self.make_link(armed=True)
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addCleanup(sender.close)
        sender.sendto(telemetry_packet(sequence=1),
                      ("127.0.0.1", link._recv_socket.getsockname()[1]))
        self.assertTrue(link.receive_frames(0.5))
        link.send_command(0.02, 0.0)
        payload, _ = self.board.recvfrom(256)
        self.assertAlmostEqual(float(payload.decode().split(",")[2]), 2.0, places=3)

    def test_syncing_never_moves_the_sequence_backwards(self):
        """A rebooted board reports -1; that must not rewind our numbering."""
        link = self.make_link(armed=True)
        link.sync_sequence(500)
        link.send_command(0.01, 0.0)
        self.board.recvfrom(256)
        self.assertGreaterEqual(link.sync_sequence(-1), 501)
        self.assertGreaterEqual(link.sync_sequence(3), 501)

    def test_silence_yields_no_frames_so_the_caller_can_brake(self):
        """PLAN.MD 11.3: losing telemetry must brake, not coast on stale data."""
        link = self.make_link(armed=True)
        self.assertEqual(link.receive_frames(0.05), [])
        link.brake()
        payload, _ = self.board.recvfrom(256)
        self.assertTrue(payload.decode().startswith("BRAKE,"))

    def test_closing_brakes_even_when_the_caller_raised(self):
        """The one command that must survive a crash on step 1."""
        link = BoardLink("127.0.0.1", command_port=self.port, telemetry_port=0,
                         armed=True)
        try:
            with link:
                link.send_command(0.03, 0.0)
                raise RuntimeError("policy blew up")
        except RuntimeError:
            pass
        payloads = []
        while True:
            try:
                payload, _ = self.board.recvfrom(256)
            except socket.timeout:
                break
            payloads.append(payload.decode())
        self.assertGreaterEqual(sum(p.startswith("BRAKE,") for p in payloads), 2)
        self.assertTrue(payloads[-1].startswith("BRAKE,"))


if __name__ == "__main__":
    unittest.main()
