"""The PC's mirror of the board latch must match the board, not resemble it.

The strong test here does not restate the rules in a second hand-written form -
that would only check my transcription against itself. It lifts
`update_tof_safety` straight out of the frozen `real_robot/board/main.py`,
executes that function, and fuzzes the mirror against it. Edit either side and
this fails.
"""

import ast
import os
import unittest

import numpy as np

from real_robot.pc.board_safety import (BoardSafetyMirror, CRITICAL_STOP_FRONT_MM,
                                        HARD_STOP_FRONT_MM, HARD_STOP_RELEASE_MM,
                                        STATE_ACTIVE, STATE_BLOCKED, STATE_CRITICAL,
                                        STATE_TRANSIENT)

BOARD_MAIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "real_robot", "board", "main.py")


def load_board_reference():
    """Extract `update_tof_safety` and its constants from the board file.

    main.py cannot be imported: its first line pulls in `arduino_alvik`. So the
    module is parsed instead and only the constant assignments and that one
    function are executed, in an otherwise empty namespace.
    """
    with open(BOARD_MAIN, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    wanted = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            wanted.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "update_tof_safety":
            wanted.append(node)
    namespace = {}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), BOARD_MAIN, "exec"), namespace)
    return namespace["update_tof_safety"]


class BoardReference:
    """The board's own main loop, reduced to its ToF branch."""

    def __init__(self):
        self.update = load_board_reference()
        self.low_counts = [0, 0, 0]
        self.release_count = 0
        self.stop_active = False
        self.critical_active = False

    def step(self, front_mm):
        (self.stop_active, self.critical_active, filter_state,
         self.release_count) = self.update(front_mm[0], front_mm[1], front_mm[2],
                                           self.low_counts, self.release_count,
                                           self.stop_active, self.critical_active)
        # The telemetry state field is the outer branch of the main loop, not
        # the filter's own label.
        if self.critical_active:
            return filter_state
        if self.stop_active:
            return "TOF_FORWARD_BLOCKED"
        if filter_state == "TOF_TRANSIENT_LOW_IGNORED":
            return filter_state
        return "COMMAND_ACTIVE"

    def drive(self, v, omega):
        if self.critical_active:
            return 0.0, 0.0
        if self.stop_active:
            return 0.0, omega
        return v, omega


class TestAgainstBoardSource(unittest.TestCase):
    def test_the_frozen_board_file_is_still_parseable(self):
        self.assertTrue(callable(load_board_reference()))

    def test_matches_the_board_on_random_sequences(self):
        rng = np.random.default_rng(7)
        for trial in range(200):
            mirror, board = BoardSafetyMirror(), BoardReference()
            # Sample around the thresholds; uniform noise over the full range
            # would almost never produce a release or a re-latch.
            values = rng.choice([40.0, 59.0, 61.0, 80.0, 99.0, 100.0, 101.0,
                                 105.0, 109.0, 110.0, 111.0, 300.0],
                                size=(40, 3))
            for index, frame in enumerate(values):
                predicted = mirror.update(frame)
                expected = board.step([float(v) for v in frame])
                self.assertEqual(predicted, expected,
                                 "trial %d frame %d %s" % (trial, index, frame))
                self.assertEqual(mirror.apply(0.03, 0.4), board.drive(0.03, 0.4))


class TestLatchBehaviour(unittest.TestCase):
    def test_two_low_frames_are_not_enough(self):
        mirror = BoardSafetyMirror()
        for _ in range(2):
            self.assertEqual(mirror.update([90.0, 300.0, 300.0]), STATE_TRANSIENT)
        self.assertEqual(mirror.apply(0.03, 0.4), (0.03, 0.4))
        self.assertEqual(mirror.update([300.0, 300.0, 300.0]), STATE_ACTIVE)

    def test_three_low_frames_on_one_channel_block(self):
        mirror = BoardSafetyMirror()
        for _ in range(2):
            mirror.update([HARD_STOP_FRONT_MM, 300.0, 300.0])
        self.assertEqual(mirror.update([HARD_STOP_FRONT_MM, 300.0, 300.0]), STATE_BLOCKED)
        self.assertEqual(mirror.blocked_activations, 1)

    def test_the_count_is_per_channel_and_does_not_pool(self):
        """CL low, then C low, then CR low is three low frames but no block."""
        mirror = BoardSafetyMirror()
        mirror.update([90.0, 300.0, 300.0])
        mirror.update([300.0, 90.0, 300.0])
        self.assertEqual(mirror.update([300.0, 300.0, 90.0]), STATE_TRANSIENT)
        self.assertFalse(mirror.forward_blocked)

    def test_blocked_keeps_turning(self):
        mirror = BoardSafetyMirror()
        for _ in range(3):
            mirror.update([90.0, 300.0, 300.0])
        # The circular footprint means rotating in place does not enlarge the
        # occupied disk, so the board lets the controller steer away.
        self.assertEqual(mirror.apply(0.03, 0.4), (0.0, 0.4))

    def test_release_needs_all_three_channels_for_three_frames(self):
        mirror = BoardSafetyMirror()
        for _ in range(3):
            mirror.update([90.0, 300.0, 300.0])
        for _ in range(5):
            # 105 mm clears the 100 mm trip but not the 110 mm release.
            self.assertEqual(mirror.update([105.0, 300.0, 300.0]), STATE_BLOCKED)
        for _ in range(2):
            self.assertEqual(mirror.update([HARD_STOP_RELEASE_MM, 300.0, 300.0]),
                             STATE_BLOCKED)
        self.assertEqual(mirror.update([HARD_STOP_RELEASE_MM, 300.0, 300.0]), STATE_ACTIVE)
        self.assertEqual(mirror.apply(0.03, 0.4), (0.03, 0.4))

    def test_one_sample_under_60_mm_stops_everything(self):
        mirror = BoardSafetyMirror()
        self.assertEqual(mirror.update([CRITICAL_STOP_FRONT_MM, 300.0, 300.0]),
                         STATE_CRITICAL)
        self.assertEqual(mirror.apply(0.03, 0.4), (0.0, 0.0))
        self.assertEqual(mirror.critical_activations, 1)

    def test_critical_stays_latched_across_a_noisy_boundary(self):
        """A 59 -> 61 mm flicker must not re-enable rotation."""
        mirror = BoardSafetyMirror()
        mirror.update([59.0, 300.0, 300.0])
        for _ in range(6):
            self.assertEqual(mirror.update([61.0, 300.0, 300.0]), STATE_CRITICAL)
            self.assertEqual(mirror.apply(0.03, 0.4), (0.0, 0.0))

    def test_activation_counters_survive_release(self):
        """The counters are the experimental record, not part of the latch."""
        mirror = BoardSafetyMirror()
        for _ in range(2):
            for _ in range(3):
                mirror.update([90.0, 300.0, 300.0])
            for _ in range(3):
                mirror.update([300.0, 300.0, 300.0])
        self.assertEqual(mirror.blocked_activations, 2)
        self.assertFalse(mirror.forward_blocked)

    def test_side_channels_are_ignored_by_design(self):
        """L and R are not watched, so a side contact is invisible to the board.

        Asserted rather than merely documented because it is a stated
        limitation of the hardware results (PLAN.MD 10.2): the collisions this
        task actually produces are side contacts, and the board neither
        prevents nor records them.
        """
        mirror = BoardSafetyMirror()
        for _ in range(10):
            self.assertEqual(mirror.update([300.0, 300.0, 300.0]), STATE_ACTIVE)
        self.assertEqual(mirror.blocked_activations, 0)

    def test_rejects_a_wrong_length_sample(self):
        with self.assertRaises(ValueError):
            BoardSafetyMirror().update([300.0, 300.0])


if __name__ == "__main__":
    unittest.main()
