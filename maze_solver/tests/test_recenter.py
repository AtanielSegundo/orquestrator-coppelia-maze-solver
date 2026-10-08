#!/usr/bin/python3
"""
Unit tests for physical recentring (runner/recenter.py).

Driven by a fake kart whose odometry is deliberately WRONG: it executes a
commanded move with a slip factor, so a test can prove the loop converges even
when every individual move undershoots. That is the whole justification for
closing on the sensor rather than sending one open-loop nudge, so it is the
property most worth pinning.

Also pinned: that the loop stops at the threshold, gives up rather than
grinding forever, never moves on an invalid reading, and never moves while the
kart reports slipping.

    cd maze_solver && python -m unittest discover -s tests
"""
import math
import os
import sys
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
MAZE_SOLVER = os.path.dirname(HERE)
REPO = os.path.dirname(MAZE_SOLVER)
sys.path.insert(0, REPO)
sys.path.insert(0, MAZE_SOLVER)

import Coppelia.helpers as cop                                    # noqa: E402
from runner.constants import (                                    # noqa: E402
    RANGE_FIX_MAX_TRUSTED_M, RECENTER_MAX_ATTEMPTS, RECENTER_MAX_STEP_M,
    RECENTER_THRESHOLD_M, SENSOR_FORWARD_OFFSET_M,
)
from runner.range_fix import RangeFix, predict_range              # noqa: E402
from runner.recenter import Recenter                              # noqa: E402
from test_range_fix import CELL, FakeView, origin_for             # noqa: E402


class FakeKart:
    """A kart whose commanded moves land short by `slip`.

    `slip=0.0` is a perfect kart; `slip=0.3` travels only 70% of what it was
    told. The recentring loop must converge in both cases -- that is the
    difference between closing on the sensor and trusting the odometry.
    """

    def __init__(self, x, slip=0.0):
        self.x = x                 # true world x (heading is +X throughout)
        self.slip = slip
        self.commands = []
        self.goal_active = False

    def drive_distance(self, distance_m, speed=0.2):
        self.commands.append(distance_m)
        self.x += distance_m * (1.0 - self.slip)
        self.goal_active = True    # cleared by the harness after one poll

    def turn(self, angle_deg, speed=1.0):
        raise AssertionError("recentring must never issue a turn")


_NO_OVERRIDE = object()        # so a test can force a reading of None


class Harness:
    """Runs a Recenter session to completion against a FakeKart.

    Models the thing that makes this subtle: the BELIEF tracks the kart. A
    commanded move of `m` advances odometry (and so the believed pose) by the
    full `m`, while the kart physically travels `m * (1 - slip)` -- because the
    server's drive_distance closes on encoders, which keep counting through a
    slip. A target recomputed from that belief would therefore retreat as fast
    as the kart advanced, which is why Recenter freezes its target range once.

    `offset` places the kart that many metres along +X from the believed pose,
    which is what the session will drive it back onto.

    The believed pose is positioned by the range it should PREDICT
    (`target_gap`), not by parking it on a cell centre. Cell centres put the
    nearest usable wall ~0.25 m or ~0.50 m out, which leaves no headroom under
    RANGE_FIX_MAX_TRUSTED_M once an offset is added -- a kart 7 cm back from a
    0.245 m target reads 0.315 m and is refused, correctly but uselessly for a
    convergence test.
    """

    WALL_FACE_X = -0.125        # near face of cell (2, 3)

    def __init__(self, offset=0.0, slip=0.0, walls=((2, 3),),
                 rows=5, cols=7, wth=0.0, target_gap=0.20):
        self.origin = origin_for(rows, cols)
        self.view = FakeView(rows, cols, walls=set(walls))
        self.fix = RangeFix(self.view, self.origin, CELL)
        self.wth = wth
        _cx, self.wy, _ = cop.cell_to_world(self.origin, CELL, (2, 1))
        self.believed_x = (self.WALL_FACE_X - SENSOR_FORWARD_OFFSET_M
                           - target_gap)
        self.center_x = self.believed_x        # the target the kart drives to
        self.kart = FakeKart(self.believed_x + offset, slip=slip)
        self.session = Recenter(self.fix, self.kart,
                                self.believed_x, self.wy, self.wth)
        self.frames = 0

    def measured_cm(self):
        """What the sensor reads from the kart's TRUE position."""
        d, _ = predict_range(self.view, self.origin, CELL,
                             self.kart.x, self.wy, self.wth, 3.0)
        return None if d is None else d * 100.0

    def run(self, max_frames=400, slipping=False, measured_cm=_NO_OVERRIDE):
        outcome = "pending"
        while outcome == "pending" and self.frames < max_frames:
            self.frames += 1
            cm = (self.measured_cm() if measured_cm is _NO_OVERRIDE
                  else measured_cm)
            tel = {"front_distance_cm": cm,
                   "goal_active": self.kart.goal_active,
                   "slip": {"slipping": slipping}}
            self.kart.goal_active = False     # move completes in one frame
            before = len(self.kart.commands)
            outcome = self.session.poll(tel)
            if len(self.kart.commands) > before:
                # Odometry reports the full commanded move, slip and all.
                self.believed_x += self.kart.commands[-1]
            self._skip_settle()
        return outcome

    def _skip_settle(self):
        """Collapse the post-move settle wait so tests don't sleep."""
        if self.session._phase == "settle":
            self.session._settle_until = 0.0

    def true_error_m(self):
        """How far the kart PHYSICALLY is from the frozen target, along +X."""
        return self.kart.x - self.center_x


class TestRecenterConvergence(unittest.TestCase):
    def test_already_centred_does_nothing(self):
        h = Harness(offset=0.0)
        self.assertEqual(h.run(), "done")
        self.assertEqual(h.kart.commands, [], "moved an already-centred kart")

    def test_moves_the_kart_when_displaced_beyond_the_threshold(self):
        # Kart physically 5 cm further along +X than the belief.
        h = Harness(offset=0.05)
        self.assertEqual(h.run(), "done")
        self.assertTrue(h.kart.commands, "did not move a displaced kart")
        # The kart sits 5 cm nearer the wall than the target, so it reads SHORT
        # and must back away from the wall: a negative (reverse) command.
        self.assertLess(h.kart.commands[0], 0.0,
                        "kart was too close to the wall but drove forward")

    def test_converges_despite_every_move_slipping(self):
        """THE reason this is a closed loop rather than one open-loop nudge.

        Each move lands 50% short, so a single nudge would leave half the error
        behind. 7 cm at 50% slip needs two passes to get inside the 2 cm
        threshold (7 -> 3.5 -> 1.75), which is what makes this test prove
        re-measurement rather than luck."""
        h = Harness(offset=0.07, slip=0.50)
        outcome = h.run()
        self.assertEqual(outcome, "done")
        self.assertGreater(len(h.kart.commands), 1,
                           "closed a 7 cm gap in one move at 50% slip?")
        self.assertLess(abs(h.true_error_m()), RECENTER_THRESHOLD_M)

    def test_an_open_loop_single_nudge_would_not_have_converged(self):
        """The counterfactual, asserted explicitly: one move of the measured
        error leaves `slip` of it behind. This is the failure the loop exists
        to avoid, so it is worth pinning that it really would occur."""
        h = Harness(offset=0.07, slip=0.50)
        first = h.session.poll(
            {"front_distance_cm": h.measured_cm(), "goal_active": False,
             "slip": {}})
        self.assertEqual(first, "pending")
        self.assertEqual(len(h.kart.commands), 1)
        # After that single move the kart is still outside the threshold.
        self.assertGreater(abs(h.true_error_m()), RECENTER_THRESHOLD_M)

    def test_converges_from_both_directions(self):
        for offset in (+0.05, -0.05, +0.09, -0.07):
            h = Harness(offset=offset, slip=0.15)
            self.assertEqual(h.run(), "done", f"offset {offset} did not settle")
            self.assertLess(abs(h.true_error_m()), RECENTER_THRESHOLD_M,
                            f"offset {offset} left {h.true_error_m():.3f} m")

    def test_single_move_is_capped(self):
        """A bad measurement must not drive a long blind move."""
        h = Harness(offset=0.13)
        h.run()
        for cmd in h.kart.commands:
            self.assertLessEqual(abs(cmd), RECENTER_MAX_STEP_M + 1e-9,
                                 f"move {cmd} exceeded the cap")


class TestRecenterGivesUp(unittest.TestCase):
    def test_gives_up_after_max_attempts_on_a_kart_that_cannot_move(self):
        """slip=1.0: the wheels turn, the kart does not. Must stop trying
        rather than grind forever -- 2 cm is not worth stalling a run."""
        h = Harness(offset=0.06, slip=1.0)
        self.assertEqual(h.run(), "giveup")
        self.assertEqual(h.session.attempts, RECENTER_MAX_ATTEMPTS)
        self.assertIn("desistiu", h.session.status)

    def test_giveup_reports_the_span_it_covered(self):
        h = Harness(offset=0.06, slip=1.0)
        h.run()
        summary = h.session.summary()
        self.assertIn("de ", summary)
        self.assertIn("para ", summary)


class TestRecenterRefusals(unittest.TestCase):
    """Never move the kart on evidence too weak to move the belief."""

    def _harness(self):
        return Harness(offset=0.06)

    def test_invalid_readings_never_move_the_kart(self):
        for bad in (None, 255.0, 300.0, 0.0, -5.0):
            h = self._harness()
            self.assertEqual(h.run(measured_cm=bad), "done")
            self.assertEqual(h.kart.commands, [],
                             f"reading {bad} caused a physical move")

    def test_slipping_never_moves_the_kart(self):
        """Same rule as range_fix: a pose integrated from slipping wheels is
        not a basis for deciding where to drive."""
        h = self._harness()
        self.assertEqual(h.run(slipping=True), "done")
        self.assertEqual(h.kart.commands, [])
        self.assertIn("derrapagem", h.session.status)

    def test_no_wall_in_range_means_no_move(self):
        h = Harness(offset=0.06, walls=())
        self.assertEqual(h.run(), "done")
        self.assertEqual(h.kart.commands, [])

    def test_never_issues_a_turn(self):
        """This corrects one DOF -- longitudinal. FakeKart.turn raises."""
        h = self._harness()
        h.run()          # would raise if a turn were issued

    def test_measurements_do_not_inflate_the_range_fix_counters(self):
        """The loop measures repeatedly; those are not applied corrections."""
        h = self._harness()
        h.run()
        self.assertEqual(h.fix.applied, 0,
                         "recentring measurements counted as anchor fixes")
        self.assertEqual(h.fix.total_correction_m, 0.0)


class TestRecenterTrustCeiling(unittest.TestCase):
    """The kart must not be DRIVEN on a reading too far to be trusted.

    The bar is not lower here than for moving the belief -- if anything the
    consequence is larger, since this one actually moves hardware.
    """

    def test_a_reading_past_the_ceiling_never_moves_the_kart(self):
        h = Harness(offset=+0.05)
        self.assertEqual(h.run(measured_cm=RANGE_FIX_MAX_TRUSTED_M * 100 + 10),
                         "done")
        self.assertEqual(h.kart.commands, [],
                         "drove on a reading past the trust ceiling")

    def test_a_reading_at_the_ceiling_is_still_acted_on(self):
        """Boundary: <= 30 cm is trusted, so exactly 30 cm must work."""
        h = Harness(offset=0.0, target_gap=0.22)
        error = h.session._error(
            {"front_distance_cm": RANGE_FIX_MAX_TRUSTED_M * 100.0,
             "slip": {}})
        self.assertIsNotNone(error, "a reading exactly at the ceiling was "
                                    "refused")

    def test_a_target_past_the_ceiling_refuses_the_whole_session(self):
        """If the map predicts a wall further than the ceiling, every pass
        would reject its own measurement. Refuse up front and say so, rather
        than spin through attempts that can never succeed."""
        h = Harness(offset=+0.05, target_gap=0.45)
        self.assertEqual(h.run(), "done")
        self.assertEqual(h.kart.commands, [])
        self.assertIn("limite de confianca", h.session.status)
        self.assertIsNone(h.session.target_range)


class TestRecenterTelemetryHandling(unittest.TestCase):
    def test_waits_for_the_move_to_finish_before_remeasuring(self):
        """`goal_active` True means the move is still running; re-measuring
        mid-move would read a distance the kart is still closing."""
        h = Harness(offset=0.06)
        tel = {"front_distance_cm": h.measured_cm(), "goal_active": False,
               "slip": {}}
        self.assertEqual(h.session.poll(tel),
                         "pending")
        self.assertEqual(len(h.kart.commands), 1)

        # Still active -> stays pending and issues nothing further.
        tel["goal_active"] = True
        for _ in range(5):
            self.assertEqual(h.session.poll(tel),
                             "pending")
        self.assertEqual(len(h.kart.commands), 1, "issued a move mid-move")

    def test_settle_delay_is_respected(self):
        h = Harness(offset=0.06)
        tel = {"front_distance_cm": h.measured_cm(), "goal_active": False,
               "slip": {}}
        h.session.poll(tel)      # issues the move
        tel["goal_active"] = True
        h.session.poll(tel)      # move running
        tel["goal_active"] = False
        h.session.poll(tel)      # -> settle
        self.assertEqual(h.session._phase, "settle")
        self.assertGreater(h.session._settle_until, time.monotonic())


if __name__ == "__main__":
    unittest.main(verbosity=2)
