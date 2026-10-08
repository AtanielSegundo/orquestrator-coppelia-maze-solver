#!/usr/bin/python3
"""
Unit tests for the 1-D longitudinal pose correction (runner/range_fix.py).

The feature moves the kart's believed position on its own, so the tests are
weighted towards the ways that could go WRONG rather than the happy path:

  * every "no measurement" the sensor stack can report must be rejected
    (None / 255 / 300 / <=0 / blind zone) -- the explicit requirement;
  * predictions must come from the ORIGINAL maze walls only, never from
    runtime discoveries, or the feature becomes a divergent SLAM loop that
    localises against obstacles it invented itself;
  * the correction must refuse off-axis headings, implausibly large residuals
    and readings taken while slipping;
  * the sign must be right -- a wrong sign doubles the drift instead of
    removing it, and would look like "the fix makes it worse".

Runs with no CoppeliaSim, no kart and no window: `predict_range` marches the
grid analytically and the view is a plain fake.

    cd maze_solver && python -m unittest discover -s tests
"""
import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
MAZE_SOLVER = os.path.dirname(HERE)
REPO = os.path.dirname(MAZE_SOLVER)
sys.path.insert(0, REPO)
sys.path.insert(0, MAZE_SOLVER)

import Coppelia.helpers as cop                                   # noqa: E402
from runner.constants import (
    SENSOR_MIN_RANGE_CM,                                   # noqa: E402
    RANGE_FIX_ALIGN_TOLERANCE_DEG, RANGE_FIX_MAX_CORRECTION_M,
    RANGE_FIX_MAX_TRUSTED_M, RANGE_FIX_MIN_CORRECTION_M,
    SENSOR_FORWARD_OFFSET_M,
)
from runner.range_fix import (                                   # noqa: E402
    RangeFix, heading_alignment_error_deg, is_trusted_reading,
    is_valid_reading, predict_range,
)

CELL = 0.25


class FakeView:
    """Minimal MazeView stand-in: an original grid plus runtime discoveries,
    with the same original-vs-live split the real one has."""

    def __init__(self, rows, cols, walls=(), discovered=()):
        self.rows = rows
        self.cols = cols
        self._walls = set(walls)
        self.obstacles = set(discovered)

    def in_bounds(self, cell):
        return 0 <= cell[0] < self.rows and 0 <= cell[1] < self.cols

    def is_original_wall(self, cell):
        if not self.in_bounds(cell):
            return True
        return cell in self._walls

    def is_wall(self, cell):            # live maze = file walls + discoveries
        return self.is_original_wall(cell) or cell in self.obstacles


def origin_for(rows, cols):
    return cop.maze_origin(rows, cols, (0.0, 0.0, 0.0), CELL)


class TestSensorValidity(unittest.TestCase):
    """The explicit requirement: use the correction ONLY when the distance
    sensor actually measured something."""

    def test_rejects_none(self):
        self.assertFalse(is_valid_reading(None))

    def test_rejects_255_failed_read_sentinel(self):
        """RPi.GPIO fallback's 'no echo'. Taken as 255 cm it would read as a
        clear road."""
        self.assertFalse(is_valid_reading(255))
        self.assertFalse(is_valid_reading(255.0))

    def test_rejects_300_max_range(self):
        """pigpio reader's far/clear value, also what a FAULTED sensor reports.
        Either way: 'I see nothing', not 'a wall at 3 m'."""
        self.assertFalse(is_valid_reading(300))
        self.assertFalse(is_valid_reading(300.0))
        self.assertFalse(is_valid_reading(301))

    def test_rejects_zero_negative_and_blind_zone(self):
        """Blind-zone cases are derived from SENSOR_MIN_RANGE_CM rather than
        hard-coded: the threshold is a tunable, and a test that pins yesterday's
        value fails on a deliberate change instead of on a regression."""
        for bad in (0, -1, -50.0):
            self.assertFalse(is_valid_reading(bad), f"{bad} accepted")
        just_under = SENSOR_MIN_RANGE_CM - 0.1
        self.assertFalse(is_valid_reading(just_under),
                         f"{just_under} accepted (below the blind-zone floor)")
        self.assertTrue(is_valid_reading(SENSOR_MIN_RANGE_CM),
                        "the floor itself should be valid")

    def test_rejects_nan_and_garbage(self):
        self.assertFalse(is_valid_reading(float("nan")))
        self.assertFalse(is_valid_reading("abc"))
        self.assertFalse(is_valid_reading(object()))

    def test_accepts_a_real_measurement(self):
        for good in (3.0, 10, 45.5, 120, 299.0):
            self.assertTrue(is_valid_reading(good), f"{good} rejected")


class TestHeadingAlignment(unittest.TestCase):
    def test_on_axis_headings_are_zero_error(self):
        for deg in (0, 90, 180, 270, 360, -90):
            self.assertAlmostEqual(
                heading_alignment_error_deg(math.radians(deg)), 0.0, places=6)

    def test_diagonal_is_the_worst_case(self):
        self.assertAlmostEqual(
            heading_alignment_error_deg(math.radians(45)), 45.0, places=6)

    def test_small_offsets_measure_correctly(self):
        self.assertAlmostEqual(
            heading_alignment_error_deg(math.radians(97)), 7.0, places=6)
        self.assertAlmostEqual(
            heading_alignment_error_deg(math.radians(-8)), 8.0, places=6)


class TestPredictRange(unittest.TestCase):
    """Raycast against the map. Geometry: col grows +X, row grows -Y."""

    def setUp(self):
        self.rows, self.cols = 5, 7
        self.origin = origin_for(self.rows, self.cols)

    def _predict(self, view, cell, heading_deg, max_range=2.0):
        wx, wy, _ = cop.cell_to_world(self.origin, CELL, cell)
        return predict_range(view, self.origin, CELL, wx, wy,
                             math.radians(heading_deg), max_range)

    def test_wall_straight_ahead_east(self):
        # Kart at (2,1) facing east (+col); wall at (2,4).
        view = FakeView(self.rows, self.cols, walls={(2, 4)})
        dist, cell = self._predict(view, (2, 1), 0.0)
        self.assertEqual(cell, (2, 4))
        # 3 cells of travel, minus the half cell into the wall's near face,
        # minus the sensor offset the ray already started past.
        expected = 3 * CELL - CELL / 2 - SENSOR_FORWARD_OFFSET_M
        self.assertAlmostEqual(dist, expected, delta=0.02)

    def test_direction_conventions_hold_on_all_four_axes(self):
        """Guards the row/col <-> X/Y mapping: a sign flip here would send the
        correction the wrong way on two of the four axes."""
        for heading, wall in ((0.0, (2, 4)),      # east  -> col +
                              (180.0, (2, 0)),    # west  -> col -
                              (90.0, (0, 3)),     # north -> row -
                              (270.0, (4, 3))):   # south -> row +
            view = FakeView(self.rows, self.cols, walls={wall})
            _dist, cell = self._predict(view, (2, 3), heading)
            self.assertEqual(cell, wall, f"heading {heading} hit {cell}")

    def test_no_wall_in_range_returns_none(self):
        view = FakeView(self.rows, self.cols, walls=set())
        dist, cell = self._predict(view, (2, 1), 0.0, max_range=0.10)
        self.assertIsNone(dist)
        self.assertIsNone(cell)

    def test_leaving_the_map_returns_none(self):
        """Out of bounds is not a wall to measure to."""
        view = FakeView(self.rows, self.cols, walls=set())
        dist, cell = self._predict(view, (2, 1), 0.0, max_range=5.0)
        self.assertIsNone(dist)
        self.assertIsNone(cell)

    def test_nearest_wall_wins(self):
        view = FakeView(self.rows, self.cols, walls={(2, 3), (2, 5)})
        _dist, cell = self._predict(view, (2, 1), 0.0)
        self.assertEqual(cell, (2, 3))

    def test_ignores_runtime_discoveries(self):
        """THE trust boundary. A discovered obstacle is a belief derived from
        the pose; letting it predict a range would let the kart localise
        against its own invention and confirm a wrong pose with growing
        confidence."""
        view = FakeView(self.rows, self.cols, walls={(2, 5)},
                        discovered={(2, 3)})
        self.assertTrue(view.is_wall((2, 3)))          # live maze blocks it
        _dist, cell = self._predict(view, (2, 1), 0.0)
        self.assertEqual(cell, (2, 5), "raycast used a discovered obstacle")

    def test_prediction_accounts_for_the_sensor_offset(self):
        """The reading is measured from the sensor face, ahead of the tracked
        center. Forgetting the offset makes every prediction one offset too
        long, which would show up as a constant bias in the correction."""
        view = FakeView(self.rows, self.cols, walls={(2, 4)})
        wx, wy, _ = cop.cell_to_world(self.origin, CELL, (2, 1))
        with_offset, _ = predict_range(view, self.origin, CELL, wx, wy, 0.0, 2.0)
        # Same geometry, ray started at the sensor position already.
        shifted, _ = predict_range(view, self.origin, CELL,
                                   wx + SENSOR_FORWARD_OFFSET_M, wy, 0.0, 2.0)
        self.assertAlmostEqual(with_offset - shifted, SENSOR_FORWARD_OFFSET_M,
                               delta=0.005)


class TestRangeFixDecisions(unittest.TestCase):
    """When the correction fires, when it declines, and with what sign."""

    #: Predicted range to aim the fixture at. Comfortably inside
    #: RANGE_FIX_MAX_TRUSTED_M so the +-6 cm residuals under test stay
    #: trusted -- the ceiling is exercised on purpose in TestTrustCeiling.
    TARGET_GAP = 0.20

    def setUp(self):
        self.rows, self.cols = 5, 7
        self.origin = origin_for(self.rows, self.cols)
        self.view = FakeView(self.rows, self.cols, walls={(2, 3)})
        self.fix = RangeFix(self.view, self.origin, CELL)
        # Position the pose by the range it should predict, rather than parking
        # it on a cell centre: cell centres put the nearest usable wall ~0.25 m
        # or ~0.50 m out, which leaves no headroom under the trust ceiling.
        wall_face_x = -0.125                            # (2,3)'s near face
        self.wx = wall_face_x - SENSOR_FORWARD_OFFSET_M - self.TARGET_GAP
        _cx, self.wy, _ = cop.cell_to_world(self.origin, CELL, (2, 1))
        self.wth = 0.0                                  # facing east
        self.predicted, _ = predict_range(self.view, self.origin, CELL,
                                          self.wx, self.wy, self.wth, 2.0)
        self.assertAlmostEqual(self.predicted, self.TARGET_GAP, delta=0.005)

    def _evaluate(self, measured_m, **kw):
        return self.fix.evaluate(self.wx, self.wy, self.wth,
                                 measured_m * 100.0, **kw)

    def test_agreement_produces_no_correction(self):
        self.assertIsNone(self._evaluate(self.predicted))
        self.assertIn("ruido", self.fix.rejected_reason)

    def test_wall_further_than_predicted_pulls_the_kart_back(self):
        """Moving forward SHORTENS the predicted range, so measuring further
        than predicted means the believed pose ran AHEAD of the truth -- it
        must come back, i.e. a NEGATIVE correction along the heading.

        Getting this backwards doubles the drift instead of removing it."""
        residual = 0.06
        got = self._evaluate(self.predicted + residual)
        self.assertIsNotNone(got)
        self.assertAlmostEqual(got, -residual, delta=0.005)
        self.assertLess(got, 0.0)

    def test_wall_closer_than_predicted_pushes_the_kart_forward(self):
        residual = -0.06
        got = self._evaluate(self.predicted + residual)
        self.assertIsNotNone(got)
        self.assertAlmostEqual(got, -residual, delta=0.005)
        self.assertGreater(got, 0.0)

    def test_invalid_readings_never_correct(self):
        for bad in (None, 255, 300, 0, -5, float("nan")):
            self.fix.evaluate(self.wx, self.wy, self.wth, bad)
            self.assertIn("sem leitura", self.fix.rejected_reason,
                          f"{bad} was not rejected as invalid")
        self.assertEqual(self.fix.applied, 0)

    def test_slipping_blocks_the_correction(self):
        """The reading is fine, but the pose it would be compared against was
        integrated from the slipping wheels."""
        self.assertIsNone(self._evaluate(self.predicted + 0.06, slipping=True))
        self.assertIn("derrapagem", self.fix.rejected_reason)
        # ...and the same reading is accepted once traction returns.
        self.assertIsNotNone(self._evaluate(self.predicted + 0.06,
                                            slipping=False))

    def test_off_axis_heading_is_refused(self):
        """Off-axis the ray grazes corners, so the residual is dominated by
        heading error -- applying it as a translation makes the pose worse."""
        bad_th = math.radians(RANGE_FIX_ALIGN_TOLERANCE_DEG + 10.0)
        self.assertIsNone(self.fix.evaluate(self.wx, self.wy, bad_th,
                                            (self.predicted + 0.06) * 100.0))
        self.assertIn("fora de eixo", self.fix.rejected_reason)

    def test_small_residual_is_treated_as_quantisation(self):
        tiny = RANGE_FIX_MIN_CORRECTION_M * 0.5
        self.assertIsNone(self._evaluate(self.predicted + tiny))

    def test_huge_residual_is_refused_as_the_wrong_wall(self):
        """Bigger than half a cell means the beam hit something other than the
        wall the map predicts. Teleporting the kart is worse than drifting.

        Tested on the NEAR side: a residual that large in the far direction
        would trip the trust ceiling first (a separate, earlier gate), so it
        would not exercise this check at all."""
        # Just past the cap, not far past: a bigger excursion from a
        # 0.20 m target lands inside the sensor's blind zone, which is a
        # different (earlier) rejection.
        huge = RANGE_FIX_MAX_CORRECTION_M + 0.01
        measured = self.predicted - huge
        self.assertGreater(measured, 0.0, "fixture geometry too tight")
        self.assertIsNone(self._evaluate(measured))
        self.assertIn("grande demais", self.fix.rejected_reason)

    def test_no_predicted_wall_means_nothing_to_compare(self):
        empty = FakeView(self.rows, self.cols, walls=set())
        fix = RangeFix(empty, self.origin, CELL)
        # A trusted reading (<= 30 cm), so the refusal is genuinely about the
        # map having no wall to compare against rather than the trust ceiling.
        self.assertIsNone(fix.evaluate(self.wx, self.wy, self.wth, 20.0))
        self.assertIn("nao preve parede", fix.rejected_reason)

    def test_counters_and_status_line(self):
        self._evaluate(self.predicted + 0.05)
        self._evaluate(self.predicted + 0.05)
        self.assertEqual(self.fix.applied, 2)
        self.assertGreater(self.fix.total_correction_m, 0.0)
        self.assertIn("range-fix", self.fix.status_line())


class TestTrustCeiling(unittest.TestCase):
    """The explicit requirement: only correct when the reading is <= 30 cm.

    Separate from validity on purpose. A 1.5 m reading IS a real measurement --
    it is just not one to move a pose on, because the HC-SR04's cone widens
    with distance and an off-axis heading costs more the further the wall is.
    Conflating the two would also break obstacle mapping, for which a long
    reading is exactly the useful case.
    """

    def test_readings_at_or_under_the_ceiling_are_trusted(self):
        ceiling_cm = RANGE_FIX_MAX_TRUSTED_M * 100.0
        for cm in (5.0, 15.0, 29.0, ceiling_cm):
            self.assertTrue(is_trusted_reading(cm), f"{cm} cm rejected")

    def test_readings_past_the_ceiling_are_not_trusted(self):
        ceiling_cm = RANGE_FIX_MAX_TRUSTED_M * 100.0
        for cm in (ceiling_cm + 0.5, ceiling_cm * 2, 120.0, 299.0):
            self.assertFalse(is_trusted_reading(cm), f"{cm} cm accepted")

    def test_still_valid_just_not_trusted(self):
        """A 1.2 m reading is a real measurement; the two predicates must
        disagree about it, which is the whole reason they are separate."""
        self.assertTrue(is_valid_reading(120.0))
        self.assertFalse(is_trusted_reading(120.0))

    def test_invalid_readings_are_never_trusted(self):
        for bad in (None, 255, 300, 0, -5, float("nan")):
            self.assertFalse(is_trusted_reading(bad))

    def test_evaluate_refuses_a_reading_past_the_ceiling(self):
        rows, cols = 5, 7
        origin = origin_for(rows, cols)
        # Wall far enough that the predicted range exceeds the ceiling.
        view = FakeView(rows, cols, walls={(2, 5)})
        fix = RangeFix(view, origin, CELL)
        wx, wy, _ = cop.cell_to_world(origin, CELL, (2, 1))
        predicted, _ = predict_range(view, origin, CELL, wx, wy, 0.0, 2.0)
        self.assertGreater(predicted, RANGE_FIX_MAX_TRUSTED_M)

        got = fix.evaluate(wx, wy, 0.0, (predicted + 0.05) * 100.0)
        self.assertIsNone(got)
        self.assertIn("limite de confianca", fix.rejected_reason)
        self.assertEqual(fix.applied, 0)

    def test_the_ceiling_is_reported_distinctly_from_no_reading(self):
        """The HUD has to distinguish 'there is no reading' from 'the reading
        is too far to act on' -- they call for different things."""
        rows, cols = 5, 7
        origin = origin_for(rows, cols)
        view = FakeView(rows, cols, walls={(2, 5)})
        fix = RangeFix(view, origin, CELL)
        wx, wy, _ = cop.cell_to_world(origin, CELL, (2, 1))

        fix.evaluate(wx, wy, 0.0, None)
        self.assertIn("sem leitura", fix.rejected_reason)
        fix.evaluate(wx, wy, 0.0, 150.0)
        self.assertIn("limite de confianca", fix.rejected_reason)


class TestCorrectionActuallyReducesError(unittest.TestCase):
    """End-to-end on the maths: inject a known odometry error, apply the
    correction, and assert the pose error shrinks. A sign error would show up
    here as the error doubling."""

    def test_correction_removes_the_injected_drift(self):
        rows, cols = 5, 7
        origin = origin_for(rows, cols)
        view = FakeView(rows, cols, walls={(2, 3)})
        fix = RangeFix(view, origin, CELL)

        # Aim at a 0.20 m gap so the drifts under test stay inside the trust
        # ceiling; see TestRangeFixDecisions.TARGET_GAP.
        true_x = -0.125 - SENSOR_FORWARD_OFFSET_M - 0.20
        _cx, true_y, _ = cop.cell_to_world(origin, CELL, (2, 1))
        wth = 0.0
        # What the sensor really measures, from the TRUE pose.
        measured, _ = predict_range(view, origin, CELL, true_x, true_y, wth, 2.0)

        for drift in (+0.05, -0.05, +0.08, -0.03):
            believed_x = true_x + drift          # odometry over/under-reports
            correction = fix.evaluate(believed_x, true_y, wth, measured * 100.0)
            self.assertIsNotNone(correction, f"declined for drift {drift}")
            corrected_x = believed_x + math.cos(wth) * correction
            self.assertLess(abs(corrected_x - true_x), abs(drift),
                            f"correction did not reduce the error for {drift}")
            self.assertAlmostEqual(corrected_x, true_x, delta=0.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
