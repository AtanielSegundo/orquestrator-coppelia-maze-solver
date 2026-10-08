#!/usr/bin/python3
"""
Unit tests for the sweep state machine (runner/sweep.py).

`scan_fit` is tested against ideal geometry elsewhere. What is pinned here is
the part that has to survive a real kart: rotations that under-rotate, telemetry
that arrives stale, headings that return nothing, and a fit that must not block
the render loop.

The fake kart answers the same surface `Recenter` and `Leg` use (`turn`, plus a
telemetry dict), and deliberately executes each commanded turn IMPERFECTLY so
the test can prove the sweep records the heading it measured rather than the one
it asked for -- the error that would otherwise be laundered into the position
fit.

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
import runner.sweep as sweep_mod                                  # noqa: E402
from runner.constants import (                                    # noqa: E402
    RANGE_FIX_MAX_PREDICTED_M, SCAN_FIT_MIN_SAMPLES,
    SWEEP_HALF_SPAN_DEG, SWEEP_STEP_DEG,
)
from runner.range_fix import predict_range                        # noqa: E402
from runner.sweep import Sweep, plan_angles                       # noqa: E402

CELL = 0.30
ORIGIN = (0.0, 0.0, 0.0)


class Corridor:
    rows, cols = 3, 12

    def in_bounds(self, c):
        return 0 <= c[0] < self.rows and 0 <= c[1] < self.cols

    def is_original_wall(self, c):
        return c[0] in (0, 2)


class DeadEnd(Corridor):
    def is_original_wall(self, c):
        return c[0] in (0, 2) or c[1] == 5


class OpenField:
    """No walls at all -- nothing for a sweep to measure against."""
    rows, cols = 5, 5

    def in_bounds(self, c):
        return 0 <= c[0] < self.rows and 0 <= c[1] < self.cols

    def is_original_wall(self, c):
        return False


def center(cell):
    x, y, _ = cop.cell_to_world(ORIGIN, CELL, cell)
    return x, y


class FakeKart:
    """Executes turns with a deliberate scale error and serves telemetry.

    `turn_scale` < 1 means every commanded rotation falls short, which is the
    realistic case (stiction, an open-loop encoder turn). The sweep must still
    produce a good fit, because it records measured headings.
    """

    def __init__(self, view, truth_xy, theta=0.0, turn_scale=1.0,
                 active_polls=2, quantise=True):
        self.view = view
        self.tx, self.ty = truth_xy
        self.theta = theta
        self.turn_scale = turn_scale
        self.active_polls = active_polls
        self.quantise = quantise
        self._active = 0
        self.turns = []

    def turn(self, angle_deg, speed=1.0):
        self.turns.append(angle_deg)
        self.theta += math.radians(angle_deg) * self.turn_scale
        self._active = self.active_polls

    def telemetry(self):
        active = self._active > 0
        if active:
            self._active -= 1
        m, _hit = predict_range(self.view, ORIGIN, CELL, self.tx, self.ty,
                                self.theta, RANGE_FIX_MAX_PREDICTED_M)
        cm = None
        if m is not None:
            cm = round(m * 100.0) if self.quantise else m * 100.0
        return {"goal_active": active, "front_distance_cm": cm}


class FakeClock:
    """A clock the test advances itself.

    The sweep waits SWEEP_SETTLE_S of real time after each rotation, so a test
    polling in a tight loop would otherwise spin forever inside one settle (it
    did: 4000 polls in 0.08 s, every one of them in "settling"). Advancing a
    fake clock per poll keeps the timing logic under test instead of sleeping
    through it.
    """

    def __init__(self, step=0.05):
        self.now = 0.0
        self.step = step

    def __call__(self):
        return self.now

    def tick(self):
        self.now += self.step


def make_sweep(kart, view, believed_xy, believed_th=0.0, clock=None):
    clock = clock or FakeClock()
    s = Sweep(view, ORIGIN, CELL, kart, believed_xy[0], believed_xy[1],
              believed_th, clock=clock)
    return s, clock


def run_sweep(kart, view, believed_xy, believed_th=0.0, max_polls=4000):
    """Drive a Sweep to completion on a fake clock. Returns (outcome, sweep)."""
    s, clock = make_sweep(kart, view, believed_xy, believed_th)
    for _ in range(max_polls):
        out = s.poll(kart.telemetry(), believed_xy[0], believed_xy[1],
                     kart.theta)
        if out != "pending":
            return out, s
        clock.tick()
        if s._phase == "fitting" and s._thread is not None:
            s._thread.join(timeout=10)      # the worker, deterministically
    raise AssertionError(f"sweep never settled (phase={s._phase}, "
                         f"index={s._index}/{len(s.angles)}, "
                         f"samples={len(s.samples)})")


class TestPlanAngles(unittest.TestCase):

    def test_full_span_when_both_sides_have_walls(self):
        view = Corridor()
        cx, cy = center((1, 4))
        angles = plan_angles(view, ORIGIN, CELL, cx, cy, 0.0)
        self.assertAlmostEqual(min(angles), -SWEEP_HALF_SPAN_DEG)
        self.assertAlmostEqual(max(angles), +SWEEP_HALF_SPAN_DEG)
        self.assertAlmostEqual(angles[1] - angles[0], SWEEP_STEP_DEG)

    def test_angles_are_monotonic(self):
        """Visited end to end, so consecutive rotations are one step apart and
        the kart never re-crosses ground it already sampled."""
        view = Corridor()
        cx, cy = center((1, 4))
        angles = plan_angles(view, ORIGIN, CELL, cx, cy, 0.0)
        self.assertEqual(angles, sorted(angles))

    def test_open_field_still_plans_a_sweep(self):
        """Nothing on either flank: the plan does not collapse to nothing, since
        a pillar dead ahead can still carry the fit. The REFUSAL for a maze with
        nothing to see belongs to the fit, not the planner."""
        view = OpenField()
        cx, cy = center((2, 2))
        angles = plan_angles(view, ORIGIN, CELL, cx, cy, 0.0)
        self.assertGreaterEqual(len(angles), SCAN_FIT_MIN_SAMPLES)

    # Frame reminder: theta 0 is +X (col+1, east), +90 is +Y (row-1, north).

    def test_border_cell_skips_headings_off_the_map(self):
        """Top row, driving east along the border: the left (north) end of the
        span faces row -1, which does not exist. +45 is the diagonal into
        (-1, 3) -- off the map as well."""
        view = OpenField()
        cx, cy = center((0, 2))
        angles = plan_angles(view, ORIGIN, CELL, cx, cy, 0.0)
        self.assertAlmostEqual(min(angles), -SWEEP_HALF_SPAN_DEG)
        self.assertAlmostEqual(max(angles), 30.0)
        self.assertEqual(angles, sorted(angles))

    def test_corner_cell_keeps_only_the_inward_quadrant(self):
        """Top-right corner, facing east: north and east are both off the map,
        so only the headings into the row below survive -- too few to fit, so
        the sweep refuses instead of turning for nothing."""
        view = OpenField()
        cx, cy = center((0, 4))
        angles = plan_angles(view, ORIGIN, CELL, cx, cy, 0.0)
        self.assertEqual(angles, [-75.0, -60.0])
        s, _clock = make_sweep(FakeKart(view, (cx, cy)), view, (cx, cy))
        self.assertEqual(s.poll({}, cx, cy, 0.0), "refused")

    def test_facing_off_the_map_drops_the_middle(self):
        """Top row, facing north (straight out): the centre of the span goes,
        both flanks stay, and the order is still one monotonic walk."""
        view = OpenField()
        cx, cy = center((0, 2))
        angles = plan_angles(view, ORIGIN, CELL, cx, cy, math.radians(90.0))
        self.assertEqual(angles, [-75.0, -60.0, 60.0, 75.0])

    def test_interior_cell_loses_nothing(self):
        view = OpenField()
        cx, cy = center((2, 2))
        angles = plan_angles(view, ORIGIN, CELL, cx, cy, 0.0)
        n = int(round(2 * SWEEP_HALF_SPAN_DEG / SWEEP_STEP_DEG)) + 1
        self.assertEqual(len(angles), n)


class TestSweepRun(unittest.TestCase):

    def test_recovers_lateral_offset_end_to_end(self):
        view = DeadEnd()
        cx, cy = center((1, 3))
        lat = 0.035
        kart = FakeKart(view, (cx, cy + lat))
        outcome, s = run_sweep(kart, view, (cx, cy))
        self.assertEqual(outcome, "done")
        self.assertIsNotNone(s.applied_lateral, s.status)
        self.assertAlmostEqual(s.applied_lateral, lat, delta=0.015, msg=s.status)

    def test_returns_to_the_starting_heading(self):
        """The leg that follows was planned from the pre-sweep heading, so the
        sweep must be transparent: it ends pointing where it began."""
        view = DeadEnd()
        cx, cy = center((1, 3))
        kart = FakeKart(view, (cx, cy + 0.02))
        run_sweep(kart, view, (cx, cy))
        self.assertAlmostEqual(math.degrees(kart.theta), 0.0, delta=1.0)

    def test_imperfect_rotations_still_fit(self):
        """Every commanded turn falls 12% short. Because each sample records the
        MEASURED heading, the fit is unaffected -- this is the property that
        makes the sweep usable on a kart with no closed-loop turn."""
        view = DeadEnd()
        cx, cy = center((1, 3))
        lat = 0.03
        kart = FakeKart(view, (cx, cy + lat), turn_scale=0.88)
        outcome, s = run_sweep(kart, view, (cx, cy))
        self.assertEqual(outcome, "done")
        self.assertIsNotNone(s.applied_lateral, s.status)
        self.assertAlmostEqual(s.applied_lateral, lat, delta=0.02, msg=s.status)

    def test_samples_carry_measured_not_commanded_headings(self):
        """Each sample must record the heading telemetry REPORTED, not the one
        that was asked for -- otherwise rotation error is laundered into the
        position fit (see ScanSample's docstring).

        Checked as a set comparison rather than per-sample: with an 80% turn
        scale the measured headings form a visibly different sequence from the
        commanded plan, and the first turn from 0 can legitimately coincide, so
        a per-sample veto would be both flaky and span-dependent.
        """
        view = DeadEnd()
        cx, cy = center((1, 3))
        kart = FakeKart(view, (cx, cy), turn_scale=0.80)
        _out, s = run_sweep(kart, view, (cx, cy))
        self.assertTrue(s.samples, "no samples collected")

        measured = [round(math.degrees(x.theta), 1) for x in s.samples]
        # The plan's own angles, as a sequence; the measured ones must not be it.
        self.assertNotEqual(measured, [round(a, 1) for a in s.angles][:len(measured)],
                            "samples reproduce the commanded plan exactly")
        # And the span actually reached is compressed by the turn scale.
        self.assertLess(max(abs(m) for m in measured), SWEEP_HALF_SPAN_DEG,
                        "measured headings reached the full commanded span "
                        "despite every turn falling 20% short")

    def test_no_walls_anywhere_is_refused_not_guessed(self):
        view = OpenField()
        cx, cy = center((2, 2))
        kart = FakeKart(view, (cx, cy))
        outcome, s = run_sweep(kart, view, (cx, cy))
        self.assertIn(outcome, ("done", "refused"))
        self.assertIsNone(s.applied_lateral, s.status)

    def test_stale_telemetry_does_not_end_a_rotation_early(self):
        """`goal_active` False on the frame right after a turn is a sample from
        before the server saw it. Same guard as Leg/Recenter."""
        view = DeadEnd()
        cx, cy = center((1, 3))
        kart = FakeKart(view, (cx, cy + 0.02), active_polls=5)
        outcome, s = run_sweep(kart, view, (cx, cy))
        self.assertEqual(outcome, "done")
        self.assertGreaterEqual(len(s.samples), SCAN_FIT_MIN_SAMPLES)

    def test_fit_runs_off_the_polling_thread(self):
        """The fit is tens of milliseconds; the render loop must not wear it.
        No single poll() may approach that cost."""
        view = DeadEnd()
        cx, cy = center((1, 3))
        kart = FakeKart(view, (cx, cy + 0.03))
        s, clock = make_sweep(kart, view, (cx, cy))
        worst = 0.0
        worst_phase = None
        # Deliberately NOT joining the worker inside the loop: joining would
        # bill the fit's own runtime to the poll that waits for it and measure
        # nothing. Production spins here for a few frames while the worker runs
        # alongside, which is exactly what this has to stay cheap under.
        for _ in range(20000):
            phase = s._phase
            t0 = time.perf_counter()
            out = s.poll(kart.telemetry(), cx, cy, kart.theta)
            elapsed = time.perf_counter() - t0
            if elapsed > worst:
                worst, worst_phase = elapsed, phase
            if out != "pending":
                break
            clock.tick()
        if s._thread is not None:
            s._thread.join(timeout=10)
        self.assertLess(worst, 0.016,
                        f"a poll() in phase {worst_phase!r} took "
                        f"{worst*1000:.1f} ms -- that is a whole frame")

    def test_slip_suspends_sampling(self):
        """A reading taken while the wheels scrub does not belong to the heading
        it would be recorded against. Same rule range_fix and Recenter apply."""
        view = DeadEnd()
        cx, cy = center((1, 3))
        kart = FakeKart(view, (cx, cy))
        s, clock = make_sweep(kart, view, (cx, cy))
        # Get into the sampling phase, then hold it there with slip.
        for _ in range(400):
            if s._phase == "sampling":
                break
            s.poll(kart.telemetry(), cx, cy, kart.theta)
            clock.tick()
        self.assertEqual(s._phase, "sampling")
        before = len(s.samples)
        for _ in range(50):
            tel = kart.telemetry()
            tel["slip"] = {"slipping": True}
            s.poll(tel, cx, cy, kart.theta)
        self.assertEqual(len(s.samples), before,
                         "samples were taken while slipping")


if __name__ == "__main__":
    unittest.main()
