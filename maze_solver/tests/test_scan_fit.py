#!/usr/bin/python3
"""
Unit tests for the 2-DOF sweep fit (runner/scan_fit.py).

The properties pinned here are the ones that decide whether this feature helps
or hurts, and they are not symmetric:

  * LATERAL must be recovered accurately. It is the whole reason to spend the
    rotations -- no other corrector in the stack can observe it, and it is the
    component that puts the kart into a wall.
  * LONGITUDINAL must be REFUSED when the geometry does not constrain it. In an
    open corridor the cost surface is exactly flat along the heading, so the
    search returns whatever it touched first; acting on that turns a 2 cm error
    into a 16 cm one. A test that only checked "a fit was produced" would pass
    while the feature drove the kart into a wall.

Mazes are plain stubs rather than real .maze files: the fit only ever asks
`in_bounds` and `is_original_wall`, and a hand-written grid makes the geometry
of each case obvious. test_sweep_tcc_maze.py covers the real maze.

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

import Coppelia.helpers as cop                                    # noqa: E402
from runner.constants import (                                    # noqa: E402
    RANGE_FIX_MAX_PREDICTED_M, SCAN_FIT_MIN_SAMPLES,
)
from runner.range_fix import is_trusted_reading, predict_range    # noqa: E402
from runner.scan_fit import (                                     # noqa: E402
    ScanSample, fit_scan, lateral_is_observable, longitudinal_is_observable,
)

CELL = 0.30
ORIGIN = (0.0, 0.0, 0.0)


class Corridor:
    """Straight east-west corridor, walls north and south, open both ends.

    The case longitudinal position is UNOBSERVABLE in: every position along the
    corridor presents the same profile of side walls.
    """
    rows, cols = 3, 12

    def in_bounds(self, c):
        return 0 <= c[0] < self.rows and 0 <= c[1] < self.cols

    def is_original_wall(self, c):
        return c[0] in (0, 2)


class DeadEnd(Corridor):
    """Same corridor, walled at column 5 -- a wall AHEAD pins the longitudinal
    axis, so the fit may report it."""

    def is_original_wall(self, c):
        return c[0] in (0, 2) or c[1] == 5


def center(cell):
    x, y, _ = cop.cell_to_world(ORIGIN, CELL, cell)
    return x, y


def sweep_samples(view, truth_xy, heading=0.0, span=60, step=15,
                  noise=None, quantise=False):
    """Ground-truth readings a sweep would collect from `truth_xy`.

    Readings are produced by the same raycast the fit predicts with, so a
    noise-free run isolates the fit's own behaviour from sensor modelling.
    """
    out = []
    for a in range(-span, span + 1, step):
        th = heading + math.radians(a)
        m, _hit = predict_range(view, ORIGIN, CELL, truth_xy[0], truth_xy[1],
                                th, RANGE_FIX_MAX_PREDICTED_M)
        if m is None:
            continue
        if noise is not None:
            m += noise(a)
        if quantise:
            m = round(m * 100) / 100.0        # whole centimetres, like the kart
        if not is_trusted_reading(m * 100.0):
            continue
        out.append(ScanSample(theta=th, measured=m))
    return out


class TestLateralRecovery(unittest.TestCase):
    """The DOF the sweep exists for."""

    def test_recovers_lateral_offset(self):
        view = Corridor()
        cx, cy = center((1, 4))
        for lat in (-0.05, -0.03, 0.0, 0.03, 0.05):
            samples = sweep_samples(view, (cx, cy + lat))
            fit = fit_scan(view, ORIGIN, CELL, cx, cy, 0.0, samples)
            self.assertIsNotNone(fit, f"no fit at lat={lat}")
            self.assertAlmostEqual(fit.lateral, lat, delta=0.006,
                                   msg=f"lat={lat} -> {fit.lateral}")

    def test_lateral_sign_is_left_of_heading(self):
        """+lateral means the kart is LEFT of where it believed, facing +X.
        Getting this backwards doubles the error, so it is pinned explicitly."""
        view = Corridor()
        cx, cy = center((1, 4))
        # World +Y is "north"/left when facing +X (see grid_heading_to_world_theta).
        samples = sweep_samples(view, (cx, cy + 0.04), heading=0.0)
        fit = fit_scan(view, ORIGIN, CELL, cx, cy, 0.0, samples)
        self.assertGreater(fit.lateral, 0.0)

    def test_lateral_recovered_under_a_rotated_heading(self):
        """The split is relative to the sweep's own heading, not the world axes.

        Driven facing NORTH, where "lateral" is the world X axis. The corridor
        is walled at one column so there is something to pin X against -- in a
        bare corridor, facing across it, lateral is the unobservable axis
        (see TestObservabilityGate.test_lateral_refused_facing_across_corridor),
        which is a different property and tested separately.
        """
        view = DeadEnd()
        cx, cy = center((1, 4))
        th = math.radians(90.0)
        lat = 0.03                       # left of north is -X
        tx = cx - math.sin(th) * lat
        ty = cy + math.cos(th) * lat
        samples = sweep_samples(view, (tx, ty), heading=th)
        self.assertGreaterEqual(len(samples), SCAN_FIT_MIN_SAMPLES)
        fit = fit_scan(view, ORIGIN, CELL, cx, cy, th, samples)
        self.assertIsNotNone(fit)
        self.assertIsNotNone(fit.lateral, "lateral should be observable here")
        self.assertAlmostEqual(fit.lateral, lat, delta=0.012)


class TestObservabilityGate(unittest.TestCase):
    """The half that keeps the feature from being dangerous."""

    def test_longitudinal_refused_in_open_corridor(self):
        """No wall ahead -> flat cost along the heading -> must report None.

        This is the measured failure the gate exists for: ungated, the search
        returns roughly -16 cm here against a true +2 cm.
        """
        view = Corridor()
        cx, cy = center((1, 4))
        samples = sweep_samples(view, (cx + 0.02, cy + 0.03))
        fit = fit_scan(view, ORIGIN, CELL, cx, cy, 0.0, samples)
        self.assertIsNotNone(fit)
        self.assertIsNone(fit.longitudinal)
        self.assertFalse(fit.longitudinal_observable)
        # ...while lateral, which IS observed, still comes through.
        self.assertAlmostEqual(fit.lateral, 0.03, delta=0.006)

    def test_longitudinal_reported_with_a_wall_ahead(self):
        view = DeadEnd()
        cx, cy = center((1, 3))
        samples = sweep_samples(view, (cx + 0.02, cy + 0.03))
        fit = fit_scan(view, ORIGIN, CELL, cx, cy, 0.0, samples)
        self.assertIsNotNone(fit)
        self.assertTrue(fit.longitudinal_observable)
        self.assertAlmostEqual(fit.longitudinal, 0.02, delta=0.01)
        self.assertAlmostEqual(fit.lateral, 0.03, delta=0.006)

    def test_lateral_refused_facing_across_corridor(self):
        """The case that is easy to miss: facing ACROSS a bare corridor, the
        beams go out through the open ends and nothing pins the kart sideways,
        so LATERAL is the blind axis this time.

        Ungated, the fit ran to the window edge and reported a confident 12 cm
        (= SCAN_FIT_MAX_SHIFT_M) against a true 4 cm. Gated, `lateral` is None
        and the offset is PROJECTED onto the surviving axis, so the correction
        the caller sees carries no component in the blind direction -- which is
        the property that matters, not whether a fit object exists.
        """
        view = Corridor()
        cx, cy = center((1, 4))
        th = math.radians(90.0)
        samples = sweep_samples(view, (cx - 0.04, cy), heading=th)
        self.assertGreaterEqual(len(samples), SCAN_FIT_MIN_SAMPLES,
                                "need samples for the point to hold")
        self.assertFalse(lateral_is_observable(
            view, ORIGIN, CELL, cx, cy, th, samples, 0.0, 0.0))

        fit = fit_scan(view, ORIGIN, CELL, cx, cy, th, samples)
        if fit is None:
            return                      # refusing outright is also correct
        self.assertIsNone(fit.lateral)
        self.assertFalse(fit.lateral_observable)
        # Nothing of the 12 cm artefact survives into the applied offset.
        self.assertLess(math.hypot(fit.dx, fit.dy), 0.005,
                        f"offset {fit.dx:+.3f},{fit.dy:+.3f} still carries the "
                        f"unobservable component")

    def test_gate_agrees_with_a_flat_cost_surface(self):
        """Direct statement of the underlying fact, independent of fit_scan:
        the corridor's cost does not rise along the heading, the dead end's
        does."""
        cx, cy = center((1, 4))
        corr = Corridor()
        self.assertFalse(longitudinal_is_observable(
            corr, ORIGIN, CELL, cx, cy, 0.0,
            sweep_samples(corr, (cx, cy)), 0.0, 0.0))

        dx, dy = center((1, 3))
        dead = DeadEnd()
        self.assertTrue(longitudinal_is_observable(
            dead, ORIGIN, CELL, dx, dy, 0.0,
            sweep_samples(dead, (dx, dy)), 0.0, 0.0))


class TestRefusals(unittest.TestCase):
    """Every failure path must leave the belief alone rather than guess."""

    def test_too_few_samples_returns_none(self):
        view = Corridor()
        cx, cy = center((1, 4))
        samples = sweep_samples(view, (cx, cy))[:SCAN_FIT_MIN_SAMPLES - 1]
        self.assertIsNone(fit_scan(view, ORIGIN, CELL, cx, cy, 0.0, samples))

    def test_no_samples_returns_none(self):
        view = Corridor()
        cx, cy = center((1, 4))
        self.assertIsNone(fit_scan(view, ORIGIN, CELL, cx, cy, 0.0, []))

    def test_reports_rms_improvement(self):
        """rms_after must actually beat rms_before when there was an error to
        find -- the caller rejects the fit on this, so it has to be real."""
        view = Corridor()
        cx, cy = center((1, 4))
        samples = sweep_samples(view, (cx, cy + 0.04))
        fit = fit_scan(view, ORIGIN, CELL, cx, cy, 0.0, samples)
        self.assertGreater(fit.rms_before, fit.rms_after)
        self.assertLess(fit.rms_after, 0.005)


class TestNoiseRobustness(unittest.TestCase):
    """With a 1 cm-noise, whole-centimetre sensor the lateral estimate must
    stay well inside the clearance it is protecting."""

    def test_lateral_survives_quantisation_and_noise(self):
        import random
        view = Corridor()
        cx, cy = center((1, 4))
        rng = random.Random(7)
        errs = []
        for _ in range(30):
            lat = rng.uniform(-0.05, 0.05)
            samples = sweep_samples(
                view, (cx, cy + lat),
                noise=lambda _a: rng.gauss(0.0, 0.01), quantise=True)
            if len(samples) < SCAN_FIT_MIN_SAMPLES:
                continue
            fit = fit_scan(view, ORIGIN, CELL, cx, cy, 0.0, samples)
            if fit is None:
                continue
            errs.append(abs(fit.lateral - lat))
        self.assertGreater(len(errs), 20, "too few fits to judge")
        errs.sort()
        median = errs[len(errs) // 2]
        p90 = errs[int(len(errs) * 0.9)]
        # Measured at ~0.3 cm median / ~0.8 cm p90; the bar is set well clear of
        # that so the test reports a real regression rather than noise.
        self.assertLess(median, 0.015, f"median {median*100:.2f} cm")
        self.assertLess(p90, 0.030, f"p90 {p90*100:.2f} cm")


if __name__ == "__main__":
    unittest.main()
