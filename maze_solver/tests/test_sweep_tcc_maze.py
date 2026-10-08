#!/usr/bin/python3
"""
The sweep and the fit, against the REAL maze (tcc.maze) rather than a stub.

Why this file exists separately: every stub maze in the other tests is a
corridor chosen to make one geometric point cleanly. tcc.maze is a 6x10 grid of
scattered pillars, which is a different and less forgiving problem -- sample
yield varies cell to cell, several cells see almost nothing, and the cells where
longitudinal is observable are not the ones a corridor would suggest. The
constants in runner/constants.py were tuned against exactly this file, so these
tests are what keeps that tuning honest.

What is asserted, and why these thresholds:

  * The fit is produced on a MINORITY of the solved path -- 7 of 17 cells -- and
    that is the honest answer, not a regression. Measured, the other ten sit in
    open stretches where most beams see nothing inside
    RANGE_FIX_MAX_PREDICTED_M and what they do see is past the 0.45 m trust
    ceiling or inside the sensor's 3 cm blind zone. Cell (5, 9), the goal, sees
    exactly two surfaces and both at 3-4 cm. Refusing there is correct: the
    alternative is shifting the belief on two blind-zone readings. A test
    demanding a majority would be a test demanding the gates be opened, so the
    bar is set at the measured yield and tightened only if the yield improves.
  * Where it fits, LATERAL is accurate -- that is the DOF the feature exists
    for, and no other corrector in the stack sees it.
  * No GROSS lateral error anywhere. A sweep that confidently reports 8 cm when
    the kart is 3 cm off is worse than no sweep at all, since the caller shifts
    the belief by it.
  * Lateral error stays well inside the clearance it protects: cells are 0.30 m
    so a wall is 0.15 m from centre, and the kart is ~0.14 m wide.

    cd maze_solver && python -m unittest discover -s tests
"""
import math
import os
import random
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
    SWEEP_HALF_SPAN_DEG, SWEEP_STEP_DEG,
)
from runner.range_fix import is_trusted_reading, predict_range     # noqa: E402
from runner.scan_fit import ScanSample, fit_scan                   # noqa: E402
from src.renders import MazeView                                   # noqa: E402
from src.solver import heading_between, solve_bfs                  # noqa: E402

CELL = 0.30                      # --cell-lenght default in runner/build_parser
ORIGIN = (0.0, 0.0, 0.0)
MAZE = os.path.join(REPO, "tcc.maze")


def load():
    view = MazeView(MAZE)
    path = solve_bfs(view, view.start, view.goal)
    return view, path


def center(cell):
    x, y, _ = cop.cell_to_world(ORIGIN, CELL, cell)
    return x, y


def leg_heading(path, i):
    if i + 1 < len(path):
        return math.radians(heading_between(path[i], path[i + 1]))
    return math.radians(heading_between(path[i - 1], path[i]))


def sweep_at(view, truth_xy, heading, rng=None, noise_cm=0.0):
    """Readings a sweep would collect, with the production span/step."""
    out = []
    span = int(SWEEP_HALF_SPAN_DEG)
    step = int(SWEEP_STEP_DEG)
    for a in range(-span, span + 1, step):
        th = heading + math.radians(a)
        m, _hit = predict_range(view, ORIGIN, CELL, truth_xy[0], truth_xy[1],
                                th, RANGE_FIX_MAX_PREDICTED_M)
        if m is None:
            continue
        if rng is not None and noise_cm:
            m += rng.gauss(0.0, noise_cm / 100.0)
        m = round(m * 100) / 100.0            # whole cm, like the kart reports
        if not is_trusted_reading(m * 100.0):
            continue
        out.append(ScanSample(theta=th, measured=m))
    return out


class TestTccMazeLoads(unittest.TestCase):

    def test_geometry_is_what_the_constants_were_tuned_against(self):
        """If the maze file changes, the tuned constants deserve another look --
        so the shape it was tuned against is pinned here."""
        view, path = load()
        self.assertEqual((view.rows, view.cols), (6, 10))
        self.assertEqual(view.start, (2, 0))
        self.assertEqual(view.goal, (5, 9))
        self.assertTrue(path, "tcc.maze has no solution")
        self.assertEqual(path[0], view.start)
        self.assertEqual(path[-1], view.goal)


class TestFitOverTheSolvedPath(unittest.TestCase):

    def _walk(self, lat, lng, noise_cm=0.0, seed=0):
        """Fit at every cell of the solved path. Returns (errors, fits, skips)."""
        view, path = load()
        rng = random.Random(seed)
        errors = []
        skips = 0
        for i, cell in enumerate(path):
            th = leg_heading(path, i)
            cx, cy = center(cell)
            c, s = math.cos(th), math.sin(th)
            tx = cx + c * lng - s * lat
            ty = cy + s * lng + c * lat
            samples = sweep_at(view, (tx, ty), th, rng, noise_cm)
            if len(samples) < SCAN_FIT_MIN_SAMPLES:
                skips += 1
                continue
            fit = fit_scan(view, ORIGIN, CELL, cx, cy, th, samples)
            if fit is None or fit.lateral is None:
                skips += 1
                continue
            errors.append((cell, fit.lateral - lat))
        return errors, len(path), skips

    def test_fit_yield_matches_the_measured_geometry(self):
        """Pins the yield in BOTH directions.

        A drop means something broke -- a gate got stricter, the span narrowed,
        a sample source dried up. A rise means the geometry assumptions in
        constants.py improved and the recorded numbers there are now stale, so
        the tuning deserves re-measuring rather than silent drift.
        """
        errors, total, skips = self._walk(lat=0.035, lng=0.02)
        self.assertEqual(len(errors) + skips, total)
        self.assertGreaterEqual(len(errors), 7,
                                f"only {len(errors)}/{total} cells produced a "
                                f"fit ({skips} refused) -- yield regressed")
        self.assertLessEqual(len(errors), 10,
                             f"{len(errors)}/{total} cells now fit, up from the "
                             f"7 the constants were tuned against -- re-measure "
                             f"the numbers in constants.py")

    def test_lateral_is_accurate_where_it_fits(self):
        """Noise-free: the fit should be near-exact, bounded only by the 5 mm
        search resolution."""
        for lat in (-0.04, -0.02, 0.02, 0.04):
            errors, _total, _skips = self._walk(lat=lat, lng=0.015)
            self.assertTrue(errors, f"no fits at lat={lat}")
            worst = max(abs(e) for _c, e in errors)
            self.assertLess(worst, 0.012,
                            f"lat={lat}: worst error {worst*100:.1f} cm at "
                            f"{max(errors, key=lambda ce: abs(ce[1]))[0]}")

    def test_no_gross_errors_with_a_realistic_sensor(self):
        """1 cm noise on top of whole-centimetre quantisation. A confident wrong
        answer is the failure that matters here: the caller shifts the belief by
        whatever comes back."""
        worst_overall = 0.0
        all_errs = []
        for seed in range(6):
            errors, _total, _skips = self._walk(lat=0.03, lng=0.02,
                                                noise_cm=1.0, seed=seed)
            for cell, e in errors:
                all_errs.append(abs(e))
                if abs(e) > worst_overall:
                    worst_overall, worst_cell = abs(e), cell
        self.assertTrue(all_errs, "no fits at all under noise")
        self.assertLess(worst_overall, 0.030,
                        f"gross lateral error {worst_overall*100:.1f} cm "
                        f"at {worst_cell}")
        median = sorted(all_errs)[len(all_errs) // 2]
        self.assertLess(median, 0.015, f"median {median*100:.2f} cm")

    def test_error_stays_inside_the_clearance_it_protects(self):
        """The point of the whole feature: a cell is 0.30 m so a wall face is
        0.15 m from centre, and the kart is ~0.14 m wide -- about 8 cm of
        clearance each side. A correction whose own error approached that would
        be self-defeating."""
        clearance = CELL / 2.0 - 0.07
        errors, _total, _skips = self._walk(lat=0.04, lng=0.02,
                                            noise_cm=1.0, seed=11)
        self.assertTrue(errors)
        worst = max(abs(e) for _c, e in errors)
        self.assertLess(worst, clearance / 2.0,
                        f"worst {worst*100:.1f} cm vs {clearance*100:.1f} cm "
                        f"clearance")

    def test_zero_offset_is_not_invented(self):
        """A kart already on centre must not be told it is somewhere else."""
        errors, _total, _skips = self._walk(lat=0.0, lng=0.0)
        self.assertTrue(errors)
        worst = max(abs(e) for _c, e in errors)
        self.assertLess(worst, 0.010,
                        f"invented a {worst*100:.1f} cm offset on a centred kart")


class TestLongitudinalGateOnTheRealMaze(unittest.TestCase):

    def test_gate_allows_longitudinal_on_this_maze(self):
        """On tcc.maze the gate should PASS wherever a fit happens at all.

        Measured, and worth stating because it is the opposite of the corridor
        case: every cell here that yields enough samples has pillars close on
        more than one side, which pins both axes. So this maze exercises the
        allow path only -- the refuse path is a corridor property and is tested
        on a corridor (see test_scan_fit.TestObservabilityGate). A gate that
        started refusing here would mean the probe had become too strict to
        ever credit a real minimum.
        """
        view, path = load()
        observable = refused = 0
        for i, cell in enumerate(path):
            th = leg_heading(path, i)
            cx, cy = center(cell)
            samples = sweep_at(view, (cx + 0.02 * math.cos(th),
                                      cy + 0.02 * math.sin(th)), th)
            if len(samples) < SCAN_FIT_MIN_SAMPLES:
                continue
            fit = fit_scan(view, ORIGIN, CELL, cx, cy, th, samples)
            if fit is None:
                continue
            if fit.longitudinal_observable:
                observable += 1
            else:
                refused += 1
        self.assertGreater(observable, 0, "gate never allowed longitudinal")
        self.assertEqual(refused, 0,
                         f"gate refused longitudinal on {refused} cell(s) of "
                         f"tcc.maze, where the geometry does constrain it")

    def test_longitudinal_is_accurate_when_allowed(self):
        """When the gate does pass it, the number has to be worth having."""
        view, path = load()
        lng = 0.025
        checked = 0
        for i, cell in enumerate(path):
            th = leg_heading(path, i)
            cx, cy = center(cell)
            c, s = math.cos(th), math.sin(th)
            samples = sweep_at(view, (cx + c * lng, cy + s * lng), th)
            if len(samples) < SCAN_FIT_MIN_SAMPLES:
                continue
            fit = fit_scan(view, ORIGIN, CELL, cx, cy, th, samples)
            if fit is None or fit.longitudinal is None:
                continue
            checked += 1
            self.assertAlmostEqual(
                fit.longitudinal, lng, delta=0.015,
                msg=f"{cell}: long {fit.longitudinal*100:+.1f} cm vs "
                    f"{lng*100:+.1f} cm")
        self.assertGreater(checked, 0, "no cell exercised the longitudinal path")


if __name__ == "__main__":
    unittest.main()


class TestBeamConeSensitivity(unittest.TestCase):
    """The open risk, pinned as a measurement rather than left in a comment.

    Every other test here feeds the fit raycasts: one ray per sample, hitting
    exactly what it points at. A real HC-SR04 returns the nearest echo anywhere
    in a ~15 degree cone, while `scan_fit` predicts along the ray -- so the
    other tests measure an upper bound on hardware performance, not hardware
    performance.

    This does not assert the sweep works under a realistic cone. It asserts
    that the DEGRADATION is what constants.py says it is, so the number quoted
    there stays true and nobody ships the lateral correction believing the
    0.25 cm figure applies on a real kart.
    """

    @staticmethod
    def _cone_reading(view, tx, ty, th, rng, cone_deg, rays=7, noise_cm=1.0):
        """Nearest echo across the cone, which is what the sensor reports."""
        best = None
        for k in range(rays):
            off = math.radians(-cone_deg + 2 * cone_deg * k / (rays - 1))
            m, _hit = predict_range(view, ORIGIN, CELL, tx, ty, th + off,
                                    RANGE_FIX_MAX_PREDICTED_M)
            if m is not None and (best is None or m < best):
                best = m
        if best is None:
            return None
        m = round((best + rng.gauss(0.0, noise_cm / 100.0)) * 100) / 100.0
        if not is_trusted_reading(m * 100.0):
            return None
        return m

    def _study(self, cone_deg, seeds=6):
        view, path = load()
        span, step = int(SWEEP_HALF_SPAN_DEG), int(SWEEP_STEP_DEG)
        errs = []
        gross = 0
        for seed in range(seeds):
            rng = random.Random(seed * 17 + 3)
            lat = rng.uniform(-0.04, 0.04)
            lng = rng.uniform(-0.03, 0.03)
            for i, cell in enumerate(path):
                th = leg_heading(path, i)
                cx, cy = center(cell)
                c, s = math.cos(th), math.sin(th)
                tx, ty = cx + c * lng - s * lat, cy + s * lng + c * lat
                samples = []
                for a in range(-span, span + 1, step):
                    t = th + math.radians(a)
                    m = self._cone_reading(view, tx, ty, t, rng, cone_deg)
                    if m is not None:
                        samples.append(ScanSample(theta=t, measured=m))
                if len(samples) < SCAN_FIT_MIN_SAMPLES:
                    continue
                fit = fit_scan(view, ORIGIN, CELL, cx, cy, th, samples)
                if fit is None or fit.lateral is None:
                    continue
                e = abs(fit.lateral - lat)
                errs.append(e)
                if e > 0.03:
                    gross += 1
        return errs, gross

    def test_narrow_cone_is_accurate(self):
        """A tightly-beamed sensor (or a cone-aware forward model) would make
        this feature work as the other tests suggest."""
        errs, gross = self._study(cone_deg=2.0)
        self.assertTrue(errs)
        self.assertLess(gross / len(errs), 0.10,
                        f"{gross}/{len(errs)} gross at a 2 deg cone")

    def test_real_cone_degrades_badly_and_that_is_recorded(self):
        """At the HC-SR04's real ~7.5 degree half-angle a large fraction of fits
        are grossly wrong. Asserted as a RANGE: if it improves, the forward
        model changed and constants.py's table needs updating; if it worsens,
        something regressed.
        """
        errs, gross = self._study(cone_deg=7.5)
        self.assertTrue(errs)
        rate = gross / len(errs)
        self.assertGreater(rate, 0.15,
                           f"only {rate:.0%} gross at the real cone width -- "
                           f"better than the 37% recorded in constants.py; "
                           f"re-measure and update that table")
        self.assertLess(rate, 0.60,
                        f"{rate:.0%} gross is worse than the 37% recorded")

    def test_degradation_is_monotonic_in_cone_width(self):
        """The mechanism claim: it is the cone, not the range ceiling or noise,
        that drives the error."""
        medians = []
        for cone in (0.0, 4.0, 7.5):
            errs, _gross = self._study(cone_deg=cone, seeds=4)
            self.assertTrue(errs, f"no fits at cone={cone}")
            errs.sort()
            medians.append(errs[len(errs) // 2])
        self.assertLess(medians[0], medians[1])
        self.assertLess(medians[1], medians[2])


class TestWallClearanceOverSuccessiveRotations(unittest.TestCase):
    """The requirement in its own terms: after many turns, does the kart still
    fit between the walls?

    Everything else in this file measures the FIT. This measures the
    consequence -- real position error accumulated over tcc.maze's whole solved
    path, with a heading error injected on every rotation, which is the error
    the kart's proprioception cannot see.

    Clearance: cells are 0.30 m so a wall face is 0.15 m from centre, and the
    kart is about 0.14 m wide, leaving ~0.08 m each side. CLEARANCE below is
    that budget; exceeding it means contact.

    The simulation is deliberately crude -- legs as straight moves, the sweep's
    correction applied to the belief -- because the quantity of interest is how
    error COMPOUNDS across legs, not the fidelity of any single one.
    """

    CLEARANCE = CELL / 2.0 - 0.07

    def _drive(self, yaw_err_deg, sweep_enabled, seed):
        """One run of the path. Returns the worst real position error."""
        view, path = load()
        rng = random.Random(seed)
        tx, ty = center(path[0])          # truth
        bx, by = tx, ty                   # belief
        worst = 0.0
        for i in range(len(path) - 1):
            target = center(path[i + 1])
            # goto: aim from the BELIEF at the target, drive that distance.
            head = math.atan2(target[1] - by, target[0] - bx)
            dist = math.hypot(target[0] - bx, target[1] - by)
            # The rotation lands off by this much, and odometry does not know:
            # it records the commanded heading. This is the error goto is blind
            # to, by construction.
            true_head = head + math.radians(rng.gauss(0.0, yaw_err_deg))
            tx += math.cos(true_head) * dist
            ty += math.sin(true_head) * dist
            bx += math.cos(head) * dist
            by += math.sin(head) * dist
            worst = max(worst, math.hypot(tx - target[0], ty - target[1]))

            if not sweep_enabled:
                continue
            th = leg_heading(path, min(i + 1, len(path) - 2))
            samples = sweep_at(view, (tx, ty), th)      # measured at the TRUTH
            if len(samples) < SCAN_FIT_MIN_SAMPLES:
                continue
            fit = fit_scan(view, ORIGIN, CELL, bx, by, th, samples)
            if fit is None or fit.lateral is None:
                continue
            c, s = math.cos(th), math.sin(th)
            along = fit.longitudinal or 0.0
            bx += c * along - s * fit.lateral
            by += s * along + c * fit.lateral
        return worst

    def test_goto_alone_does_not_survive_heading_error(self):
        """Pins the negative result, because it is the reason the sweep exists.

        The absolute goto bounds error that ODOMETRY can see. A heading error is
        not that: at 3 deg per rotation the kart ends up outside its clearance
        with no sweep, however good the goto is.
        """
        worst = max(self._drive(3.0, sweep_enabled=False, seed=s)
                    for s in range(8))
        self.assertGreater(worst, self.CLEARANCE,
                           f"worst {worst*100:.1f} cm -- expected the unswept "
                           f"run to exceed {self.CLEARANCE*100:.1f} cm of "
                           f"clearance at 3 deg/turn")

    def test_sweep_keeps_the_kart_clear_up_to_five_degrees(self):
        """The requirement. Up to 5 deg of per-rotation heading error, the kart
        stays inside its clearance for the whole 17-cell path."""
        for yaw_err in (1.0, 2.0, 3.0, 5.0):
            worst = max(self._drive(yaw_err, sweep_enabled=True, seed=s)
                        for s in range(8))
            self.assertLess(worst, self.CLEARANCE,
                            f"{yaw_err} deg/turn: worst {worst*100:.1f} cm "
                            f"exceeds {self.CLEARANCE*100:.1f} cm clearance")

    def test_sweep_beats_no_sweep_at_every_error_level(self):
        """Monotone benefit, so a regression that quietly disables the sweep
        shows up as a number rather than as a crash."""
        for yaw_err in (2.0, 3.0, 5.0):
            with_sweep = max(self._drive(yaw_err, True, s) for s in range(8))
            without = max(self._drive(yaw_err, False, s) for s in range(8))
            self.assertLess(with_sweep, without,
                            f"{yaw_err} deg/turn: sweep {with_sweep*100:.1f} cm "
                            f"vs unswept {without*100:.1f} cm")

    def test_eight_degrees_still_exceeds_clearance(self):
        """The honest limit. At 8 deg per rotation even per-leg sweeping does
        not keep the kart clear (11.8 cm measured), so the sweep is not a
        licence to leave the turns uncalibrated -- it buys roughly 5 deg of
        tolerance, not immunity.
        """
        worst = max(self._drive(8.0, sweep_enabled=True, seed=s)
                    for s in range(8))
        self.assertGreater(worst, self.CLEARANCE,
                           f"8 deg/turn now stays clear at {worst*100:.1f} cm "
                           f"-- better than recorded; re-measure the table in "
                           f"constants.py (SWEEP_EVERY_N_LEGS)")
