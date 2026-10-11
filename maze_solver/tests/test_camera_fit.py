#!/usr/bin/python3
"""
The camera pose fit (runner/camera_fit.py), on synthetic measurements rendered
per pixel by the reference ray caster at a perturbed TRUE pose -- side views
with a real pan error the fit has to absorb, contact rows with noise.

What is asserted, and why these thresholds:

  * Lateral is recovered to millimetres wherever the side walls are in view.
    It is the axis the feature exists for; the ultrasonic sweep it replaces
    measured 0.25 cm median under ideal rays and 2.5 cm under the real cone.
  * Longitudinal is reported as None in an open corridor -- the side walls
    look identical along it -- instead of a number the search happened to
    land on (the same trap scan_fit's observability gate exists for).
  * A view of nothing is refused, not fitted.
  * Over tcc.maze's solved path, measured with this setup: 14 of 17 cells give
    a lateral (the sweep: 7 of 17), median error 1.1 mm, max 2.1 mm, with 1 deg
    servo pan errors and 0.7 px row noise. The bars below leave room for that
    without letting a gross error (> 1 cm) through.

    cd maze_solver && python -m unittest discover -s tests
"""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera_fixtures import (CELL, ORIGIN, TCC, Corridor, OpenField,       # noqa: E402
                             Pillars, measure, offset_pose)

from runner.camera_fit import (CONFIRMED, CORRECTED, REFUSED,             # noqa: E402
                               FitParams, fit_camera_pose)
from runner.wall_world import build_wall_world                           # noqa: E402
from src.renders import MazeView                                         # noqa: E402
from src.solver import heading_between, solve_bfs                        # noqa: E402

PARAMS = FitParams(yield_s=0.0)


def run(maze, cell, heading, a, b, yaw, pan_err=None, noise=0.0, seed=0):
    world = build_wall_world(maze, ORIGIN, CELL)
    truth = offset_pose(cell, heading, a, b, yaw)
    believed = offset_pose(cell, heading, 0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    meas = measure(world, truth, pan_err, noise, rng)
    return fit_camera_pose(meas, world, believed, PARAMS)


class TestStubGeometry(unittest.TestCase):

    def test_corridor_recovers_lateral_and_yaw_not_longitudinal(self):
        fit = run(Corridor(), (1, 2), 0.0, 0.01, 0.03, 2.0,
                  pan_err={"direita": 0.6, "esquerda": -0.8}, noise=0.5)
        self.assertEqual(fit.outcome, CORRECTED, fit.reason)
        self.assertAlmostEqual(fit.lateral, 0.03, delta=0.003)
        self.assertIsNone(fit.longitudinal)
        self.assertIsNotNone(fit.yaw)
        self.assertAlmostEqual(math.degrees(fit.yaw), 2.0, delta=0.5)

    def test_pillar_field_recovers_all_three(self):
        fit = run(Pillars(), (2, 2), 0.0, -0.02, -0.04, -3.0,
                  pan_err={"direita": -0.4, "esquerda": 0.3}, noise=0.5)
        self.assertEqual(fit.outcome, CORRECTED, fit.reason)
        self.assertAlmostEqual(fit.lateral, -0.04, delta=0.003)
        self.assertAlmostEqual(fit.longitudinal, -0.02, delta=0.005)
        self.assertAlmostEqual(math.degrees(fit.yaw), -3.0, delta=0.5)

    def test_centred_kart_is_confirmed_not_corrected(self):
        fit = run(Pillars(), (2, 2), 0.0, 0.0, 0.0, 0.0)
        self.assertEqual(fit.outcome, CONFIRMED, fit.reason)
        self.assertLess(abs(fit.lateral), 0.008)

    def test_nothing_in_view_is_refused(self):
        fit = run(OpenField(), (3, 3), 0.0, 0.01, 0.02, 1.0)
        self.assertEqual(fit.outcome, REFUSED)
        self.assertIsNone(fit.lateral)

    def test_truth_outside_the_window_is_refused(self):
        """Further off than the +-10 cm search window is a lost pose. The best
        the search can do is sit on the window's edge with residuals the model
        does not explain -- both gates must refuse it rather than hand back the
        edge as if it were drift."""
        for maze, cell in ((Corridor(), (1, 3)), (Pillars(), (2, 2))):
            for b in (0.11, 0.13):
                fit = run(maze, cell, 0.0, 0.0, b, 0.0)
                self.assertEqual(fit.outcome, REFUSED, f"{type(maze).__name__} {b}")
                self.assertIsNone(fit.lateral)


class TestOverTccPath(unittest.TestCase):

    def test_yield_and_accuracy_over_the_solved_path(self):
        view = MazeView(TCC)
        path = solve_bfs(view, view.start, view.goal)
        world = build_wall_world(view, ORIGIN, CELL)
        rng = np.random.default_rng(3)
        errors, refused = [], 0
        for i, cell in enumerate(path):
            nxt = path[i + 1] if i + 1 < len(path) else None
            hd = (heading_between(cell, nxt) if nxt is not None
                  else heading_between(path[i - 1], cell))
            th = math.radians(hd)
            a, b, yaw = rng.uniform(-0.04, 0.04), rng.uniform(-0.05, 0.05), rng.uniform(-3, 3)
            truth = offset_pose(cell, th, a, b, yaw)
            pan_err = {"direita": rng.uniform(-1, 1), "esquerda": rng.uniform(-1, 1)}
            meas = measure(world, truth, pan_err, 0.7, rng)
            fit = fit_camera_pose(meas, world, offset_pose(cell, th, 0, 0, 0), PARAMS)
            if fit.lateral is None:
                refused += 1
                continue
            errors.append(abs(fit.lateral - b))
        self.assertGreaterEqual(len(errors), 12, f"so {len(errors)} celulas com lateral")
        self.assertLess(float(np.median(errors)), 0.004)
        self.assertLess(max(errors), 0.01, "erro lateral grosseiro aceito")


if __name__ == "__main__":
    unittest.main()
