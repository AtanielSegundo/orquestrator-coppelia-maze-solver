#!/usr/bin/python3
"""
The camera's forward model, against an independent one.

`contact_model` predicts each column's floor contact by raycasting the maze on
the FLOOR from the camera's foot -- exact only if (a) the per-sample bearings
are right when columns are not vertical planes (a pitched camera) and (b) the
2-D visibility argument holds at box corners. Both are checked here against
`reference_renderer`, a brute-force per-pixel 3-D ray caster, at the real
default calibration (25 deg down, +-45 deg side views) and off-centre poses.

Also pinned: the default calibration's conventions (where a pitched, panned
camera actually looks), the 2-D ray caster against a fine march, and the
profile extraction's column kinds.

    cd maze_solver && python -m unittest discover -s tests
"""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera_fixtures import (CALIB, CELL, GEOMS, ORIGIN, Corridor, Pillars,  # noqa: E402
                             offset_pose)

from camera_link.contact_profile import (FREE, HIT, INVALID, NEAR,
                                         extract_contact_profile, median_profile)
from camera_link.pinhole_view import PinholeView
from runner.camera_setup import nominal_focal_px
from runner.contact_model import UNKNOWN, interval_gaps, predict_at
from runner.floor_visibility import exit_distance, first_box_hits
from runner.reference_renderer import WALL, render_labels
from runner.wall_world import build_wall_world


class TestCalibrationConventions(unittest.TestCase):

    def test_nominal_focal_length_of_the_pi_camera_v13(self):
        # 3.6 mm over the 2560 px window of the 640x480 mode, 400 px wide.
        self.assertAlmostEqual(nominal_focal_px(400), 401.8, places=1)
        hfov = 2 * math.degrees(math.atan(200 / nominal_focal_px(400)))
        self.assertAlmostEqual(hfov, 52.9, delta=0.2)

    def test_forward_view_centre_ray_hits_the_floor_ahead(self):
        """Optical axis 25 deg down from 10 cm: floor 0.214 m ahead of the lens."""
        pin = PinholeView.from_calibration(CALIB, CALIB.view("frente"))
        q = pin.floor_points(pin.cx, pin.cy)
        self.assertAlmostEqual(q[0], CALIB.lens_xyz[0] + 0.10 / math.tan(math.radians(25)),
                               places=6)
        self.assertAlmostEqual(q[1], 0.0, places=9)

    def test_left_view_looks_left_and_moves_the_lens_left(self):
        pin = PinholeView.from_calibration(CALIB, CALIB.view("esquerda"))
        q = pin.floor_points(pin.cx, pin.cy)
        self.assertGreater(q[1], 0.05)                  # + y is left
        self.assertGreater(pin.C[1], 0.0)               # pan arm swings the lens
        right = PinholeView.from_calibration(CALIB, CALIB.view("direita"))
        self.assertAlmostEqual(right.C[1], -pin.C[1], places=12)

    def test_image_right_is_kart_right(self):
        pin = PinholeView.from_calibration(CALIB, CALIB.view("frente"))
        q = pin.floor_points(pin.width - 1, pin.cy)
        self.assertLess(q[1], 0.0)

    def test_projection_round_trips(self):
        pin = PinholeView.from_calibration(CALIB, CALIB.view("esquerda"))
        u, v = np.array([30.0, 200.0, 370.0]), np.array([250.0, 180.0, 120.0])
        q = pin.floor_points(u, v)
        uu, vv, z = pin.project(q)
        np.testing.assert_allclose(uu, u, atol=1e-9)
        np.testing.assert_allclose(vv, v, atol=1e-9)
        self.assertTrue(np.all(z > 0))


class TestFloorVisibility(unittest.TestCase):

    def test_slab_caster_matches_a_fine_march(self):
        boxes = np.array([[0.2, 0.5, -0.1, 0.1], [0.0, 0.3, 0.4, 0.6]])
        rng = np.random.default_rng(1)
        for _ in range(50):
            ox, oy = rng.uniform(-0.3, 0.0), rng.uniform(-0.3, 0.3)
            th = rng.uniform(-math.pi, math.pi)
            d = first_box_hits(ox, oy, math.cos(th), math.sin(th), boxes)
            t = np.arange(0, 2.0, 1e-4)
            px, py = ox + t * math.cos(th), oy + t * math.sin(th)
            inside = np.zeros_like(t, dtype=bool)
            for x0, x1, y0, y1 in boxes:
                inside |= (px >= x0) & (px <= x1) & (py >= y0) & (py <= y1)
            ref = t[np.argmax(inside)] if inside.any() else math.inf
            if math.isinf(ref):
                self.assertTrue(math.isinf(d))
            else:
                self.assertAlmostEqual(float(d), ref, delta=2e-4)

    def test_origin_inside_a_box_is_hit_at_zero(self):
        self.assertEqual(float(first_box_hits(0.0, 0.0, 1.0, 0.0,
                                              np.array([[-1, 1, -1, 1.0]]))), 0.0)

    def test_exit_distance(self):
        rect = (0.0, 1.0, 0.0, 2.0)
        self.assertAlmostEqual(float(exit_distance(0.5, 0.5, 1.0, 0.0, rect)), 0.5)
        self.assertAlmostEqual(float(exit_distance(0.5, 0.5, 0.0, -1.0, rect)), 0.5)
        self.assertEqual(float(exit_distance(2.0, 0.5, 1.0, 0.0, rect)), 0.0)


class TestContactModelAgainstPerPixelRaycast(unittest.TestCase):
    """The analytic intervals must reproduce what a per-pixel 3-D ray caster
    plus the profile extraction see -- kinds exactly, contact rows to well
    under a pixel (the extraction itself quantises to 0.5 px)."""

    CASES = [
        (Corridor(), (1, 1), 0.0, (0.02, -0.03, 3.0)),
        (Corridor(), (1, 4), 0.0, (-0.01, 0.02, -2.0)),
        (Pillars(), (2, 2), 0.0, (0.02, -0.03, 3.0)),
        (Pillars(), (2, 4), math.pi / 2, (0.0, 0.03, -4.0)),
        (Pillars(), (3, 5), math.pi, (-0.02, 0.0, 1.5)),
    ]

    def test_kinds_and_rows_agree(self):
        worst_median = 0.0
        for maze, cell, heading, (a, b, yaw) in self.CASES:
            world = build_wall_world(maze, ORIGIN, CELL)
            pose = offset_pose(cell, heading, a, b, yaw)
            for view in CALIB.views:
                g = GEOMS[view.name]
                lab = render_labels(g.pinhole, pose, world)
                prof = extract_contact_profile(lab == WALL, g.columns, g.v_near,
                                               g.v_far, 1)
                lo, hi, kind = predict_at(g, pose, world)
                usable = np.isfinite(g.v_far)
                # UNKNOWN (the floor left the maze first) is a promise about an
                # interval, not a kind: whatever the pixels show must lie at or
                # beyond the exit row.
                unknown = usable & (kind == UNKNOWN)
                m_lo, m_hi = prof.intervals()
                self.assertTrue(np.all(np.nan_to_num(
                    interval_gaps(m_lo[unknown], m_hi[unknown],
                                  lo[unknown], hi[unknown])) < 1.0))
                usable &= ~unknown
                differ = usable & (prof.kind != kind)
                # The only tolerated disagreement: a contact within a sample
                # step of the range limit, which one side samples on pixel rows
                # and the other on ROW_STEP rows. The interval residual turns
                # it into a gap of that same pixel or two, not a mismatch.
                at_limit = (np.abs(np.nan_to_num(lo, nan=-1e9) - g.v_far) < 3.0) \
                    | (np.abs(np.nan_to_num(prof.rows, nan=-1e9) - g.v_far) < 3.0)
                self.assertFalse(np.any(differ & ~at_limit),
                                 f"{type(maze).__name__} {cell} {view.name}: "
                                 f"{np.flatnonzero(differ & ~at_limit)}")
                both = (prof.kind == HIT) & (kind == HIT)
                if np.any(both):
                    d = np.abs(prof.rows[both] - lo[both])
                    worst_median = max(worst_median, float(np.median(d)))
                    # Corners: a column right at an occluding edge may differ
                    # by the bearing bin; everywhere else sub-pixel.
                    self.assertLess(np.percentile(d, 90), 1.0,
                                    f"{type(maze).__name__} {cell} {view.name}")
        self.assertLess(worst_median, 0.5)

    def test_floor_beyond_the_maze_edge_is_unknown_not_free(self):
        """tcc.maze has no perimeter: a border cell looks out into the room,
        where the map has nothing to say."""
        maze = Corridor()
        world = build_wall_world(type("Open", (Corridor,), {
            "_wall": lambda self, c: False})(), ORIGIN, CELL)
        pose = offset_pose((1, 10), 0.0, 0.0, 0.0, 0.0)
        _lo, _hi, kind = predict_at(GEOMS["frente"], pose, world)
        self.assertTrue(np.any(kind == UNKNOWN))
        self.assertFalse(np.any(kind == HIT))
        del maze


class TestContactProfile(unittest.TestCase):

    def _wall(self, rows_by_col, h=100, w=8):
        m = np.zeros((h, w), dtype=bool)
        for c, r in rows_by_col.items():
            if r is not None:
                m[:r + 1, c] = True          # wall from the top down to row r
        return m

    def test_kinds(self):
        wall = self._wall({0: 50, 1: 95, 2: None, 3: 10})
        prof = extract_contact_profile(wall, [0, 1, 2, 3], 90,
                                       np.array([20.0, 20.0, 20.0, 20.0]), 3)
        self.assertEqual(prof.kind[0], HIT)
        self.assertAlmostEqual(prof.rows[0], 50.5)
        self.assertEqual(prof.kind[1], NEAR)
        self.assertEqual(prof.kind[2], FREE)
        self.assertEqual(prof.kind[3], FREE)      # wall only beyond v_far

    def test_isolated_specks_do_not_end_the_column(self):
        wall = self._wall({0: 30})
        wall[70, 0] = wall[71, 0] = True          # 2-pixel speck on the floor
        prof = extract_contact_profile(wall, [0], 90, np.array([20.0]), 3)
        self.assertAlmostEqual(prof.rows[0], 30.5)

    def test_unusable_column_is_invalid(self):
        wall = self._wall({0: 30})
        prof = extract_contact_profile(wall, [0], 90, np.array([np.nan]), 3)
        self.assertEqual(prof.kind[0], INVALID)

    def test_median_of_frames_votes_and_medians(self):
        base = extract_contact_profile(self._wall({0: 40, 1: 40}), [0, 1], 90,
                                       np.array([20.0, 20.0]), 3)
        other = base._replace(rows=np.array([44.5, np.nan]),
                              kind=np.array([HIT, FREE], dtype=np.int8))
        third = base._replace(rows=np.array([42.5, 40.5]))
        m = median_profile([base, other, third])
        self.assertEqual(m.kind[0], HIT)
        self.assertAlmostEqual(m.rows[0], 42.5)
        self.assertEqual(m.kind[1], HIT)          # 2 of 3 voted HIT
        self.assertAlmostEqual(m.rows[1], 40.5)

    def test_interval_gaps(self):
        inf = np.inf
        g = interval_gaps(np.array([50.0, -inf, 90.0]), np.array([50.0, 20.0, inf]),
                          np.array([47.0, 25.0, -inf]), np.array([47.0, 25.0, 60.0]))
        np.testing.assert_allclose(g, [3.0, 5.0, 30.0])


if __name__ == "__main__":
    unittest.main()
