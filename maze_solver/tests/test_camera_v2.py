"""
Camera pose fix v2: the pan x tilt calibration model, the cell schedule, the
edge fit on synthetic frames, and the whole measure -> step loop against a
simulated kart whose turns are 3% off.

Frames come from the INDEPENDENT per-pixel ray caster (camera_fixtures), at
the kart's TRUE pose; the fix only knows the believed one.
"""
import math
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera_fixtures import CELL, ORIGIN, TCC, offset_pose, synth_frame  # noqa: E402

import numpy as np                                                       # noqa: E402

from camera_link.camera_calibration import CameraCalibration             # noqa: E402
from camera_link.pinhole_view import PinholeView                         # noqa: E402
from runner.camera_cells import camera_cell, find_camera_cell            # noqa: E402
from runner.camera_fix_v2 import (ACCEPTED, CORRECTED, CameraFixV2,      # noqa: E402
                                  PoseParams, make_segmenter, measure_view)
from runner.camera_setup import default_calibration, v2_views            # noqa: E402
from runner.edge_match import solve_edges                                # noqa: E402
from runner.reference_renderer import render_labels                      # noqa: E402
from runner.wall_world import build_wall_world                           # noqa: E402
from shadow import SimMirror, wrap                                       # noqa: E402
from src.renders import MazeView                                         # noqa: E402

VIEW = MazeView(TCC)
WORLD = build_wall_world(VIEW, ORIGIN, CELL)
CELL_OK, HEADING_OK = (3, 6), math.pi          # walls ahead and on the sides


def calib_v2():
    c = default_calibration()
    # Pitched up enough to see the tops of the boxes two cells ahead.
    c.tilt("cima").pitch_deg = -28.0
    return c


def real_views(calib, truth, believed, pan_err=0.0, seed=0):
    rng = np.random.default_rng(seed)
    seg = make_segmenter()
    reals = {}
    for v in v2_views(calib):
        pe = pan_err * {"direita": 1.0, "esquerda": -1.0}.get(v.pan_name, 0.0)
        pin = PinholeView.from_calibration(calib, v, extra_pan_deg=pe)
        frames = [synth_frame(render_labels(pin, truth, WORLD), rng) for _ in range(2)]
        real, _why, _img = measure_view(calib, v, frames, believed, WORLD, seg)
        if real is not None:
            reals[v.name] = real
    return reals


class TestTiltModel(unittest.TestCase):

    def test_views_are_pans_times_tilts_in_visiting_order(self):
        names = [v.name for v in v2_views(calib_v2())]
        self.assertEqual(names, ["frente_cima", "frente_baixo", "direita_cima",
                                 "direita_baixo", "esquerda_cima", "esquerda_baixo"])

    def test_held_tilt_is_the_v1_view(self):
        c = calib_v2()
        v1 = c.view("direita")
        R1, C1 = c.pose_center_cam(v1)
        R2, C2 = c.pose_center_cam(c.head_view(v1, "baixo"))
        np.testing.assert_allclose(R1, R2)
        np.testing.assert_allclose(C1, C2)

    def test_uncalibrated_tilt_is_derived_from_the_held_one(self):
        c = default_calibration()
        c.pitch_deg, c.tilt_pct, c.tilt_deg_per_pct = 11.0, 62, 0.5
        up = c.tilt("cima")
        self.assertAlmostEqual(c.tilt_pitch_deg(up), 11.0 - (62 - up.tilt_pct) * 0.5)

    def test_tilt_arm_moves_the_lens_only_away_from_the_held_tilt(self):
        c = calib_v2()
        c.tilt_arm_m = 0.02
        v = c.view("frente")
        _R, C_low = c.pose_center_cam(c.head_view(v, "baixo"))
        np.testing.assert_allclose(C_low, c.lens_xyz)
        _R, C_up = c.pose_center_cam(c.head_view(v, "cima"))
        self.assertGreater(C_up[2], c.lens_xyz[2])          # looking up raises it

    def test_save_load_keeps_tilts_and_plain_pan_views(self):
        c = calib_v2()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "calib.json")
            c.save(path)
            import json
            data = json.load(open(path))
            self.assertNotIn("tilt", data["views"][0])
            back = CameraCalibration.load(path, default_calibration())
        self.assertAlmostEqual(back.tilt_pitch_deg(back.tilt("cima")), -28.0)


class TestCameraCells(unittest.TestCase):

    def test_pocket_ahead_is_good_open_border_is_not(self):
        self.assertTrue(camera_cell(VIEW, CELL_OK, HEADING_OK).ok)
        q = camera_cell(VIEW, (2, 0), math.pi / 2)
        self.assertFalse(q.ok)
        self.assertEqual(q.front_obstacles, 1)

    def test_wall_right_ahead_is_refused(self):
        # (3, 5) facing 180 has the wall (3, 4) in the very next cell.
        self.assertIn("fechada", camera_cell(VIEW, (3, 5), math.pi).reason)

    def test_look_ahead_finds_the_first_good_cell(self):
        # Facing 180 along row 3: (3, 7) already has (2, 5) and (4, 5) two
        # cells ahead and walls on both sides of (3, 6).
        path = [(3, 8), (3, 7), (3, 6), (3, 5)]
        self.assertEqual(find_camera_cell(VIEW, path, 0, 8), 1)
        self.assertEqual(find_camera_cell(VIEW, path, 1, 8), 2)
        self.assertIsNone(find_camera_cell(VIEW, path, 0, 0))


class TestEdgeFit(unittest.TestCase):

    def _fit(self, yaw, along, lat, pan_err=0.0):
        calib = calib_v2()
        truth = offset_pose(CELL_OK, HEADING_OK, along, lat, yaw)
        believed = offset_pose(CELL_OK, HEADING_OK, 0.0, 0.0, 0.0)
        reals = real_views(calib, truth, believed, pan_err)
        self.assertEqual(len(reals), 6)
        par = PoseParams(calib, [v for v in v2_views(calib) if v.name in reals], believed)
        return solve_edges(reals, WORLD, par.build, par.x0, par.sigma, par.steps,
                           coarse=par.coarse)

    def test_recovers_yaw_and_longitudinal(self):
        fit = self._fit(2.0, 0.02, 0.005)
        self.assertTrue(fit.ok, fit.reason)
        self.assertAlmostEqual(math.degrees(fit.x[0]), 2.0, delta=0.3)
        self.assertAlmostEqual(fit.x[1], 0.02, delta=0.004)

    def test_three_degrees_the_other_way(self):
        fit = self._fit(-3.0, -0.015, -0.01)
        self.assertAlmostEqual(math.degrees(fit.x[0]), -3.0, delta=0.4)
        self.assertAlmostEqual(fit.x[1], -0.015, delta=0.004)

    def test_side_pan_error_is_not_read_as_yaw(self):
        fit = self._fit(-2.0, 0.01, 0.0, pan_err=1.5)
        self.assertAlmostEqual(math.degrees(fit.x[0]), -2.0, delta=0.3)

    def test_groups_split_the_centre_pan_into_half_planes(self):
        fit = self._fit(2.0, 0.0, 0.0)
        self.assertIn("frente_esq_baixo", fit.groups)
        self.assertIn("frente_dir_cima", fit.groups)
        # Yaw: both halves shift the same way.
        le, ri = fit.groups["frente_esq_baixo"], fit.groups["frente_dir_baixo"]
        self.assertGreater(le.shift[0] * ri.shift[0], 0.0)


# -- the whole loop against a simulated kart ----------------------------------
class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class SimKart:
    """Odometry follows the commands exactly; the body turns 3% more."""

    def __init__(self, truth):
        self.truth = list(truth)
        self.odom = [0.0, 0.0, 0.0]
        self.pulse = False

    def turn(self, angle_deg, speed=1.0):
        d = math.radians(angle_deg)
        self.odom[2] = wrap(self.odom[2] + d)
        self.truth[2] = wrap(self.truth[2] + d * 1.03)
        self.pulse = True

    def servo_telemetry(self):
        return {}


class Head:
    PAN_CHANNEL, TILT_CHANNEL = 5, 1

    def __init__(self, calib):
        self.pan, self.tilt = calib.pan_home_pct, calib.tilt_pct

    def set_percents(self, pan, tilt):
        self.pan, self.tilt = int(pan), int(tilt)


class Stream:
    def __init__(self, calib, kart, head, clock):
        self.calib, self.kart, self.head, self.clock = calib, kart, head, clock
        self.views = v2_views(calib)
        self.count = 0
        self.rng = np.random.default_rng(0)
        self._cache = {}

    def latest_stamped(self):
        self.count += 1
        key = (self.head.pan, self.head.tilt, tuple(round(v, 6) for v in self.kart.truth))
        if key not in self._cache:
            view = next((v for v in self.views
                         if self.calib.pan_pct(v) == self.head.pan
                         and self.calib.view_tilt(v)[0] == self.head.tilt),
                        self.calib.view("frente"))
            pin = PinholeView.from_calibration(self.calib, view)
            self._cache[key] = synth_frame(render_labels(pin, tuple(self.kart.truth), WORLD),
                                           self.rng)
        return self._cache[key], self.count, self.clock()


class TestMeasureStepLoop(unittest.TestCase):

    def test_turns_out_the_yaw_and_reanchors_the_longitudinal(self):
        calib = calib_v2()
        believed = offset_pose(CELL_OK, HEADING_OK, 0.0, 0.0, 0.0)
        kart = SimKart(offset_pose(CELL_OK, HEADING_OK, 0.015, 0.0, 2.5))
        clock = Clock()
        head = Head(calib)
        mirror = SimMirror.__new__(SimMirror)
        mirror.initial = believed
        ref = (0.0, 0.0, 0.0)

        def on_step(rel, capture_pose, a, psi):
            x0, y0, th0 = mirror.world_pose_of(rel)
            c, s = math.cos(capture_pose[2]), math.sin(capture_pose[2])
            mirror.reanchor(rel, (x0 + c * a, y0 + s * a, th0 + psi))

        fix = CameraFixV2(calib, head, Stream(calib, kart, head, clock), kart, WORLD,
                          ref, on_step=on_step, clock=clock)
        deadline = time.monotonic() + 240.0
        while time.monotonic() < deadline:
            active, kart.pulse = kart.pulse, False
            belief = mirror.world_pose(tuple(kart.odom), ref)
            if fix.poll({"goal_active": active}, tuple(kart.odom), *belief) != "pending":
                break
            clock.t += 0.05
            if fix._phase == "processing":
                time.sleep(0.01)
        res = fix.result
        self.assertIsNotNone(res, "nao terminou")
        report = res.summary()
        self.assertIn(res.outcome, (ACCEPTED, CORRECTED), report)
        belief = mirror.world_pose(tuple(kart.odom), ref)
        # The kart is back on the axis, and the belief knows where it is.
        self.assertLess(abs(math.degrees(wrap(kart.truth[2] - math.pi))), 0.7, report)
        self.assertLess(abs(math.degrees(wrap(belief[2] - kart.truth[2]))), 0.7, report)
        c, s = math.cos(math.pi), math.sin(math.pi)
        along = (belief[0] - kart.truth[0]) * c + (belief[1] - kart.truth[1]) * s
        self.assertLess(abs(along), 0.006, report)
        self.assertNotEqual(res.turned_deg, 0.0, report)


# -- scheduling inside the runner ---------------------------------------------
class Pocket:
    """Row 2 runs east; (2, 3) faces a pocket: walls (1, 4), (3, 4), (2, 5)."""
    rows, cols = 5, 8
    obstacles = ()
    walls = frozenset({(1, 4), (3, 4), (2, 5)})

    def in_bounds(self, c):
        return 0 <= c[0] < self.rows and 0 <= c[1] < self.cols

    def is_original_wall(self, c):
        return (not self.in_bounds(c)) or c in self.walls

    def is_wall(self, c):
        return self.is_original_wall(c)


class TestRunnerSchedule(unittest.TestCase):
    PATH = [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)]

    def runner(self):
        import runner.live_maze_runner as lmr
        r = lmr.LiveMazeRunner.__new__(lmr.LiveMazeRunner)
        r.view = Pocket()
        r.origin, r.cell_length = ORIGIN, CELL
        r.plan_path, r.plan_index = list(self.PATH), 1
        r.camera_calib = calib_v2()
        r.camera = Head(r.camera_calib)
        r.camera_stream = None
        r.link = SimKart((0.0, 0.0, 0.0))
        r.odom_ref = (0.0, 0.0, 0.0)
        r.turn_speed = 0.35
        r._cells_since_camera = 3
        return r

    def pose(self, cell):
        x, y = offset_pose(cell, 0.0, 0.0, 0.0, 0.0)[:2]
        return x, y, 0.0

    def test_not_due_does_nothing(self):
        from runner.arrival_policy import MEASURE
        r = self.runner()
        r._cells_since_camera = 0
        self.assertFalse(r._start_camera_v2(MEASURE, *self.pose((2, 1))))
        self.assertIsNone(r._camera_next)
        self.assertIsNone(r.camera_fix)

    def test_postponed_to_the_pocket_then_run_there(self):
        from runner.arrival_policy import MEASURE
        r = self.runner()
        self.assertFalse(r._start_camera_v2(MEASURE, *self.pose((2, 1))))
        self.assertEqual(r._camera_next, (3, (2, 3)))
        # On the way (index 2): still one cell at a time, nothing started.
        r.plan_index = 2
        self.assertFalse(r._start_camera_v2(MEASURE, *self.pose((2, 2))))
        self.assertEqual(r._camera_next, (3, (2, 3)))
        r.plan_index = 3
        self.assertTrue(r._start_camera_v2(MEASURE, *self.pose((2, 3))))
        self.assertIsInstance(r.camera_fix, CameraFixV2)
        self.assertIsNone(r._camera_next)

    def test_nothing_good_ahead_stays_due(self):
        from runner.arrival_policy import MEASURE
        r = self.runner()
        r.view.walls = frozenset()
        self.assertFalse(r._start_camera_v2(MEASURE, *self.pose((2, 1))))
        self.assertIsNone(r._camera_next)
        self.assertEqual(r._cells_since_camera, 3)


if __name__ == "__main__":
    unittest.main()
