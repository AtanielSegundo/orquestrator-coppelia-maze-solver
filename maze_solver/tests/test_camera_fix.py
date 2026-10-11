#!/usr/bin/python3
"""
The camera fix SESSION (runner/camera_fix.py) and the pieces around it: the
self-supervised wall/floor colour model, the after-fix decision table, and the
centring move.

The session is driven by fakes -- a head that records commands, a stream whose
frames are rendered from a true pose at whatever pan the head was last sent to,
a fake clock -- so the things that only go wrong on hardware are pinned:

  * every view is approached from the same side (preposition below the lowest
    pan), and the head ends on the home pan at the held fix tilt;
  * a frame decoded before the head settled is never used;
  * the kart moving during the capture aborts the session;
  * the fit runs off the main thread: `poll()` itself stays cheap.

    cd maze_solver && python -m unittest discover -s tests
"""
import math
import os
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera_fixtures import (CALIB, CELL, GEOMS, ORIGIN, Corridor, Pillars,  # noqa: E402
                             offset_pose, synth_frame)

from camera_link.floor_segmenter import FloorSegmenter                     # noqa: E402
from camera_link.pinhole_view import PinholeView                           # noqa: E402
from runner.arrival_policy import (MEASURE, VERIFY, NOTHING,               # noqa: E402
                                   plan_after_camera)
from runner.camera_fit import CONFIRMED, CORRECTED, REFUSED, CameraFit     # noqa: E402
from runner.camera_fix import CameraFix                                    # noqa: E402
from runner.center_move import CenterMove, cell_behind                     # noqa: E402
from runner.reference_renderer import FLOOR, WALL, render_labels          # noqa: E402
from runner.wall_world import build_wall_world                             # noqa: E402
from shadow import SimMirror                                               # noqa: E402


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class Head:
    PAN_CHANNEL = 5
    TILT_CHANNEL = 1

    def __init__(self):
        self.pan, self.tilt = CALIB.pan_home_pct, 37
        self.history = []

    def set_percents(self, pan, tilt):
        self.pan, self.tilt = int(pan), int(tilt)
        self.history.append((self.pan, self.tilt))


class Stream:
    """Renders the true scene at the head's current pan. Every call is a new
    frame, stamped with the fake clock -- minus `lag`, to model frames taken
    before they arrive."""

    def __init__(self, head, clock, world, truth, pan_err=None, lag=0.0):
        self.head, self.clock, self.world, self.truth = head, clock, world, truth
        self.pan_err = pan_err or {}
        self.lag = lag
        self.count = 0
        self.rng = np.random.default_rng(0)
        self._cache = {}

    def latest_stamped(self):
        self.count += 1
        pan_deg = (self.head.pan - CALIB.pan_home_pct) * CALIB.deg_per_pct
        view = min(CALIB.views, key=lambda v: abs(v.pan_deg - pan_deg))
        key = (self.head.pan, self.head.tilt)
        if key not in self._cache:
            pin = PinholeView.from_calibration(
                CALIB, view, extra_pan_deg=pan_deg - view.pan_deg + self.pan_err.get(view.name, 0.0))
            self._cache[key] = render_labels(pin, self.truth, self.world)
        return synth_frame(self._cache[key], self.rng), self.count, self.clock() - self.lag


class Link:
    def servo_telemetry(self):
        return {}


def run_session(maze, cell, heading, a, b, yaw, moved_at=None, pan_err=None):
    world = build_wall_world(maze, ORIGIN, CELL)
    truth = offset_pose(cell, heading, a, b, yaw)
    believed = offset_pose(cell, heading, 0, 0, 0)
    clock, head = Clock(), Head()
    stream = Stream(head, clock, world, truth, pan_err)
    fix = CameraFix(CALIB, GEOMS, head, stream, Link(), world, odom_ref=(0, 0, 0),
                    clock=clock)
    odom = (0.0, 0.0, 0.0)
    worst = 0.0
    for i in range(20000):
        if moved_at is not None and i == moved_at:
            odom = (0.02, 0.0, 0.0)
        t0 = time.perf_counter()
        out = fix.poll({"goal_active": False}, odom, *believed)
        worst = max(worst, time.perf_counter() - t0)
        if out != "pending":
            return fix, head, out, worst
        clock.t += 1 / 60
        if fix._phase == "processing":
            time.sleep(0.002)               # let the worker run
    raise AssertionError("sessao nao terminou")


class TestCameraFixSession(unittest.TestCase):

    def test_measures_the_offset_and_restores_the_head(self):
        fix, head, out, _ = run_session(Corridor(), (1, 2), 0.0, 0.01, 0.03, 1.5,
                                        pan_err={"direita": 0.5, "esquerda": -0.5})
        self.assertEqual(out, "done")
        res = fix.result
        self.assertEqual(res.outcome, CORRECTED, res.reason)
        self.assertAlmostEqual(res.fit.lateral, 0.03, delta=0.005)
        self.assertEqual(head.history[-1], (CALIB.pan_home_pct, CALIB.tilt_pct))
        self.assertEqual(res.capture_rel, (0.0, 0.0, 0.0))

    def test_views_are_approached_from_one_side(self):
        _fix, head, _out, _ = run_session(Corridor(), (1, 2), 0.0, 0.0, 0.02, 0.0)
        pans = [p for p, _t in head.history]
        lowest = min(CALIB.pan_pct(v) for v in CALIB.views)
        self.assertLess(pans[0], lowest)                      # preposition first
        views = pans[1:-1]
        self.assertEqual(views, sorted(views))                # always increasing
        self.assertTrue(all(t == CALIB.tilt_pct for _p, t in head.history))

    def test_kart_moving_during_capture_aborts(self):
        fix, head, out, _ = run_session(Corridor(), (1, 2), 0.0, 0.0, 0.02, 0.0,
                                        moved_at=40)
        self.assertEqual(out, "refused")
        self.assertIn("mexeu", fix.result.reason)
        self.assertEqual(head.history[-1], (CALIB.pan_home_pct, CALIB.tilt_pct))

    def test_poll_stays_cheap(self):
        """The heavy work is the worker's. The stream fake renders frames on
        the main thread, so this bounds session + fake together, loosely."""
        _fix, _head, _out, worst = run_session(Pillars(), (2, 2), 0.0, 0.0, 0.02, 0.0)
        self.assertLess(worst, 0.25)

    def test_stale_frames_are_not_used(self):
        """A stream whose frames arrive stamped before the head settled never
        satisfies the freshness rule: the view times out instead."""
        world = build_wall_world(Corridor(), ORIGIN, CELL)
        believed = offset_pose((1, 2), 0.0, 0, 0, 0)
        clock, head = Clock(), Head()
        stream = Stream(head, clock, world, believed, lag=10.0)
        fix = CameraFix(CALIB, GEOMS, head, stream, Link(), world,
                        odom_ref=(0, 0, 0), clock=clock)
        for _ in range(5000):
            out = fix.poll({"goal_active": False}, (0, 0, 0), *believed)
            if out != "pending":
                break
            clock.t += 1 / 60
            if fix._phase == "processing":
                time.sleep(0.002)
        self.assertEqual(fix.result.outcome, REFUSED)
        self.assertIn("sem frames", fix.result.reason)


class TestFloorSegmenter(unittest.TestCase):

    def test_learns_walls_and_floor_from_predicted_labels(self):
        world = build_wall_world(Corridor(), ORIGIN, CELL)
        pose = offset_pose((1, 2), 0.0, 0, 0, 0)
        pin = GEOMS["esquerda"].pinhole
        labels = render_labels(pin, pose, world)
        rng = np.random.default_rng(2)
        frame = synth_frame(labels, rng)
        # Train on labels from a pose 2 cm off: the margin must absorb it.
        wrong = render_labels(pin, offset_pose((1, 2), 0.0, 0, 0.02, 0), world)
        seg = FloorSegmenter()
        model, why = seg.train(frame, wrong)
        self.assertIsNotNone(model, why)
        wall = FloorSegmenter.wall_mask(seg.llr(frame, model))
        body = np.zeros_like(wall)
        body[-15:, :] = True
        known = (labels != 2) & ~body
        acc = np.mean(wall[known] == (labels[known] == WALL))
        self.assertGreater(acc, 0.97)

    def test_refuses_when_wall_and_floor_look_alike(self):
        world = build_wall_world(Corridor(), ORIGIN, CELL)
        pin = GEOMS["esquerda"].pinhole
        labels = render_labels(pin, offset_pose((1, 2), 0.0, 0, 0, 0), world)
        rng = np.random.default_rng(3)
        frame = synth_frame(np.where(labels == WALL, FLOOR, labels), rng)
        model, why = FloorSegmenter().train(frame, labels)
        self.assertIsNone(model)
        self.assertIn("parecidas", why)


def _fit(outcome=CORRECTED, lat=0.04, lng=0.01, yaw=0.02):
    return CameraFit(outcome, "", lng, lat, yaw, (lng or 0, lat or 0, yaw or 0),
                     (0.001, 0.001, 0.001), {}, 1.0, 10.0, 1.0, 100, 100, 1.0,
                     0.5, ("frente",))


class TestArrivalPolicy(unittest.TestCase):

    def test_refused_or_confirmed_apply_nothing(self):
        self.assertEqual(plan_after_camera(None, 0.0, MEASURE, 0, False), NOTHING)
        self.assertEqual(plan_after_camera(_fit(REFUSED), 0.05, MEASURE, 90, False), NOTHING)
        self.assertEqual(plan_after_camera(_fit(CONFIRMED), 0.05, MEASURE, 90, False), NOTHING)

    def test_longitudinal_waits_for_recenter_unless_the_kart_moves(self):
        d = plan_after_camera(_fit(), 0.01, MEASURE, 0, False, policy="always")
        self.assertTrue(d.apply_lat and d.apply_yaw)
        self.assertFalse(d.move)
        self.assertFalse(d.apply_long_now)
        self.assertTrue(d.stash_long)
        d = plan_after_camera(_fit(), 0.05, MEASURE, 0, False, policy="always")
        self.assertTrue(d.move and d.apply_long_now and not d.stash_long)

    def test_before_turns_policy(self):
        straight = plan_after_camera(_fit(), 0.05, MEASURE, 0, False, policy="before_turns")
        turning = plan_after_camera(_fit(), 0.05, MEASURE, 90, False, policy="before_turns")
        at_goal = plan_after_camera(_fit(), 0.05, MEASURE, 0, True, policy="before_turns")
        self.assertFalse(straight.move)
        self.assertTrue(turning.move)
        self.assertTrue(at_goal.move)

    def test_verify_pass_never_moves(self):
        d = plan_after_camera(_fit(), 0.06, VERIFY, 90, True, policy="always")
        self.assertFalse(d.move)
        self.assertTrue(d.apply_lat)


class FakeKart:
    def __init__(self):
        self.calls = []
        self.goal_active = False

    def drive_distance(self, d, speed=0.2):
        self.calls.append(("drive", d))

    def goto(self, x, y, th=None):
        self.calls.append(("goto", x, y, th))


class TestCenterMove(unittest.TestCase):

    def _mirror(self):
        m = SimMirror.__new__(SimMirror)
        m.initial = (0.0, 0.0, 0.0)
        return m

    def _drive(self, mv, kart, clock, pose_after):
        tel = {"goal_active": False}
        for _ in range(500):
            out = mv.poll(tel, *pose_after)
            if out != "pending":
                return out
            clock.t += 0.05
        raise AssertionError("nao terminou")

    def test_cell_behind(self):
        self.assertEqual(cell_behind((1, 2), 0.0), (1, 1))
        self.assertEqual(cell_behind((1, 2), math.pi / 2), (2, 2))
        self.assertEqual(cell_behind((1, 2), math.pi), (1, 3))
        self.assertEqual(cell_behind((1, 2), -math.pi / 2), (0, 2))

    def test_backs_off_then_goes_to_the_centre(self):
        clock, kart = Clock(), FakeKart()
        cx, cy = (0.6, -0.3)
        mv = CenterMove(kart, self._mirror(), (0, 0, 0), Corridor(), ORIGIN, CELL,
                        (1, 2), 0.0, mode="backoff", clock=clock)
        out = self._drive(mv, kart, clock, (cx + 0.005, cy))
        self.assertEqual(out, "done")
        self.assertEqual(kart.calls[0][0], "drive")
        self.assertLess(kart.calls[0][1], 0.0)
        self.assertEqual(kart.calls[1][0], "goto")
        self.assertAlmostEqual(kart.calls[1][1], cx)
        self.assertAlmostEqual(kart.calls[1][2], cy)

    def test_no_room_behind_falls_back_to_a_plain_goto(self):
        clock, kart = Clock(), FakeKart()
        # Facing +Y (north) in the corridor: the cell behind is row 2, a wall.
        mv = CenterMove(kart, self._mirror(), (0, 0, 0), Corridor(), ORIGIN, CELL,
                        (1, 2), math.pi / 2, mode="backoff", clock=clock)
        self.assertEqual(mv.mode, "goto")
        self._drive(mv, kart, clock, (0.6, -0.3))
        self.assertEqual(kart.calls[0][0], "goto")

    def test_stopping_short_is_blocked(self):
        clock, kart = Clock(), FakeKart()
        mv = CenterMove(kart, self._mirror(), (0, 0, 0), Corridor(), ORIGIN, CELL,
                        (1, 2), 0.0, mode="goto", clock=clock)
        self.assertEqual(self._drive(mv, kart, clock, (0.6 - 0.06, -0.3)), "blocked")


if __name__ == "__main__":
    unittest.main()
