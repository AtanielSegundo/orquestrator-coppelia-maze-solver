#!/usr/bin/python3
"""
Closed loop: the REAL LiveMazeRunner arrival chain (camera fix -> centring ->
Recenter, runner/live_maze_runner.py) driving a simulated kart over part of
tcc.maze, with the error the camera exists for injected:

    every turn the kart makes comes out 3% too large, and odometry does not see
    it -- the heading error that makes the absolute goto walk off sideways
    (see SWEEP_EVERY_N_LEGS in runner/constants.py).

The kart answers `goto` / `drive_distance` / `turn` like the server does (odom
lands exactly on the command, the body does not), reports the sonar range from
its TRUE pose, and the camera stream renders frames from its true pose at the
head's current pan.

What this pins, beyond the unit tests: the signs and frames of the whole chain.
A correction applied with the wrong sign, in the wrong frame, against the wrong
odometry sample, or through `shift_anchor` instead of `reanchor` does not fail
any single-module test -- it makes this kart end up FURTHER from the centres
than with no camera at all.

Slow-ish (tens of seconds): every arrival runs a real fit.

    cd maze_solver && python -m unittest discover -s tests
"""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera_fixtures import CALIB, CELL, GEOMS, ORIGIN, TCC, synth_frame  # noqa: E402

import Coppelia.helpers as cop                                           # noqa: E402
import runner.camera_fix as camera_fix_mod                               # noqa: E402
import runner.center_move as center_move_mod                             # noqa: E402
import runner.live_maze_runner as runner_mod                             # noqa: E402
import runner.recenter as recenter_mod                                   # noqa: E402
from camera_link.pinhole_view import PinholeView                         # noqa: E402
from runner.arrival_log import ArrivalLog                                # noqa: E402
from runner.live_maze_runner import LiveMazeRunner                       # noqa: E402
from runner.range_fix import RangeFix, predict_range                     # noqa: E402
from runner.reference_renderer import render_labels                      # noqa: E402
from runner.wall_world import build_wall_world                           # noqa: E402
from shadow import SimMirror, grid_heading_to_world_theta, wrap          # noqa: E402
from src.renders import MazeView                                         # noqa: E402
from src.solver import heading_between, solve_bfs                        # noqa: E402

TURN_ERROR = 0.03          # every real turn is 3% larger than commanded
GOAL = (1, 6)              # 9 cells from tcc.maze's start, three turns


class SimKart:
    """Odometry follows the commands exactly; the body turns 3% more."""

    def __init__(self, truth):
        self.truth = list(truth)
        self.odom = [0.0, 0.0, 0.0]
        self._pulse = False

    # -- commands (KartLink surface) -------------------------------------------
    def _turn(self, d):
        self.odom[2] = wrap(self.odom[2] + d)
        self.truth[2] = wrap(self.truth[2] + d * (1.0 + TURN_ERROR))

    def _drive(self, d):
        for pose in (self.odom, self.truth):
            pose[0] += d * math.cos(pose[2])
            pose[1] += d * math.sin(pose[2])

    def goto(self, x, y, theta_deg=None):
        dx, dy = x - self.odom[0], y - self.odom[1]
        dist = math.hypot(dx, dy)
        if dist >= 0.01:
            self._turn(wrap(math.atan2(dy, dx) - self.odom[2]))
            self._drive(math.hypot(x - self.odom[0], y - self.odom[1]))
        if theta_deg is not None:
            self._turn(wrap(math.radians(theta_deg) - self.odom[2]))
        self._pulse = True

    def drive_distance(self, d, speed=0.2):
        self._drive(d)
        self._pulse = True

    def turn(self, angle_deg, speed=1.0):
        self._turn(math.radians(angle_deg))
        self._pulse = True

    def servo_telemetry(self):
        return {}

    # -- telemetry -------------------------------------------------------------
    def telemetry(self, view):
        r, _ = predict_range(view, ORIGIN, CELL, *self.truth, 3.0)
        active, self._pulse = self._pulse, False
        return {"goal_active": active,
                "front_distance_cm": 300.0 if r is None else round(r * 100.0)}


class Head:
    PAN_CHANNEL, TILT_CHANNEL = 5, 1

    def __init__(self):
        self.pan, self.tilt = CALIB.pan_home_pct, CALIB.tilt_pct

    def set_percents(self, pan, tilt):
        self.pan, self.tilt = int(pan), int(tilt)


class Stream:
    def __init__(self, kart, head, world):
        self.kart, self.head, self.world = kart, head, world
        self.count = 0
        self.rng = np.random.default_rng(0)

    def latest_stamped(self):
        import time
        self.count += 1
        pan_deg = (self.head.pan - CALIB.pan_home_pct) * CALIB.deg_per_pct
        view = min(CALIB.views, key=lambda v: abs(v.pan_deg - pan_deg))
        pin = PinholeView.from_calibration(CALIB, view, extra_pan_deg=pan_deg - view.pan_deg)
        labels = render_labels(pin, tuple(self.kart.truth), self.world)
        return synth_frame(labels, self.rng), self.count, time.monotonic()


class FakeMirror(SimMirror):
    def __init__(self, initial):
        self.initial = tuple(initial)

    def apply_world_pose(self, wx, wy, wtheta):
        pass


def build_runner(view, path, start_pose, kart, camera: bool):
    r = LiveMazeRunner.__new__(LiveMazeRunner)
    world = build_wall_world(view, ORIGIN, CELL)
    r.view, r.origin, r.cell_length = view, ORIGIN, CELL
    r.range_fix = RangeFix(view, ORIGIN, CELL)
    r.link = kart
    r.mirror = FakeMirror(start_pose)
    r.odom_ref = (0.0, 0.0, 0.0)
    r.plan_path, r.plan_index, r.visited = list(path), 0, [path[0]]
    r.goal_cell = path[-1]
    r.leg = r.recenter = r.sweep = r.camera_fix = r.center_move = None
    r.state, r.phase = "running", "live"
    r.replan_requested, r.waypoint = False, None
    r.should_wait, r._step_released, r._waiting_to_step = False, False, False
    r.turn_speed = r.drive_speed = 0.2
    r.status = ""
    r.last_world_pose = None
    r._legs_since_lateral_fix, r.sweeps_done, r.last_sweep_summary = 0, 0, ""
    r.last_camera_summary, r.camera_fixes_done = "", 0
    r._camera_lateral_ok, r._camera_pending_long = False, None
    r._needs_start_fix, r._arrival = camera, None
    r.arrival_log = ArrivalLog("", enabled=False)
    r._update_kart_view = lambda *a: None
    head = Head()
    r.camera = head
    if camera:
        r.camera_calib, r.camera_geoms = CALIB, GEOMS
        r.camera_stream = Stream(kart, head, world)
    else:
        r.camera_calib = r.camera_geoms = r.camera_stream = None
    return r


class TestClosedLoop(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Real time is not what is being tested: settle and grace waits off.
        cls._saved = {}
        for mod, names in ((camera_fix_mod, ("CAMERA_FIX_KART_SETTLE_S",
                                             "CAMERA_FIX_SERVO_SETTLE_S",
                                             "CAMERA_FIX_FRAME_LATENCY_S")),
                           (recenter_mod, ("RECENTER_SETTLE_S",)),
                           (center_move_mod, ("RECENTER_SETTLE_S",)),
                           (runner_mod, ("SWEEP_EVERY_N_LEGS",))):
            for n in names:
                cls._saved[(mod, n)] = getattr(mod, n)
                setattr(mod, n, 99 if n == "SWEEP_EVERY_N_LEGS" else 0.0)
        cls._saved_dump = runner_mod.CAMERA_FIX_DEBUG_DUMP
        runner_mod.CAMERA_FIX_DEBUG_DUMP = False

    @classmethod
    def tearDownClass(cls):
        for (mod, n), v in cls._saved.items():
            setattr(mod, n, v)
        runner_mod.CAMERA_FIX_DEBUG_DUMP = cls._saved_dump

    def _drive(self, camera: bool):
        view = MazeView(TCC)
        path = solve_bfs(view, view.start, GOAL)
        start = path[0]
        sx, sy, _ = cop.cell_to_world(ORIGIN, CELL, start)
        sth = grid_heading_to_world_theta(heading_between(path[0], path[1]))
        kart = SimKart((sx, sy, sth))
        r = build_runner(view, path, (sx, sy, sth), kart, camera)
        departures = []          # true lateral offset when each leg starts
        original_leg = runner_mod.Leg

        def leg_spy(frm, *a, **kw):
            cx, cy, _ = cop.cell_to_world(ORIGIN, CELL, frm)
            th = kart.truth[2]
            ax = round(th / (math.pi / 2)) * (math.pi / 2)
            dx, dy = kart.truth[0] - cx, kart.truth[1] - cy
            departures.append(abs(-math.sin(ax) * dx + math.cos(ax) * dy))
            return original_leg(frm, *a, **kw)

        runner_mod.Leg = leg_spy
        try:
            for _ in range(200000):
                odom = tuple(kart.odom)
                wx, wy, wth = r.mirror.world_pose(odom, r.odom_ref)
                tel = kart.telemetry(view)
                r._advance(tel, odom, wx, wy, wth)
                if r.state == "idle":
                    break
                if r.camera_fix is not None and r.camera_fix._phase == "processing":
                    import time
                    time.sleep(0.002)
            else:
                self.fail("nao chegou ao goal")
        finally:
            runner_mod.Leg = original_leg
        cx, cy, _ = cop.cell_to_world(ORIGIN, CELL, GOAL)
        final = math.hypot(kart.truth[0] - cx, kart.truth[1] - cy)
        return departures, final, r

    def test_camera_keeps_the_kart_on_the_centre_line(self):
        dep_cam, final_cam, r = self._drive(camera=True)
        dep_ref, final_ref, _ = self._drive(camera=False)
        self.assertGreater(r.camera_fixes_done, 0)
        # Without the camera the turn error walks the kart sideways; with it
        # the offset each leg starts from stays small.
        self.assertLess(max(dep_cam), 0.02, f"com camera: {np.round(dep_cam, 3)}")
        self.assertLess(max(dep_cam), max(dep_ref),
                        f"com camera {np.round(dep_cam, 3)} vs sem {np.round(dep_ref, 3)}")
        self.assertLess(final_cam, final_ref)


if __name__ == "__main__":
    unittest.main()
