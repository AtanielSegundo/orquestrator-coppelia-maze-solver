#!/usr/bin/python3
"""
Closed loop: the REAL LiveMazeRunner boundary chain

    Recenter -> [closest obstacle reference, when due] -> camera

driving a simulated kart over part of tcc.maze, with the error the camera is
there for injected:

    every turn the kart makes comes out 3% too large, and odometry does not see
    it -- the heading error that makes the absolute goto walk off sideways
    (see SWEEP_EVERY_N_LEGS in runner/constants.py).

The kart answers `goto` / `drive_distance` / `turn` like the server does (odom
lands exactly on the command, the body does not), reports the sonar range from
its TRUE pose, and the camera stream renders frames from its true pose at the
head's current pan.

Three runs of the same path, same kart:

    none        no lateral corrector at all (Recenter only)
    sonar       CLOSEST_OBSTACLE_AS_REFERENCE -- the simple-features setup
    sonar+cam   the same, plus the camera's small corrections last

What this pins: the camera WORKS WITH the closest obstacle reference instead of
replacing it -- the reference still runs on its own schedule -- and adding it
leaves the kart closer to the centre line than the reference alone. Beyond the
unit tests, it also pins the signs and frames of the whole chain: a correction
applied with the wrong sign, in the wrong frame, or through `shift_anchor`
instead of `reanchor` would put the kart FURTHER off than without the camera.

Slow-ish (tens of seconds): every boundary runs a real fit.

    cd maze_solver && python -m unittest discover -s tests
"""
import math
import os
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera_fixtures import CALIB, CELL, GEOMS, ORIGIN, TCC, synth_frame  # noqa: E402

import Coppelia.helpers as cop                                           # noqa: E402
import runner.camera_fix as camera_fix_mod                               # noqa: E402
import runner.center_move as center_move_mod                             # noqa: E402
import runner.closest_obstacle as closest_mod                            # noqa: E402
import runner.live_maze_runner as runner_mod                             # noqa: E402
import runner.recenter as recenter_mod                                   # noqa: E402
from camera_link.pinhole_view import PinholeView                         # noqa: E402
from runner.arrival_log import ArrivalLog                                # noqa: E402
from runner.constants import LEG_MAX_CELLS                               # noqa: E402
from runner.live_maze_runner import LiveMazeRunner                       # noqa: E402
from runner.range_fix import RangeFix, predict_range                     # noqa: E402
from runner.reference_renderer import render_labels                      # noqa: E402
from runner.wall_world import build_wall_world                           # noqa: E402
from shadow import SimMirror, grid_heading_to_world_theta, wrap          # noqa: E402
from src.renders import MazeView                                         # noqa: E402
from src.solver import heading_between, solve_bfs                        # noqa: E402

TURN_ERROR = 0.03          # every real turn is 3% larger than commanded
GOAL = (1, 6)              # from tcc.maze's start: three turns, two straights


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
        if math.hypot(dx, dy) >= 0.01:
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

    def halt(self):
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
    r.view, r.origin, r.cell_length = view, ORIGIN, CELL
    r.range_fix = RangeFix(view, ORIGIN, CELL)
    r.link = kart
    r.mirror = FakeMirror(start_pose)
    r.odom_ref = (0.0, 0.0, 0.0)
    r.plan_path, r.plan_index, r.visited = list(path), 0, [path[0]]
    r.goal_cell = path[-1]
    r.leg = r.recenter = r.sweep = r.side_ref = None
    r.state, r.phase = "running", "live"
    r.replan_requested, r.waypoint = False, None
    r.should_wait, r._step_released, r._waiting_to_step = False, False, False
    r.turn_speed = r.drive_speed = 0.2
    r.status = ""
    r.last_world_pose = None
    r._cells_since_sweep, r.sweeps_done, r.last_sweep_summary = 0, 0, ""
    r._last_side_turn = 0
    r._next_best, r._side_ref_scheduled = None, False
    r._leg_max_cells = r._leg_max_cells_saved = LEG_MAX_CELLS
    r._suspect, r._dismissed = None, set()
    r._needs_start_fix = camera
    r.arrival_log = ArrivalLog("", enabled=False)
    r._update_kart_view = lambda *a: None
    head = Head()
    r.camera = head
    if camera:
        r.camera_calib, r.camera_geoms = CALIB, GEOMS
        r.camera_stream = Stream(kart, head, build_wall_world(view, ORIGIN, CELL))
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
                           (closest_mod, ("SWEEP_SETTLE_S",))):
            for n in names:
                cls._saved[(mod, n)] = getattr(mod, n)
                setattr(mod, n, 0.0)
        # This pins the v1 per-boundary fix; v2 has its own loop test
        # (tests/test_camera_v2.py) and a head this test's stream cannot tilt.
        for n, v in (("CAMERA_FIX_DEBUG_DUMP", False),
                     ("CLOSEST_OBSTACLE_AS_REFERENCE", True),
                     ("SWEEP_EVERY_N_LEGS", 1),
                     ("CAMERA_V2_ENABLED", False)):
            cls._saved[(runner_mod, n)] = getattr(runner_mod, n)
            setattr(runner_mod, n, v)

    @classmethod
    def tearDownClass(cls):
        for (mod, n), v in cls._saved.items():
            setattr(mod, n, v)

    def _drive(self, camera: bool, big: bool):
        saved = (runner_mod.CLOSEST_OBSTACLE_AS_REFERENCE, runner_mod.SWEEP_EVERY_N_LEGS)
        runner_mod.CLOSEST_OBSTACLE_AS_REFERENCE = big
        runner_mod.SWEEP_EVERY_N_LEGS = 1 if big else 10 ** 6    # no sweep either
        view = MazeView(TCC)
        path = solve_bfs(view, view.start, GOAL)
        sx, sy, _ = cop.cell_to_world(ORIGIN, CELL, path[0])
        sth = grid_heading_to_world_theta(heading_between(path[0], path[1]))
        kart = SimKart((sx, sy, sth))
        r = build_runner(view, path, (sx, sy, sth), kart, camera)
        departures = []          # true lateral offset when each leg starts
        original_leg = runner_mod.Leg

        def leg_spy(frm, *a, **kw):
            cx, cy, _ = cop.cell_to_world(ORIGIN, CELL, frm)
            ax = round(kart.truth[2] / (math.pi / 2)) * (math.pi / 2)
            dx, dy = kart.truth[0] - cx, kart.truth[1] - cy
            departures.append(abs(-math.sin(ax) * dx + math.cos(ax) * dy))
            return original_leg(frm, *a, **kw)

        runner_mod.Leg = leg_spy
        try:
            for _ in range(200000):
                odom = tuple(kart.odom)
                wx, wy, wth = r.mirror.world_pose(odom, r.odom_ref)
                r._advance(kart.telemetry(view), wx, wy, wth, odom=odom)
                if r.state == "idle":
                    break
                if r.camera_fix is not None and r.camera_fix._phase == "processing":
                    time.sleep(0.002)
            else:
                self.fail("nao chegou ao goal")
        finally:
            runner_mod.Leg = original_leg
            runner_mod.CLOSEST_OBSTACLE_AS_REFERENCE, runner_mod.SWEEP_EVERY_N_LEGS = saved
        cx, cy, _ = cop.cell_to_world(ORIGIN, CELL, GOAL)
        final = math.hypot(kart.truth[0] - cx, kart.truth[1] - cy)
        return departures, final, r

    def test_camera_and_closest_obstacle_work_together(self):
        dep_none, final_none, _ = self._drive(camera=False, big=False)
        dep_big, final_big, r_big = self._drive(camera=False, big=True)
        dep_both, final_both, r_both = self._drive(camera=True, big=True)
        report = (f"\nnone      {np.round(np.array(dep_none) * 100, 1)} final {final_none * 100:.1f} cm"
                  f"\nsonar     {np.round(np.array(dep_big) * 100, 1)} final {final_big * 100:.1f} cm"
                  f"\nsonar+cam {np.round(np.array(dep_both) * 100, 1)} final {final_both * 100:.1f} cm")
        # The closest obstacle reference still runs on its own schedule with
        # the camera on: the camera adds to it, it does not replace it.
        self.assertGreater(r_big.sweeps_done, 0, report)
        self.assertGreater(r_both.sweeps_done, 0, report)
        self.assertGreater(r_both.camera_fixes_done, 0, report)
        # Together they keep the kart closer to the centre line than the
        # reference alone, and well inside the corridor.
        self.assertLess(max(dep_both), max(dep_big), report)
        self.assertLess(max(dep_both), 0.02, report)
        self.assertLess(final_both, final_big, report)


if __name__ == "__main__":
    unittest.main()
