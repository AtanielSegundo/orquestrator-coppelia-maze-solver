#!/usr/bin/python3
"""
Unit tests for how the front sensor writes the LIVE maze
(LiveMazeRunner._scan_obstacle and the suspect hold around it).

The rule being pinned: a reading only BLOCKS a cell when the kart is steady,
in the cell right before it, facing it. A moving kart that sees something on
the cells its leg is about to enter is stopped short -- in that cell before --
and the steady check decides from there.

The runner is built without its __init__ (that needs CoppeliaSim and the
camera); only the state these methods read is filled in.

    cd maze_solver && python -m unittest discover -s tests
"""
import math
import os
import sys
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
MAZE_SOLVER = os.path.dirname(HERE)
REPO = os.path.dirname(MAZE_SOLVER)
sys.path.insert(0, REPO)
sys.path.insert(0, MAZE_SOLVER)

import Coppelia.helpers as cop                                    # noqa: E402
import runner.live_maze_runner as lmr                             # noqa: E402
from runner.constants import (                                    # noqa: E402
    OBSTACLE_CONFIRM_S, OBSTACLE_SUSPECT_TIMEOUT_S, SENSOR_FORWARD_OFFSET_M,
)
from runner.leg import Leg                                        # noqa: E402
from test_range_fix import FakeView                               # noqa: E402

CELL = 0.30                     # tcc.maze's cell, the one the bench uses
ORIGIN = (0.0, 0.0, 0.0)
EAST = 0.0                      # world theta of grid heading 0


class GridView(FakeView):
    """FakeView plus the two writes the runner makes."""

    def mark_obstacle(self, cell):
        if not self.in_bounds(cell) or self.is_wall(cell):
            return False
        self.obstacles.add(cell)
        return True

    def clear_obstacle(self, cell):
        if cell not in self.obstacles:
            return False
        self.obstacles.discard(cell)
        return True


class FakeLink:
    def __init__(self):
        self.commands = []

    def goto(self, x, y, theta_deg=None):
        self.commands.append(("goto", x, y, theta_deg))

    def turn(self, angle_deg, speed=1.0):
        self.commands.append(("turn", angle_deg))

    def drive_distance(self, distance_m, speed=0.2):
        self.commands.append(("drive_distance", distance_m))

    def halt(self):
        self.commands.append(("halt",))


class IdentityMirror:
    def odom_target(self, odom_ref, wx, wy, wtheta=None):
        return wx, wy, math.degrees(wtheta)


class Clock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now


def center(cell):
    x, y, _ = cop.cell_to_world(ORIGIN, CELL, cell)
    return x, y


def reading_cm(surface_cells):
    """Front reading for a surface `surface_cells` ahead of the kart's centre."""
    return round((surface_cells * CELL - SENSOR_FORWARD_OFFSET_M) * 100.0)


def make_runner(walls=(), path=None, goal=None):
    r = lmr.LiveMazeRunner.__new__(lmr.LiveMazeRunner)
    r.view = GridView(4, 8, walls=walls)
    r.origin = ORIGIN
    r.cell_length = CELL
    r.sim = None
    r.maze_tag = "t"
    r.link = FakeLink()
    r.mirror = IdentityMirror()
    r.odom_ref = (0.0, 0.0, 0.0)
    r.plan_path = list(path or [])
    r.plan_index = 0
    r.visited = list(r.plan_path[:1])
    r.goal_cell = goal if goal is not None else (3, 7)
    r.leg = None
    r.status = ""
    r.replan_requested = False
    r.obstacles_found = 0
    r._last_slip = {}
    r._dist_seen_value = None
    r._dist_seen_pose = None
    r._clear_looks = {}
    r._live_revision = 0
    r._cells_since_sweep = 0
    r._ahead_verdict = None
    r._suspect = None
    r._suspect_verdict = None
    r._suspect_deadline = None
    r._dismissed = set()
    return r


class RunnerCase(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        for target, value in ((lmr, "time"), (lmr.cop, "add_maze_obstacle"),
                              (lmr.cop, "remove_object")):
            stub = self.clock if value == "time" else (lambda *a, **k: None)
            patcher = mock.patch.object(target, value, stub)
            patcher.start()
            self.addCleanup(patcher.stop)

    def look(self, r, cell, dist_cm, steady, wth=EAST, along=0.0, frames=1,
             dt=0.05):
        """`frames` telemetry frames from `along` metres east of `cell`'s
        centre, `dt` seconds apart."""
        x, y = center(cell)
        tel = {"front_distance_cm": dist_cm}
        for _ in range(frames):
            r._scan_obstacle(tel, x + along, y, wth, steady)
            self.clock.now += dt

    def hold_still(self, r, cell, dist_cm, wth=EAST):
        """Steady, same reading, a little longer than the confirm window."""
        frames = int(OBSTACLE_CONFIRM_S / 0.05) + 3
        self.look(r, cell, dist_cm, steady=True, wth=wth, frames=frames)


class TestOnlyASteadyKartMarks(RunnerCase):

    def test_steady_kart_marks_the_cell_in_front(self):
        r = make_runner()
        self.look(r, (1, 1), reading_cm(0.6), steady=True, frames=2)
        self.assertEqual(r.view.obstacles, set(), "marked before confirming")
        self.hold_still(r, (1, 1), reading_cm(0.6))
        self.assertEqual(r.view.obstacles, {(1, 2)})

    def test_moving_kart_never_marks(self):
        r = make_runner()
        self.look(r, (1, 1), reading_cm(0.6), steady=False, frames=40)
        self.assertEqual(r.view.obstacles, set())

    def test_two_cells_ahead_is_not_marked(self):
        r = make_runner()
        self.hold_still(r, (1, 1), reading_cm(1.7))
        self.assertEqual(r.view.obstacles, set())

    def test_off_axis_kart_does_not_mark(self):
        r = make_runner()
        self.hold_still(r, (1, 1), reading_cm(0.6), wth=math.radians(30))
        self.assertEqual(r.view.obstacles, set())

    def test_a_moving_frame_restarts_the_confirmation(self):
        r = make_runner()
        half = int(OBSTACLE_CONFIRM_S / 0.05 * 0.75)
        self.look(r, (1, 1), reading_cm(0.6), steady=True, frames=half)
        self.look(r, (1, 1), reading_cm(0.6), steady=False)
        self.look(r, (1, 1), reading_cm(0.6), steady=True, frames=half)
        self.assertEqual(r.view.obstacles, set())

    def test_wall_face_on_the_boundary_does_not_mark_the_open_cell(self):
        """Open cell, then a wall: its face sits exactly on the boundary, and
        without the depth bias whole-cm rounding decided which cell got it."""
        for dist in (33, 34):                    # 0.45 - 0.113 = 33.7 cm
            r = make_runner(walls={(1, 3)})
            self.hold_still(r, (1, 1), dist)
            self.assertEqual(r.view.obstacles, set(), f"{dist} cm")

    def test_goal_is_never_marked(self):
        r = make_runner(goal=(1, 2))
        self.hold_still(r, (1, 1), reading_cm(0.6))
        self.assertEqual(r.view.obstacles, set())

    def test_clear_reading_still_undiscovers_while_moving(self):
        """The live half is unchanged: a beam through a discovered cell
        clears it, moving or not."""
        r = make_runner()
        r.view.obstacles.add((1, 2))
        self.look(r, (1, 1), 200, steady=False, frames=10)
        self.assertEqual(r.view.obstacles, set())


class TestSuspectStopsTheLegShort(RunnerCase):
    PATH = [(1, 0), (1, 1), (1, 2), (1, 3)]

    def runner_on_leg(self):
        r = make_runner(path=self.PATH, goal=(1, 3))
        tw, ot = r._leg_target((1, 3))
        r.leg = Leg((1, 0), (1, 3), EAST, CELL,
                    target_world=tw, odom_target=ot)
        r.leg.poll(r.link, {}, *center((1, 0)))           # goto issued
        return r

    def test_far_suspect_retargets_to_the_cell_before_it(self):
        r = self.runner_on_leg()
        self.look(r, (1, 0), reading_cm(1.6), steady=False)    # hit in (1, 2)

        self.assertEqual(r._suspect, (1, 2))
        self.assertEqual((r.leg.to_cell, r.leg.n_cells), ((1, 1), 1))
        _, gx, gy, _ = r.link.commands[-1]
        self.assertEqual((gx, gy), center((1, 1)))
        self.assertEqual(r.view.obstacles, set(), "a suspect is not a mark")

    def test_adjacent_suspect_halts_in_place(self):
        r = self.runner_on_leg()
        self.look(r, (1, 0), reading_cm(0.7), steady=False)    # hit in (1, 1)

        self.assertEqual(r._suspect, (1, 1))
        self.assertIsNone(r.leg)
        self.assertEqual(r.link.commands[-1], ("halt",))
        self.assertEqual(r.plan_index, 0)

    def test_already_in_the_cell_before_halts_and_advances_the_plan(self):
        r = self.runner_on_leg()
        # Kart just past (1, 1)'s centre when (1, 2) shows up.
        self.look(r, (1, 1), reading_cm(0.6), steady=False, along=0.03)

        self.assertEqual(r.link.commands[-1], ("halt",))
        self.assertEqual(r.plan_index, 1)
        self.assertEqual(r.visited, [(1, 0), (1, 1)])
        self.assertEqual(r._cells_since_sweep, 1)

    def test_still_turning_onto_the_run_is_no_suspect(self):
        r = self.runner_on_leg()
        self.look(r, (1, 0), reading_cm(0.7), steady=False,
                  wth=math.radians(25))
        self.assertIsNone(r._suspect)

    def test_hit_off_the_route_is_no_suspect(self):
        r = make_runner(path=[(1, 0), (1, 1), (0, 1)], goal=(0, 1))
        r.leg = Leg((1, 0), (1, 1), EAST, CELL,
                    target_world=center((1, 1)), odom_target=(0, 0, 0))
        # Facing east, the hit two cells out is past where this leg turns.
        self.look(r, (1, 0), reading_cm(1.6), steady=False)
        self.assertIsNone(r._suspect)


class TestHoldSettlesTheSuspect(RunnerCase):

    def halted_runner(self):
        r = make_runner(path=[(1, 0), (1, 1), (1, 2)], goal=(1, 2))
        r.leg = Leg((1, 0), (1, 2), EAST, CELL,
                    target_world=center((1, 2)), odom_target=(0, 0, 0))
        self.look(r, (1, 0), reading_cm(0.7), steady=False)
        self.assertEqual(r._suspect, (1, 1))
        self.assertFalse(r._hold_for_suspect(), "must hold until settled")
        return r

    def test_confirmed_suspect_is_marked_and_replanned(self):
        r = self.halted_runner()
        self.hold_still(r, (1, 0), reading_cm(0.7))

        self.assertTrue(r._hold_for_suspect())
        self.assertEqual(r.view.obstacles, {(1, 1)})
        self.assertTrue(r.replan_requested)
        self.assertEqual(r._dismissed, set())

    def test_clear_suspect_is_dismissed_for_the_next_leg(self):
        r = self.halted_runner()
        self.hold_still(r, (1, 0), 200)

        self.assertTrue(r._hold_for_suspect())
        self.assertEqual(r.view.obstacles, set())
        self.assertIn((1, 1), r._dismissed)

        r.leg = Leg((1, 0), (1, 2), EAST, CELL,
                    target_world=center((1, 2)), odom_target=(0, 0, 0))
        self.look(r, (1, 0), reading_cm(0.7), steady=False)
        self.assertIsNone(r._suspect, "stopped again for a dismissed cell")

    def test_no_usable_reading_times_out(self):
        r = self.halted_runner()
        self.look(r, (1, 0), None, steady=True, frames=5)
        self.assertFalse(r._hold_for_suspect())
        self.clock.now += OBSTACLE_SUSPECT_TIMEOUT_S
        self.assertTrue(r._hold_for_suspect())
        self.assertIn((1, 1), r._dismissed)


if __name__ == "__main__":
    unittest.main()
