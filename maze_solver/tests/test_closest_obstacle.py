#!/usr/bin/python3
"""
Unit tests for the closest-side-obstacle lateral correction
(runner/closest_obstacle.py, behind CLOSEST_OBSTACLE_AS_REFERENCE).

Driven by a fake kart that keeps its TRUE pose apart from the BELIEVED one:
the sensor reads from the truth, the session only ever sees the belief, and a
move shifts both (the truth short by `slip`) -- which is exactly why the anchor
has to be corrected after the move. The property pinned throughout: when the
session ends, the belief (with the anchor shift applied) is on the truth.

    cd maze_solver && python -m unittest discover -s tests
"""
import functools
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
from runner.closest_obstacle import (                             # noqa: E402
    ClosestObstacleFix, pick_side,
)
from runner.constants import SWEEP_EVERY_N_LEGS                   # noqa: E402
from runner.range_fix import predict_range                        # noqa: E402
from test_obstacle_confirm import Clock, GridView                 # noqa: E402

CELL = 0.30
ORIGIN = (0.0, 0.0, 0.0)
NORTH = math.pi / 2           # world theta of grid heading 90 (row - 1)
HERE_CELL = (2, 2)


def center(cell):
    x, y, _ = cop.cell_to_world(ORIGIN, CELL, cell)
    return x, y


class SideKart:
    """KartLink + kart: true pose and believed pose kept apart."""

    def __init__(self, view, true_xy, belief_xy, th=NORTH, slip=0.0,
                 active_polls=2, sensor=True):
        self.view = view
        self.x, self.y = true_xy
        self.bx, self.by = belief_xy
        self.th = th
        self.slip = slip
        self.active_polls = active_polls
        self.sensor = sensor
        self._active = 0
        self.commands = []

    def turn(self, angle_deg, speed=1.0):
        self.commands.append(("turn", angle_deg))
        self.th += math.radians(angle_deg)
        self._active = self.active_polls

    def drive_distance(self, m, speed=0.2):
        self.commands.append(("drive_distance", m))
        c, s = math.cos(self.th), math.sin(self.th)
        self.x += c * m * (1.0 - self.slip)      # the wheels slip...
        self.y += s * m * (1.0 - self.slip)
        self.bx += c * m                         # ...odometry counts it all
        self.by += s * m
        self._active = self.active_polls

    def telemetry(self):
        active = self._active > 0
        if active:
            self._active -= 1
        dist = None
        if self.sensor:
            d, _ = predict_range(self.view, ORIGIN, CELL, self.x, self.y,
                                 self.th, 3.0)
            dist = None if d is None else round(d * 100.0)
        return {"goal_active": active, "front_distance_cm": dist}

    def belief(self):
        return self.bx, self.by, self.th


def run(fix, kart, clock):
    for _ in range(2000):
        out = fix.poll(kart.telemetry(), *kart.belief())
        clock.now += 0.05
        if out == "done":
            if fix.anchor_shift is not None:
                kart.bx += fix.anchor_shift[0]
                kart.by += fix.anchor_shift[1]
            return
    raise AssertionError("session never finished")


class TestPickSide(unittest.TestCase):
    """Kart at (2, 2) heading north; the orthogonal axis is the row."""

    def pick(self, walls, last=0, obstacles=()):
        view = GridView(5, 5, walls=walls)
        view.obstacles.update(obstacles)
        x, y = center(HERE_CELL)
        return pick_side(view, ORIGIN, CELL, x, y, NORTH, last)

    def test_adjacent_walls_are_in_the_blind_zone(self):
        """0.15 m half-cell minus the 0.13 m sensor offset = 2 cm: unreadable."""
        sign, cell, reasons = self.pick({(2, 1), (2, 3)})
        self.assertIsNone(sign)
        self.assertTrue(all("zona cega" in r for r in reasons), reasons)

    def test_wall_one_open_cell_away_is_used(self):
        sign, cell, _ = self.pick({(2, 0)})
        self.assertEqual((sign, cell), (+1, (2, 0)))     # +90 from north = west

    def test_alternates_when_both_sides_qualify(self):
        walls = {(2, 0), (2, 4)}
        self.assertEqual(self.pick(walls, last=+1)[0], -1)
        self.assertEqual(self.pick(walls, last=-1)[0], +1)
        self.assertEqual(self.pick(walls, last=0)[0], +1)

    def test_only_side_is_used_even_if_it_repeats(self):
        self.assertEqual(self.pick({(2, 0)}, last=+1)[0], +1)

    def test_discovered_obstacle_is_never_a_reference(self):
        sign, _, reasons = self.pick({(2, 0)}, obstacles={(2, 1)})
        self.assertIsNone(sign)
        self.assertTrue(any("descoberto" in r for r in reasons), reasons)

    def test_map_edge_is_no_reference(self):
        sign, _, reasons = self.pick(set())
        self.assertIsNone(sign)
        self.assertTrue(all("borda" in r for r in reasons), reasons)


class TestCorrection(unittest.TestCase):
    """Reference wall at (2, 0), 32 cm from the centre of (2, 2) facing west."""

    def setUp(self):
        self.view = GridView(5, 5, walls={(2, 0)})
        self.clock = Clock()
        self.c = center(HERE_CELL)

    def session(self, kart, last=0):
        return ClosestObstacleFix(self.view, ORIGIN, CELL, kart, *kart.belief(),
                                  last_turn=last, clock=self.clock.monotonic)

    def assert_on_truth(self, kart):
        self.assertAlmostEqual(kart.bx, kart.x, places=6)
        self.assertAlmostEqual(kart.by, kart.y, places=6)
        self.assertAlmostEqual(math.remainder(kart.th - NORTH, 2 * math.pi), 0.0,
                               places=9, msg="not back on the leg heading")

    def test_kart_drifted_toward_the_wall_is_moved_back_and_belief_fixed(self):
        # Truth 3 cm west of where it is believed (the cell centre).
        kart = SideKart(self.view, (self.c[0] - 0.03, self.c[1]), self.c)
        fix = self.session(kart)
        run(fix, kart, self.clock)

        self.assertTrue(fix.succeeded)
        self.assertEqual([c[0] for c in kart.commands],
                         ["turn", "drive_distance", "turn"])
        self.assertAlmostEqual(kart.commands[1][1], -0.03, places=6)  # backs off
        self.assertAlmostEqual(kart.x, self.c[0], places=6)  # physically centred
        self.assert_on_truth(kart)

    def test_slip_is_caught_by_the_reread(self):
        """Half the move lost to slip: the anchor still lands on the truth,
        because it comes from the second reading, not the commanded move."""
        kart = SideKart(self.view, (self.c[0] - 0.04, self.c[1]), self.c,
                        slip=0.5)
        fix = self.session(kart)
        run(fix, kart, self.clock)
        self.assertTrue(fix.succeeded)
        self.assert_on_truth(kart)

    def test_alpha_scales_the_move_not_the_belief(self):
        kart = SideKart(self.view, (self.c[0] - 0.06, self.c[1]), self.c)
        with mock.patch("runner.closest_obstacle.CLOSEST_OBSTACLE_ALPHA", 0.5):
            fix = self.session(kart)
            run(fix, kart, self.clock)
        self.assertAlmostEqual(kart.commands[1][1], -0.03, places=6)
        self.assert_on_truth(kart)

    def test_error_under_the_minimal_delta_does_not_move(self):
        kart = SideKart(self.view, (self.c[0] - 0.01, self.c[1]), self.c)
        fix = self.session(kart)
        run(fix, kart, self.clock)
        self.assertTrue(fix.succeeded)
        self.assertEqual([c[0] for c in kart.commands], ["turn", "turn"])
        self.assertIsNone(fix.anchor_shift)      # 1 cm: under the 2 cm floor

    def test_no_reading_is_not_a_success_and_still_turns_back(self):
        kart = SideKart(self.view, self.c, self.c, sensor=False)
        fix = self.session(kart)
        run(fix, kart, self.clock)
        self.assertFalse(fix.succeeded)
        self.assertIsNone(fix.anchor_shift)
        self.assertEqual([c[0] for c in kart.commands], ["turn", "turn"])
        self.assertAlmostEqual(math.remainder(kart.th - NORTH, 2 * math.pi), 0.0)

    def test_turn_back_also_removes_arrival_heading_error(self):
        kart = SideKart(self.view, self.c, self.c, th=NORTH + math.radians(4))
        run(self.session(kart), kart, self.clock)
        self.assertAlmostEqual(math.remainder(kart.th - NORTH, 2 * math.pi), 0.0,
                               places=9)


class IdentityMirror:
    def __init__(self):
        self.shifts = []

    def shift_anchor(self, dx, dy):
        self.shifts.append((dx, dy))

    def apply_world_pose(self, *a):
        pass


class TestRunnerInterval(unittest.TestCase):
    """The interval restarts only after a real reading, and turns alternate."""

    def setUp(self):
        self.clock = Clock()
        for name, value in (
                ("ClosestObstacleFix",
                 functools.partial(ClosestObstacleFix, clock=self.clock.monotonic)),
                ("CLOSEST_OBSTACLE_AS_REFERENCE", True)):
            patcher = mock.patch.object(lmr, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def runner(self, walls, kart):
        r = lmr.LiveMazeRunner.__new__(lmr.LiveMazeRunner)
        r.view = GridView(5, 5, walls=walls)
        r.origin, r.cell_length = ORIGIN, CELL
        r.link = kart
        kart.view = r.view
        r.mirror = IdentityMirror()
        r.turn_speed = 0.35
        r.side_ref = None
        r.sweep = None
        r._last_side_turn = 0
        r._cells_since_sweep = SWEEP_EVERY_N_LEGS
        r.sweeps_done = 0
        r.last_sweep_summary = ""
        r.status = ""
        r._update_kart_view = lambda *a: None
        return r

    def drive(self, r, kart):
        self.assertTrue(r._start_sweep(*kart.belief()))
        while r.side_ref is not None:
            r._poll_side_ref(kart.telemetry(), *kart.belief())
            self.clock.now += 0.05

    def test_success_restarts_the_interval_and_shifts_the_anchor(self):
        c = center(HERE_CELL)
        kart = SideKart(None, (c[0] - 0.03, c[1]), c)
        r = self.runner({(2, 0)}, kart)
        self.drive(r, kart)
        self.assertEqual(r._cells_since_sweep, 0)
        self.assertEqual(r.sweeps_done, 1)
        self.assertEqual(len(r.mirror.shifts), 1)

    def test_no_reference_keeps_the_interval_standing(self):
        c = center(HERE_CELL)
        kart = SideKart(None, c, c)
        r = self.runner({(2, 1), (2, 3)}, kart)          # blind zone both sides
        self.assertFalse(r._start_sweep(*kart.belief()))
        self.assertIsNone(r.side_ref)
        self.assertEqual(r._cells_since_sweep, SWEEP_EVERY_N_LEGS)
        self.assertEqual(kart.commands, [])

    def test_failed_reading_keeps_the_interval_standing(self):
        c = center(HERE_CELL)
        kart = SideKart(None, c, c, sensor=False)
        r = self.runner({(2, 0)}, kart)
        self.drive(r, kart)
        self.assertEqual(r._cells_since_sweep, SWEEP_EVERY_N_LEGS)
        self.assertEqual(r.sweeps_done, 0)

    def test_consecutive_corrections_alternate_sides(self):
        c = center(HERE_CELL)
        kart = SideKart(None, c, c)
        r = self.runner({(2, 0), (2, 4)}, kart)
        turns = []
        for _ in range(3):
            r._cells_since_sweep = SWEEP_EVERY_N_LEGS
            kart.commands.clear()
            self.drive(r, kart)
            turns.append(round(kart.commands[0][1]))
        self.assertEqual(turns, [90, -90, 90])


if __name__ == "__main__":
    unittest.main()
