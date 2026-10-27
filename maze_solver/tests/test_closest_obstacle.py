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
import runner.closest_obstacle as closest                         # noqa: E402
import runner.live_maze_runner as lmr                             # noqa: E402
from runner.closest_obstacle import (                             # noqa: E402
    ClosestObstacleFix, find_next_best, pick_side, side_refs,
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

    @mock.patch("runner.range_fix.SENSOR_FORWARD_OFFSET_M", 0.13)
    def test_adjacent_walls_are_in_the_blind_zone(self):
        """0.15 m half-cell minus a 0.13 m sensor offset = 2 cm: unreadable."""
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
                                  last_turn=last, both=False,
                                  clock=self.clock.monotonic)

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


class RunnerHarness(unittest.TestCase):
    """A runner built without __init__, with the flags pinned so the tests do
    not move when constants.py is tuned. No tests of its own."""
    NEXT_BEST = False

    def setUp(self):
        self.clock = Clock()
        for target, name, value in (
                (lmr, "ClosestObstacleFix",
                 functools.partial(ClosestObstacleFix, clock=self.clock.monotonic)),
                (lmr, "CLOSEST_OBSTACLE_AS_REFERENCE", True),
                (lmr, "NEXT_BEST_CLOSEST_AS_REFERENCE", self.NEXT_BEST),
                (lmr, "MAX_NEXT_BEST_CELLS", 3),
                (closest, "CLOSEST_OBSTACLE_ALWAYS_CHECKS_THE_TWO", False)):
            patcher = mock.patch.object(target, name, value)
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
        r.plan_path = [HERE_CELL]
        r.plan_index = 0
        r._next_best = None
        r._leg_max_cells = r._leg_max_cells_saved = 3
        r._side_ref_scheduled = False
        r._update_kart_view = lambda *a: None
        return r

    def drive(self, r, kart):
        self.assertTrue(r._start_sweep(*kart.belief()))
        while r.side_ref is not None:
            r._poll_side_ref(kart.telemetry(), *kart.belief())
            self.clock.now += 0.05



class TestRunnerInterval(RunnerHarness):
    """The interval restarts only after a real reading, and turns alternate."""

    def test_success_restarts_the_interval_and_shifts_the_anchor(self):
        c = center(HERE_CELL)
        kart = SideKart(None, (c[0] - 0.03, c[1]), c)
        r = self.runner({(2, 0)}, kart)
        self.drive(r, kart)
        self.assertEqual(r._cells_since_sweep, 0)
        self.assertEqual(r.sweeps_done, 1)
        self.assertEqual(len(r.mirror.shifts), 1)

    @mock.patch("runner.range_fix.SENSOR_FORWARD_OFFSET_M", 0.13)
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


# The geometry the "better" reference is about: 0.30 m cells, sensor 0.10 m
# ahead, so an adjacent wall reads 5 cm and one past an empty cell 35 cm.
OFFSET_010 = mock.patch("runner.range_fix.SENSOR_FORWARD_OFFSET_M", 0.10)


@OFFSET_010
class TestBetterReference(unittest.TestCase):

    def refs(self, walls):
        x, y = center(HERE_CELL)
        return side_refs(GridView(5, 7, walls=walls), ORIGIN, CELL, x, y, NORTH)

    def test_adjacent_wall_is_usable_but_not_better(self):
        ref = self.refs({(2, 1)})[+1]
        self.assertTrue(ref.usable)
        self.assertFalse(ref.better)
        self.assertEqual(ref.gap, 0)

    def test_one_empty_cell_between_is_better(self):
        ref = self.refs({(2, 0)})[+1]
        self.assertTrue(ref.better)
        self.assertAlmostEqual(ref.expected, 0.35, places=6)

    def test_two_empty_cells_is_past_the_trust_ceiling(self):
        ref = self.refs({(2, 5)})[-1]                    # 65 cm east
        self.assertFalse(ref.usable)


@OFFSET_010
class TestFindNextBest(unittest.TestCase):
    NORTH_PATH = [(4, 2), (3, 2), (2, 2), (1, 2)]

    def test_finds_the_first_cell_ahead_with_a_better_side(self):
        view = GridView(5, 5, walls={(2, 0)})
        self.assertEqual(find_next_best(view, ORIGIN, CELL, self.NORTH_PATH,
                                        1, 3), (2, +1))

    def test_respects_max_cells(self):
        view = GridView(5, 5, walls={(2, 0)})
        self.assertIsNone(find_next_best(view, ORIGIN, CELL, self.NORTH_PATH,
                                         0, 1))

    def test_judged_on_the_heading_the_kart_arrives_with(self):
        """Path west then north: at (2, 2) the kart arrives heading WEST, so
        its sides are north/south -- the wall at (0, 2) behind the empty
        (1, 2) is a better reference there, on -90."""
        path = [(2, 4), (2, 3), (2, 2), (1, 2)]
        view = GridView(5, 5, walls={(0, 2)})
        self.assertEqual(find_next_best(view, ORIGIN, CELL, path, 0, 3),
                         (2, -1))


@OFFSET_010
class TestBothSides(unittest.TestCase):

    def setUp(self):
        self.clock = Clock()
        self.c = center(HERE_CELL)

    def session(self, walls, kart, **kw):
        view = GridView(5, 5, walls=walls)
        kart.view = view
        return ClosestObstacleFix(view, ORIGIN, CELL, kart, *kart.belief(),
                                  clock=self.clock.monotonic, **kw)

    def test_visits_both_sides_and_lands_the_belief_on_the_truth(self):
        kart = SideKart(None, (self.c[0] - 0.03, self.c[1]), self.c)
        fix = self.session({(2, 0), (2, 4)}, kart, both=True)
        run(fix, kart, self.clock)

        self.assertEqual(fix.sides, [+1, -1])
        kinds = [c[0] for c in kart.commands]
        self.assertEqual(kinds, ["turn", "drive_distance", "turn", "turn", "turn"])
        turns = [c[1] for c in kart.commands if c[0] == "turn"]
        self.assertTrue(all(abs(t) <= 90.0 + 1e-6 for t in turns), turns)
        self.assertAlmostEqual(kart.bx, kart.x, places=6)
        self.assertAlmostEqual(kart.x, self.c[0], places=6)
        self.assertAlmostEqual(math.remainder(kart.th - NORTH, 2 * math.pi), 0.0,
                               places=9)

    def test_a_side_without_an_obstacle_is_not_turned_to(self):
        kart = SideKart(None, self.c, self.c)
        fix = self.session({(2, 0)}, kart, both=True)
        run(fix, kart, self.clock)
        self.assertEqual(fix.sides, [+1])
        self.assertEqual([c[0] for c in kart.commands], ["turn", "turn"])

    def test_forced_side_beats_alternation(self):
        kart = SideKart(None, self.c, self.c)
        fix = self.session({(2, 0), (2, 4)}, kart, last_turn=+1, force_side=+1,
                           both=False)
        self.assertEqual(fix.sides, [+1])          # alternation alone says -1


@OFFSET_010
class TestRunnerNextBest(RunnerHarness):
    """Kart at (3, 2) heading north, adjacent walls both sides (plain, 5 cm);
    (2, 2) next on the path has (2, 1) empty and (2, 0) a wall -- better."""
    NEXT_BEST = True
    PATH = [(4, 2), (3, 2), (2, 2), (1, 2)]
    WALLS = {(3, 1), (3, 3), (2, 0), (2, 3)}

    def at(self, kart, cell):
        kart.x, kart.y = center(cell)
        kart.bx, kart.by = center(cell)

    def scheduled_runner(self, walls=WALLS, sensor=True):
        kart = SideKart(None, center((3, 2)), center((3, 2)), sensor=sensor)
        r = self.runner(walls, kart)
        r.plan_path, r.plan_index = list(self.PATH), 1
        r._last_side_turn = +1         # alternation alone would pick -90 next
        return r, kart

    def test_postpones_to_the_better_cell_one_cell_at_a_time(self):
        r, kart = self.scheduled_runner()
        self.assertFalse(r._start_sweep(*kart.belief()))
        self.assertEqual(r._next_best, (2, (2, 2), +1))
        self.assertEqual(r._leg_max_cells, 1)
        self.assertEqual(kart.commands, [])

    def test_runs_there_on_the_better_side_then_restores(self):
        r, kart = self.scheduled_runner()
        r._start_sweep(*kart.belief())
        r.plan_index = 2
        self.at(kart, (2, 2))
        self.drive(r, kart)

        # (2, 3) is a usable plain wall on -90, and -90 is what alternation
        # wanted -- the postponed correction still faces the better +90 one.
        self.assertEqual(round(kart.commands[0][1]), 90)
        self.assertIsNone(r._next_best)
        self.assertEqual(r._leg_max_cells, 3)
        self.assertEqual(r._cells_since_sweep, 0)

    def test_postponed_run_restarts_the_count_even_without_a_reading(self):
        r, kart = self.scheduled_runner(sensor=False)
        r._start_sweep(*kart.belief())
        r.plan_index = 2
        self.at(kart, (2, 2))
        self.drive(r, kart)
        self.assertEqual(r._cells_since_sweep, 0)
        self.assertEqual(r._leg_max_cells, 3)

    def test_nothing_better_ahead_corrects_here(self):
        r, kart = self.scheduled_runner(walls={(3, 1), (3, 3)})
        self.assertTrue(r._start_sweep(*kart.belief()))
        self.assertIsNone(r._next_best)
        self.assertEqual(r._leg_max_cells, 3)

    def test_better_here_is_used_here(self):
        kart = SideKart(None, center((3, 2)), center((3, 2)))
        r = self.runner({(3, 0)}, kart)
        r.plan_path, r.plan_index = list(self.PATH), 1
        self.assertTrue(r._start_sweep(*kart.belief()))
        self.assertIsNone(r._next_best)

    def test_replan_cancels_and_restores(self):
        r, kart = self.scheduled_runner()
        r._start_sweep(*kart.belief())
        r.waypoint, r.goal_cell, r.state = None, (1, 2), "running"
        r._live_revision = r._planned_revision = 0
        r.view = GridView(5, 5, walls=self.WALLS)
        r._replan_live((3, 2))
        self.assertIsNone(r._next_best)
        self.assertEqual(r._leg_max_cells, 3)

    def test_off_flag_never_postpones(self):
        r, kart = self.scheduled_runner()
        with mock.patch.object(lmr, "NEXT_BEST_CLOSEST_AS_REFERENCE", False):
            self.assertTrue(r._start_sweep(*kart.belief()))
        self.assertIsNone(r._next_best)


if __name__ == "__main__":
    unittest.main()
