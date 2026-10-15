#!/usr/bin/python3
"""
Unit tests for one path edge (runner/leg.py).

The property worth pinning is the one that motivated the absolute rewrite: a
leg entered with an offset must END ON the cell center regardless, instead of
carrying that offset through to the next cell. The relative hop cannot do it
(it is tested here too, as the thing being compared against and as the live
fallback path), so these two cases are driven side by side against the same
fake kart.

The fake kart plays the role DriveController does for real: a `goto` is
executed as face-the-point -> drive-straight, recomputed from the pose it
actually holds, which is what makes the absolute version self-correcting.
`goal_active` is held True for a configurable number of polls so the stale
telemetry guard in Leg is exercised rather than bypassed.

    cd maze_solver && python -m unittest discover -s tests
"""
import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
MAZE_SOLVER = os.path.dirname(HERE)
REPO = os.path.dirname(MAZE_SOLVER)
sys.path.insert(0, REPO)
sys.path.insert(0, MAZE_SOLVER)

import Coppelia.helpers as cop                                    # noqa: E402
from runner.leg import (Leg, run_heading,                         # noqa: E402
                        straight_run_end)
from shadow import grid_heading_to_world_theta                    # noqa: E402

CELL = 0.25
ORIGIN = (0.0, 0.0, 0.0)


class FakeKart:
    """Stands in for KartLink + DriveController.

    Holds a world pose and executes commands against it. `slip` scales how far
    a straight drive actually gets, so a test can show the absolute leg still
    arriving when the move itself undershoots -- the same justification the
    recentring tests use.

    Here the odom frame IS the world frame (identity anchor), so the tests can
    talk about one set of coordinates; the frame conversion itself is
    SimMirror.odom_target's job and is tested separately.
    """

    def __init__(self, x, y, theta, slip=0.0, active_polls=2):
        self.x, self.y, self.theta = x, y, theta
        self.slip = slip
        self.active_polls = active_polls
        self._active = 0
        self.commands = []
        self.turn1_sweep_deg = None

    # -- KartLink surface --------------------------------------------------
    def goto(self, x, y, theta_deg=None):
        # Recorded with the heading held on arrival, so a test can see how wide
        # a sweep turn1 was asked to make in one command -- the server issues
        # it as a single _turn_to, which is exactly why Leg pre-chunks.
        self.commands.append(("goto", x, y, theta_deg))
        # Wrapped to (-180, 180] the way the server's _turn_to does it, so a
        # -45 deg turn is measured as 45 and not as 315.
        sweep = math.atan2(y - self.y, x - self.x) - self.theta
        self.turn1_sweep_deg = abs(math.degrees(
            (sweep + math.pi) % (2 * math.pi) - math.pi))
        # turn1 + drive, recomputed from the pose actually held -- the whole
        # point of an absolute command.
        dx, dy = x - self.x, y - self.y
        dist = math.hypot(dx, dy) * (1.0 - self.slip)
        heading = math.atan2(dy, dx)
        self.x += math.cos(heading) * dist
        self.y += math.sin(heading) * dist
        # turn2: settle on the commanded final heading.
        self.theta = heading if theta_deg is None else math.radians(theta_deg)
        self._active = self.active_polls

    def turn(self, angle_deg, speed=1.0):
        self.commands.append(("turn", angle_deg, speed))
        self.theta += math.radians(angle_deg)
        self._active = self.active_polls

    def drive_distance(self, distance_m, speed=0.2):
        self.commands.append(("drive_distance", distance_m, speed))
        d = distance_m * (1.0 - self.slip)
        self.x += math.cos(self.theta) * d
        self.y += math.sin(self.theta) * d
        self._active = self.active_polls

    # -- telemetry ---------------------------------------------------------
    def telemetry(self):
        active = self._active > 0
        if active:
            self._active -= 1
        return {"goal_active": active}


def run_leg(kart, frm, to, max_polls=200, **kw):
    """Drive one leg to completion against a FakeKart. Returns its outcome."""
    leg = Leg(frm, to, kart.theta, CELL, **kw)
    for _ in range(max_polls):
        outcome = leg.poll(kart, kart.telemetry(), kart.x, kart.y)
        if outcome != "pending":
            return outcome, leg
    raise AssertionError("leg never settled")


def center(cell):
    x, y, _ = cop.cell_to_world(ORIGIN, CELL, cell)
    return x, y


def absolute_kw(to_cell, from_cell):
    """The target pair the runner builds (see LiveMazeRunner._leg_target),
    with the identity anchor this fake kart uses."""
    tx, ty = center(to_cell)
    wth = grid_heading_to_world_theta(run_heading(from_cell, to_cell))
    return {"target_world": (tx, ty),
            "odom_target": (tx, ty, math.degrees(wth))}


class TestAbsoluteLeg(unittest.TestCase):

    def test_offset_entry_still_lands_on_center(self):
        """The motivating case: the kart enters the leg off-center in BOTH
        axes, and must leave it ON the next center -- not displaced by the
        same offset it came in with."""
        frm, to = (1, 1), (1, 2)
        cx, cy = center(frm)
        dx, dy = 0.031, -0.024               # longitudinal AND lateral error
        kart = FakeKart(cx + dx, cy + dy, grid_heading_to_world_theta(0))

        outcome, _ = run_leg(kart, frm, to, **absolute_kw(to, frm))
        self.assertEqual(outcome, "arrived")

        tx, ty = center(to)
        self.assertAlmostEqual(kart.x, tx, places=9)
        self.assertAlmostEqual(kart.y, ty, places=9)

    def test_error_does_not_accumulate_over_many_legs(self):
        """Four legs with a fresh error injected before each one. Absolute
        targets mean the final error is bounded by the LAST injection, not by
        their sum."""
        path = [(1, 1), (1, 2), (1, 3), (2, 3), (3, 3)]
        cx, cy = center(path[0])
        kart = FakeKart(cx, cy, grid_heading_to_world_theta(0))

        kick = 0.02
        for frm, to in zip(path, path[1:]):
            kart.x += kick                   # drift injected every leg
            kart.y -= kick
            outcome, _ = run_leg(kart, frm, to, **absolute_kw(to, frm))
            self.assertEqual(outcome, "arrived")

        tx, ty = center(path[-1])
        # Each leg absorbed its own kick: zero residual, and in particular not
        # the 4 * kick a relative hop would have carried.
        self.assertLess(math.hypot(kart.x - tx, kart.y - ty), 1e-9)

    def test_issues_goto_with_the_commanded_heading(self):
        frm, to = (2, 2), (1, 2)             # north: grid heading 90
        cx, cy = center(frm)
        kart = FakeKart(cx, cy, grid_heading_to_world_theta(0))

        run_leg(kart, frm, to, **absolute_kw(to, frm))

        kinds = [c[0] for c in kart.commands]
        self.assertEqual(kinds, ["goto"])    # one command for the whole leg
        _, gx, gy, gth = kart.commands[0]
        tx, ty = center(to)
        self.assertAlmostEqual(gx, tx, places=9)
        self.assertAlmostEqual(gy, ty, places=9)
        self.assertAlmostEqual(gth, 90.0, places=6)

    def test_short_of_target_reports_blocked_with_distance_remaining(self):
        """A move that stops short is judged by how far from the TARGET it
        ended, and `shortfall` carries that remaining distance."""
        frm, to = (1, 1), (1, 2)
        cx, cy = center(frm)
        kart = FakeKart(cx, cy, grid_heading_to_world_theta(0), slip=0.5)

        outcome, leg = run_leg(kart, frm, to, **absolute_kw(to, frm))
        self.assertEqual(outcome, "blocked")
        tx, ty = center(to)
        self.assertAlmostEqual(leg.shortfall,
                               math.hypot(tx - kart.x, ty - kart.y), places=9)

    def test_guard_blocked_is_recorded_for_the_failure_message(self):
        frm, to = (1, 1), (1, 2)
        cx, cy = center(frm)
        kart = FakeKart(cx, cy, grid_heading_to_world_theta(0), slip=0.9)
        leg = Leg(frm, to, kart.theta, CELL, **absolute_kw(to, frm))

        outcome = "pending"
        while outcome == "pending":
            tel = kart.telemetry()
            tel["guard_blocked"] = True
            outcome = leg.poll(kart, tel, kart.x, kart.y)
        self.assertEqual(outcome, "blocked")
        self.assertTrue(leg.guard_seen)

    def test_falls_back_to_relative_without_both_frames(self):
        """No usable target -> the legacy hop, not a goto at an unknown point.
        This is the path a kart with a pre-goto server takes."""
        frm, to = (1, 1), (1, 2)
        cx, cy = center(frm)
        for kw in ({}, {"target_world": center(to)},
                   {"odom_target": (0.0, 0.0, 0.0)}):
            kart = FakeKart(cx, cy, grid_heading_to_world_theta(0))
            run_leg(kart, frm, to, **kw)
            self.assertNotIn("goto", [c[0] for c in kart.commands])

    def test_relative_flag_forces_the_legacy_path(self):
        frm, to = (1, 1), (1, 2)
        cx, cy = center(frm)
        kart = FakeKart(cx, cy, grid_heading_to_world_theta(0))
        kw = absolute_kw(to, frm)
        run_leg(kart, frm, to, relative=True, **kw)
        self.assertEqual([c[0] for c in kart.commands], ["drive_distance"])


class TestBoundedTurnSteps(unittest.TestCase):
    """The absolute target must not cost the bounded turn steps.

    `goto` issues each of its turn phases as ONE command
    (DriveController._turn_to), so a wide heading change left to the server
    would sweep in a single sweep -- overshooting more, and giving the front
    sensor only the two extremes to sample. The chunking runs before the goto
    instead of being replaced by it.
    """

    def test_wide_turn_is_chunked_before_the_goto(self):
        frm, to = (1, 1), (0, 1)             # 90 deg from heading 0
        cx, cy = center(frm)
        kart = FakeKart(cx, cy, grid_heading_to_world_theta(0))

        outcome, _ = run_leg(kart, frm, to, turn_max_angle_step=30.0,
                             **absolute_kw(to, frm))
        self.assertEqual(outcome, "arrived")

        kinds = [c[0] for c in kart.commands]
        # Chunks first, then exactly one goto for the translation.
        self.assertEqual(kinds, ["turn", "turn", "goto"])
        turns = [c[1] for c in kart.commands if c[0] == "turn"]
        self.assertTrue(all(abs(t) <= 30.0 + 1e-9 for t in turns), turns)

    def test_chunks_stop_once_the_remainder_fits_one_step(self):
        """The pre-turn does not grind the error to zero: it stops as soon as
        what is left fits in a step and lets goto's gyro-closed turn1 finish,
        which is the better instrument for the last few degrees."""
        frm, to = (1, 1), (0, 1)
        cx, cy = center(frm)
        kart = FakeKart(cx, cy, grid_heading_to_world_theta(0))

        leg = Leg(frm, to, kart.theta, CELL, turn_max_angle_step=40.0,
                  **absolute_kw(to, frm))
        outcome = "pending"
        while outcome == "pending":
            outcome = leg.poll(kart, kart.telemetry(), kart.x, kart.y)

        turns = [c[1] for c in kart.commands if c[0] == "turn"]
        # 90 deg at a 40 deg step: two chunks leave 10 deg, inside one step.
        self.assertEqual(len(turns), 2)
        self.assertAlmostEqual(sum(turns), 80.0, places=6)
        self.assertIn("goto", [c[0] for c in kart.commands])

    def test_turn_inside_one_step_goes_straight_to_goto(self):
        """Nothing to slice -> no pre-turn at all, goto's turn1 does it."""
        frm, to = (1, 1), (0, 1)
        cx, cy = center(frm)
        kart = FakeKart(cx, cy, grid_heading_to_world_theta(0))
        run_leg(kart, frm, to, turn_max_angle_step=90.0, **absolute_kw(to, frm))
        self.assertEqual([c[0] for c in kart.commands], ["goto"])

    def test_no_step_limit_means_no_chunking(self):
        frm, to = (1, 1), (0, 1)
        cx, cy = center(frm)
        for step in (0.0, None):
            kart = FakeKart(cx, cy, grid_heading_to_world_theta(0))
            run_leg(kart, frm, to, turn_max_angle_step=step,
                    **absolute_kw(to, frm))
            self.assertEqual([c[0] for c in kart.commands], ["goto"])

    def test_chunked_turn_still_lands_on_center(self):
        """The pre-turn must not disturb the property the rewrite was for."""
        frm, to = (1, 1), (0, 1)
        cx, cy = center(frm)
        kart = FakeKart(cx + 0.02, cy - 0.018, grid_heading_to_world_theta(0))

        outcome, _ = run_leg(kart, frm, to, turn_max_angle_step=30.0,
                             **absolute_kw(to, frm))
        self.assertEqual(outcome, "arrived")
        tx, ty = center(to)
        self.assertAlmostEqual(kart.x, tx, places=9)
        self.assertAlmostEqual(kart.y, ty, places=9)

    def test_180_turn_is_sampled_in_steps(self):
        """A reversal is the case that motivated the step bound: the sensor
        gets intermediate looks instead of only the two extremes."""
        frm, to = (1, 2), (1, 1)            # 180 deg from heading 0
        cx, cy = center(frm)
        kart = FakeKart(cx, cy, grid_heading_to_world_theta(0))
        run_leg(kart, frm, to, turn_max_angle_step=45.0, **absolute_kw(to, frm))
        turns = [c[1] for c in kart.commands if c[0] == "turn"]
        self.assertGreaterEqual(len(turns), 3)
        self.assertTrue(all(abs(t) <= 45.0 + 1e-9 for t in turns), turns)

    def test_server_never_receives_a_sweep_wider_than_one_step(self):
        """The invariant that makes the whole arrangement worth it: no single
        command -- chunk or goto turn1 -- asks for more than one step."""
        step = 45.0
        for frm, to in [((1, 2), (1, 1)), ((1, 1), (0, 1)), ((1, 1), (2, 1))]:
            cx, cy = center(frm)
            kart = FakeKart(cx, cy, grid_heading_to_world_theta(0))
            run_leg(kart, frm, to, turn_max_angle_step=step,
                    **absolute_kw(to, frm))
            for c in kart.commands:
                if c[0] == "turn":
                    self.assertLessEqual(abs(c[1]), step + 1e-9, c)
            self.assertLessEqual(kart.turn1_sweep_deg, step + 1e-6,
                                 f"{frm}->{to}: turn1 swept "
                                 f"{kart.turn1_sweep_deg:.1f} deg in one command")


class TestRelativeLegCarriesError(unittest.TestCase):
    """The behaviour the rewrite exists to replace, pinned so the contrast is
    explicit: the relative hop reports "arrived" while leaving the kart
    displaced by exactly the offset it started with."""

    def test_offset_is_carried_through_not_corrected(self):
        frm, to = (1, 1), (1, 2)
        cx, cy = center(frm)
        dx, dy = 0.02, -0.015
        kart = FakeKart(cx + dx, cy + dy, grid_heading_to_world_theta(0))

        outcome, _ = run_leg(kart, frm, to, relative=True)
        self.assertEqual(outcome, "arrived")   # it thinks it did its job

        tx, ty = center(to)
        self.assertAlmostEqual(kart.x - tx, dx, places=9)
        self.assertAlmostEqual(kart.y - ty, dy, places=9)

    def test_turn_is_sliced_into_bounded_chunks(self):
        frm, to = (1, 1), (0, 1)            # 90 deg from heading 0
        cx, cy = center(frm)
        kart = FakeKart(cx, cy, grid_heading_to_world_theta(0))
        run_leg(kart, frm, to, relative=True, turn_max_angle_step=30.0)
        turns = [c[1] for c in kart.commands if c[0] == "turn"]
        self.assertEqual(len(turns), 3)
        self.assertAlmostEqual(sum(turns), 90.0, places=6)


class TestStraightRuns(unittest.TestCase):
    """Consecutive cells with one heading are driven as ONE leg."""

    EAST5 = [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5)]

    def test_run_is_capped_at_max_cells(self):
        self.assertEqual(straight_run_end(self.EAST5, 0, 3), 3)
        # The rest of the corridor is the next leg.
        self.assertEqual(straight_run_end(self.EAST5, 3, 3), 5)

    def test_run_ends_where_the_heading_changes(self):
        path = [(1, 0), (1, 1), (1, 2), (0, 2), (0, 3)]
        self.assertEqual(straight_run_end(path, 0, 3), 2)
        self.assertEqual(straight_run_end(path, 2, 3), 3)   # the turn north

    def test_cap_of_one_is_cell_by_cell(self):
        for i in range(len(self.EAST5) - 1):
            self.assertEqual(straight_run_end(self.EAST5, i, 1), i + 1)

    def test_multi_cell_leg_is_one_goto_to_the_far_center(self):
        frm, to = (1, 1), (1, 4)
        cx, cy = center(frm)
        kart = FakeKart(cx + 0.02, cy - 0.01, grid_heading_to_world_theta(0))

        outcome, leg = run_leg(kart, frm, to, **absolute_kw(to, frm))
        self.assertEqual(outcome, "arrived")
        self.assertEqual(leg.n_cells, 3)
        self.assertEqual([c[0] for c in kart.commands], ["goto"])
        tx, ty = center(to)
        self.assertAlmostEqual(kart.x, tx, places=9)
        self.assertAlmostEqual(kart.y, ty, places=9)

    def test_multi_cell_relative_leg_drives_the_whole_run(self):
        frm, to = (3, 1), (0, 1)                 # three cells north
        cx, cy = center(frm)
        kart = FakeKart(cx, cy, grid_heading_to_world_theta(90))

        outcome, _ = run_leg(kart, frm, to, relative=True)
        self.assertEqual(outcome, "arrived")
        self.assertEqual(kart.commands, [("drive_distance", 3 * CELL, 0.2)])

    def test_a_bend_is_not_a_leg(self):
        with self.assertRaises(ValueError):
            Leg((1, 1), (2, 2), 0.0, CELL)


class LazyKart(FakeKart):
    """A FakeKart whose moves only land when they FINISH, and a newer command
    replaces one in flight -- which is what the server does with a re-issued
    goto or drive_distance, and what retarget relies on."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._pending = None

    def goto(self, x, y, theta_deg=None):
        self.commands.append(("goto", x, y, theta_deg))
        self._pending = ("goto", x, y, theta_deg)
        self._active = self.active_polls

    def drive_distance(self, distance_m, speed=0.2):
        self.commands.append(("drive_distance", distance_m, speed))
        self._pending = ("drive_distance", distance_m)
        self._active = self.active_polls

    def telemetry(self):
        tel = super().telemetry()
        if not tel["goal_active"] and self._pending is not None:
            move, self._pending = self._pending, None
            if move[0] == "goto":
                self.x, self.y = move[1], move[2]
            else:
                self.x += math.cos(self.theta) * move[1]
                self.y += math.sin(self.theta) * move[1]
        return tel


class TestRetarget(unittest.TestCase):
    """Cutting a run short in flight -- how the runner stops the kart in the
    cell before a suspected obstacle."""

    def test_in_flight_goto_is_reissued_to_the_nearer_center(self):
        frm, far, near = (1, 0), (1, 3), (1, 1)
        cx, cy = center(frm)
        kart = LazyKart(cx, cy, grid_heading_to_world_theta(0), active_polls=3)
        leg = Leg(frm, far, kart.theta, CELL, **absolute_kw(far, frm))

        leg.poll(kart, kart.telemetry(), kart.x, kart.y)       # goto -> far
        leg.retarget(kart, near, kart.x, kart.y, **absolute_kw(near, frm))
        outcome = "pending"
        while outcome == "pending":
            outcome = leg.poll(kart, kart.telemetry(), kart.x, kart.y)

        self.assertEqual(outcome, "arrived")
        self.assertEqual((leg.to_cell, leg.n_cells), (near, 1))
        gotos = [c for c in kart.commands if c[0] == "goto"]
        self.assertEqual(len(gotos), 2)
        tx, ty = center(near)
        self.assertAlmostEqual(kart.x, tx, places=9)
        self.assertAlmostEqual(kart.y, ty, places=9)

    def test_before_the_goto_is_issued_nothing_is_sent_early(self):
        """Still pre-turning: the goto that follows simply aims at the new end."""
        frm, far, near = (1, 1), (4, 1), (2, 1)             # south, from heading 90
        cx, cy = center(frm)
        kart = FakeKart(cx, cy, grid_heading_to_world_theta(90))
        leg = Leg(frm, far, kart.theta, CELL, turn_max_angle_step=45.0,
                  **absolute_kw(far, frm))
        leg.poll(kart, kart.telemetry(), kart.x, kart.y)       # first turn chunk
        leg.retarget(kart, near, kart.x, kart.y, **absolute_kw(near, frm))
        outcome = "pending"
        while outcome == "pending":
            outcome = leg.poll(kart, kart.telemetry(), kart.x, kart.y)

        self.assertEqual(outcome, "arrived")
        gotos = [c for c in kart.commands if c[0] == "goto"]
        self.assertEqual(len(gotos), 1)
        tx, ty = center(near)
        self.assertAlmostEqual(gotos[0][1], tx, places=9)
        self.assertAlmostEqual(gotos[0][2], ty, places=9)

    def test_relative_leg_drives_only_what_is_left(self):
        frm, far, near = (1, 0), (1, 3), (1, 2)
        cx, cy = center(frm)
        kart = LazyKart(cx, cy, grid_heading_to_world_theta(0), active_polls=3)
        leg = Leg(frm, far, kart.theta, CELL, relative=True)

        leg.poll(kart, kart.telemetry(), kart.x, kart.y)       # drive 3 cells
        # Pretend the kart is already 0.1 m along when the suspect shows up.
        leg.retarget(kart, near, cx + 0.1, cy)
        self.assertAlmostEqual(kart.commands[-1][1], 2 * CELL - 0.1, places=9)
        self.assertEqual(leg.n_cells, 2)


if __name__ == "__main__":
    unittest.main()
