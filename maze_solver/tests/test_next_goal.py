#!/usr/bin/python3
"""
Unit tests for picking the next goal once the kart has reached the current
one (LiveMazeRunner._pick_next_goal and the "ready" state).

The rule being pinned: a pick while parked on the goal makes the kart's own
cell the start and the picked cell the goal, plans the route right away, and
issues NOTHING to the kart until 'ok'.

The runner is built without its __init__ (that needs CoppeliaSim and the
camera); only the state these methods read is filled in.

    cd maze_solver && python -m unittest discover -s tests
"""
import os
import sys
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
MAZE_SOLVER = os.path.dirname(HERE)
REPO = os.path.dirname(MAZE_SOLVER)
sys.path.insert(0, REPO)
sys.path.insert(0, MAZE_SOLVER)

import runner.live_maze_runner as lmr                             # noqa: E402
from runner.constants import LEG_MAX_CELLS                        # noqa: E402
from test_obstacle_confirm import (                               # noqa: E402
    CELL, EAST, ORIGIN, FakeLink, GridView, IdentityMirror, center,
)

GOAL = (1, 1)                   # where the last run ended, kart parked on it


class StartGoalView(GridView):
    """GridView plus the start/goal markers the runner moves."""

    def set_start_goal(self, start, goal):
        self.start, self.goal = start, goal


class TelemetryLink(FakeLink):
    """FakeLink plus the telemetry step() reads: a kart standing still, no
    front reading (so the sensor never touches the live maze)."""

    def latest(self):
        return (0.0, 0.0, 0.0), (0.0, 0.0)

    def drive_telemetry(self):
        return {}


class ParkedMirror(IdentityMirror):
    """Reports the runner's last pose back as the world pose, so the kart
    stays exactly where the test put it."""

    def __init__(self, runner):
        self.runner = runner

    def world_pose(self, pose, odom_ref):
        return self.runner.last_world_pose

    def apply_world_pose(self, wx, wy, wth):
        pass

    def spin_wheels(self, left, right, dt):
        pass


def parked_runner(walls=()):
    """Live, idle, kart sitting on GOAL after driving (1, 0) -> GOAL."""
    r = lmr.LiveMazeRunner.__new__(lmr.LiveMazeRunner)
    r.view = StartGoalView(4, 8, walls=walls)
    r.view.set_start_goal((1, 0), GOAL)
    r.origin = ORIGIN
    r.cell_length = CELL
    r.sim = None
    r.maze_tag = "t"
    r.link = TelemetryLink()
    r.mirror = ParkedMirror(r)
    r.odom_ref = (0.0, 0.0, 0.0)
    r.camera = mock.Mock()
    r._update_kart_view = lambda *a: None
    r._last_slip = {}
    r.phase = "live"
    r.state = "idle"
    r.start_cell = (1, 0)
    r.goal_cell = GOAL
    r.waypoint = None
    r.plan_path = [(1, 0), GOAL]
    r.plan_index = 1
    r.visited = [(1, 0), GOAL]
    x, y = center(GOAL)
    r.last_world_pose = (x, y, EAST)
    r.leg = None
    r.status = ""
    r.replan_requested = False
    r.should_wait = False
    r._waiting_to_step = False
    r._step_released = False
    r.recenter = None
    r.sweep = None
    r.side_ref = None
    r._suspect = None
    r._next_best = None
    r._leg_max_cells = LEG_MAX_CELLS
    r._live_revision = 0
    r._planned_revision = -1
    r.turn_speed = 1.0
    r.drive_speed = 0.2
    return r


class NextGoalCase(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(lmr.cop, "move_maze_marker",
                                    lambda *a, **k: None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tick(self, r, frames=3):
        for _ in range(frames):
            r.step(0.05)


class TestPickWhileParked(NextGoalCase):

    def test_kart_cell_becomes_start_and_pick_the_goal(self):
        r = parked_runner()
        self.assertTrue(r.set_goal((1, 4)))
        self.assertEqual((r.start_cell, r.goal_cell), (GOAL, (1, 4)))
        self.assertEqual((r.view.start, r.view.goal), (GOAL, (1, 4)))
        self.assertEqual(r.plan_path, [(1, 1), (1, 2), (1, 3), (1, 4)])
        self.assertEqual(r.plan_index, 0)
        self.assertEqual(r.visited, [GOAL])
        self.assertEqual(r.state, "ready")

    def test_left_click_also_picks_the_goal(self):
        r = parked_runner()
        self.assertTrue(r.set_start((1, 4)))
        self.assertEqual((r.start_cell, r.goal_cell), (GOAL, (1, 4)))
        self.assertIsNone(r.waypoint)
        self.assertEqual(r.state, "ready")

    def test_nothing_moves_before_ok(self):
        r = parked_runner()
        r.set_goal((1, 4))
        self.tick(r)
        self.assertEqual(r.link.commands, [])
        self.assertIsNone(r.leg)
        self.assertEqual(r.state, "ready")

    def test_ok_drives_the_planned_route(self):
        r = parked_runner()
        r.set_goal((1, 4))
        r.confirm_ok()
        self.assertEqual(r.state, "running")
        self.tick(r)
        self.assertEqual(r.leg.from_cell, GOAL)
        self.assertIn(r.leg.to_cell, r.plan_path[1:])
        self.assertEqual([c[0] for c in r.link.commands], ["goto"])

    def test_repick_while_ready_keeps_the_kart_cell_as_start(self):
        r = parked_runner()
        r.set_goal((1, 4))
        self.assertTrue(r.set_goal((3, 1)))
        self.assertEqual((r.start_cell, r.goal_cell), (GOAL, (3, 1)))
        self.assertEqual(r.plan_path[0], GOAL)
        self.assertEqual(r.state, "ready")

    def test_wall_or_kart_cell_is_refused(self):
        r = parked_runner(walls={(2, 2)})
        self.assertFalse(r.set_goal((2, 2)))
        self.assertFalse(r.set_goal(GOAL))
        self.assertEqual(r.state, "idle")
        self.assertEqual(r.goal_cell, GOAL)

    def test_ok_without_a_route_does_not_start(self):
        # (1, 3) walled in on every side.
        r = parked_runner(walls={(0, 3), (2, 3), (1, 2), (1, 4)})
        self.assertTrue(r.set_goal((1, 3)))
        self.assertEqual(r.plan_path, [])
        r.confirm_ok()
        self.assertEqual(r.state, "ready")
        self.tick(r)
        self.assertEqual(r.link.commands, [])

    def test_live_maze_change_while_ready_replans(self):
        r = parked_runner()
        r.set_goal((1, 4))
        r.view.mark_obstacle((1, 2))       # sensor blocks the route on display
        r._live_revision += 1
        self.tick(r, frames=1)
        self.assertNotIn((1, 2), r.plan_path)
        self.assertEqual((r.plan_path[0], r.plan_path[-1]), (GOAL, (1, 4)))
        self.assertEqual(r.state, "ready")


if __name__ == "__main__":
    unittest.main()
