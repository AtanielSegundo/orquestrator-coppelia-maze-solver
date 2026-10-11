#!/usr/bin/python3
"""
SimMirror.reanchor -- the camera fix's way into the belief.

Unlike `shift_anchor` it corrects the HEADING too, and that is the property
worth pinning: after a yaw re-anchor the current pose must come out exactly as
corrected, and every later odometry increment must be laid down along the
corrected heading (the persistent fix for a turn that over- or under-rotated).

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

from shadow import SimMirror, compose                             # noqa: E402


def mirror_at(initial):
    """A SimMirror with only the pose algebra -- no simulator behind it."""
    m = SimMirror.__new__(SimMirror)
    m.initial = tuple(initial)
    return m


def close(a, b, tol=1e-9):
    return all(abs(x - y) < tol for x, y in zip(a, b))


class TestReanchor(unittest.TestCase):
    REF = (0.4, -0.2, 0.3)            # first odometry sample, arbitrary on purpose

    def test_current_pose_becomes_the_corrected_one(self):
        m = mirror_at((1.0, 2.0, 0.5))
        odom = compose(self.REF, (0.6, 0.1, 1.2))
        corrected = (1.234, 1.987, 0.5 + 1.2 + math.radians(2.0))
        m.reanchor(SimMirror.relative(odom, self.REF), corrected)
        self.assertTrue(close(m.world_pose(odom, self.REF), corrected))

    def test_translation_only_equals_shift_anchor(self):
        a = mirror_at((1.0, 2.0, 0.5))
        b = mirror_at((1.0, 2.0, 0.5))
        odom = compose(self.REF, (0.3, -0.2, 0.7))
        wx, wy, wth = a.world_pose(odom, self.REF)
        a.reanchor(SimMirror.relative(odom, self.REF), (wx + 0.03, wy - 0.01, wth))
        b.shift_anchor(0.03, -0.01)
        self.assertTrue(close(a.initial, b.initial))

    def test_odom_target_stays_the_exact_inverse(self):
        m = mirror_at((1.0, 2.0, 0.5))
        odom = compose(self.REF, (0.6, 0.1, 1.2))
        m.reanchor(SimMirror.relative(odom, self.REF), (1.1, 2.3, 1.9))
        ox, oy, oth = m.odom_target(self.REF, 1.5, 2.6, 0.7)
        back = m.world_pose((ox, oy, math.radians(oth)), self.REF)
        self.assertTrue(close(back, (1.5, 2.6, 0.7)))

    def test_future_motion_follows_the_corrected_heading(self):
        """Drive 0.30 m straight after a +3 deg yaw correction: the believed
        displacement must point along the corrected heading."""
        m = mirror_at((0.0, 0.0, 0.0))
        here = compose(self.REF, (0.0, 0.0, 0.0))
        before = m.world_pose(here, self.REF)
        dpsi = math.radians(3.0)
        m.reanchor(SimMirror.relative(here, self.REF),
                   (before[0], before[1], before[2] + dpsi))
        later = compose(here, (0.30, 0.0, 0.0))     # straight ahead in odometry
        x, y, th = m.world_pose(later, self.REF)
        self.assertAlmostEqual(math.atan2(y - before[1], x - before[0]),
                               before[2] + dpsi, places=9)
        self.assertAlmostEqual(math.hypot(x - before[0], y - before[1]), 0.30, places=9)


if __name__ == "__main__":
    unittest.main()
