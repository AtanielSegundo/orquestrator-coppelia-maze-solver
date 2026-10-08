#!/usr/bin/python3
"""
Unit tests for the world -> odometry conversion (SimMirror.odom_target).

Why this needs its own test: the kart's `goto` closes its loop in the ODOMETRY
frame, while every maze target is a world-frame cell center. Handing world
coordinates straight to `goto` is not a small approximation, it is a different
destination -- and the discrepancy is neither constant nor small (the anchor
carries the kart's start pose, including its heading, so a rotated anchor turns
the whole target field). A silent mix-up here would look exactly like a
badly-tuned PID, so the round trip is pinned.

SimMirror's constructor needs a live CoppeliaSim scene, so these tests exercise
the transform through a stub that holds only the fields odom_target touches
(`initial`), which is the entire state the conversion depends on. The arithmetic
under test is the real method, called unbound.

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

from shadow.sim_mirror import SimMirror                           # noqa: E402


class Anchored:
    """Just enough of a SimMirror for the frame arithmetic: the anchor, plus
    the two real methods under test, borrowed unbound."""

    def __init__(self, initial):
        self.initial = initial

    odom_target = SimMirror.odom_target
    world_pose = SimMirror.world_pose
    shift_anchor = SimMirror.shift_anchor


ANCHOR = (1.3, -0.7, math.radians(90.0))     # kart started facing north
ODOM_REF = (0.05, -0.02, math.radians(5.0))  # first telemetry sample, non-zero


class TestOdomTarget(unittest.TestCase):

    def assert_pose(self, got, want, places=9):
        self.assertAlmostEqual(got[0], want[0], places=places)
        self.assertAlmostEqual(got[1], want[1], places=places)

    def test_round_trip_is_exact(self):
        """odom_target is the exact inverse of world_pose, which is the only
        property that makes the two safe to use together."""
        m = Anchored(ANCHOR)
        for odom_pose in [(0.4, 0.1, math.radians(33.0)),
                          (-1.2, 2.0, math.radians(-110.0)),
                          (0.0, 0.0, 0.0)]:
            w = m.world_pose(odom_pose, ODOM_REF)
            back = m.odom_target(ODOM_REF, w[0], w[1], w[2])
            self.assert_pose(back, odom_pose)
            self.assertAlmostEqual(math.radians(back[2]), odom_pose[2], places=9)

    def test_commanding_the_odom_target_lands_on_the_world_target(self):
        """The claim the runner relies on: a kart that reaches the commanded
        odom pose is standing on the world point that was asked for."""
        m = Anchored(ANCHOR)
        want = (2.05, -1.45)
        ox, oy, oth = m.odom_target(ODOM_REF, want[0], want[1], math.pi)
        landed = m.world_pose((ox, oy, math.radians(oth)), ODOM_REF)
        self.assert_pose(landed, want)
        self.assertAlmostEqual(math.degrees(landed[2]), 180.0, places=7)

    def test_world_coordinates_sent_raw_would_be_wrong(self):
        """Guards against anyone 'simplifying' the conversion away: with a
        rotated anchor the same numbers denote a point metres off."""
        m = Anchored(ANCHOR)
        want = (2.05, -1.45)
        naive = m.world_pose((want[0], want[1], 0.0), ODOM_REF)
        self.assertGreater(math.hypot(naive[0] - want[0], naive[1] - want[1]), 1.0)

    def test_theta_none_passes_through(self):
        """No final heading asked for -> none commanded, so `goto` leaves the
        heading where the drive ends."""
        m = Anchored(ANCHOR)
        _, _, oth = m.odom_target(ODOM_REF, 1.0, 1.0, None)
        self.assertIsNone(oth)

    def test_follows_an_anchor_correction(self):
        """range_fix shifts the anchor mid-run. The conversion must be computed
        from the anchor as it stands, so a target built after a correction aims
        at the corrected place -- this is why the runner rebuilds the target per
        leg instead of caching it with the plan."""
        m = Anchored(ANCHOR)
        want = (2.05, -1.45)
        before = m.odom_target(ODOM_REF, want[0], want[1], 0.0)

        m.shift_anchor(0.04, -0.03)
        after = m.odom_target(ODOM_REF, want[0], want[1], 0.0)
        self.assertNotAlmostEqual(before[0], after[0], places=6)

        # And it is still the SAME world point under the new anchor.
        landed = m.world_pose((after[0], after[1], math.radians(after[2])),
                              ODOM_REF)
        self.assert_pose(landed, want)

    def test_identity_anchor_is_a_no_op(self):
        """Sanity: with the anchor and reference both at the origin the two
        frames coincide, so the conversion must not perturb anything."""
        m = Anchored((0.0, 0.0, 0.0))
        got = m.odom_target((0.0, 0.0, 0.0), 1.25, -0.5, math.radians(90.0))
        self.assert_pose(got, (1.25, -0.5))
        self.assertAlmostEqual(got[2], 90.0, places=9)


if __name__ == "__main__":
    unittest.main()
