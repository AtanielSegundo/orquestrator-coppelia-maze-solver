"""
Sim pose mirror, anchored to whatever pose `base` had at construction time.
Anchor it to the kart's real starting cell by calling
Coppelia.helpers.set_planar_pose(sim, base, x, y, theta) BEFORE constructing
this, so `initial` below captures that placement.
"""
import math

from Coppelia.helpers.local_bounding_box import local_bounding_box

from shadow.compose import compose
from shadow.inverse import inverse


class SimMirror:
    def __init__(self, sim, base, motor_joints, wheel, center_offset_local=(0.0, 0.0)):
        self.sim = sim
        self.base = base
        self.left_joints = [motor_joints[0], motor_joints[2]]
        self.right_joints = [motor_joints[1], motor_joints[3]]
        self.r = wheel.radius

        # Initial world matrix of the kart -> anchor frame (x0, y0, yaw0).
        m0 = sim.getObjectMatrix(base, sim.handle_world)
        x0, y0, self.z0 = m0[3], m0[7], m0[11]
        self.yaw0 = math.atan2(m0[4], m0[0])

        # Local-frame vector from the object's own origin to its geometric
        # center (bounding-box midpoint, plus an optional hand-tuned fudge --
        # see Coppelia/helpers/kart_center_offset.py) -- constant, independent
        # of world pose. `base`'s own origin isn't at its footprint's center
        # (e.g. kart_base's sits at a corner), so tracking/rotating the raw
        # origin would turn the model around that corner instead of in place,
        # and the tracked position (also fed to the Raylib mirror) would drift
        # further off the further the heading has turned from the start.
        # local_bounding_box() covers `base`'s whole model tree, since
        # kart_base itself is a dummy with no shape of its own -- see its
        # docstring. Same trick as Coppelia/drive.py's DiffDrive and
        # Coppelia/helpers/set_planar_pose.py -- keep `center_offset_local`
        # the same in both, or the initial placement and the ongoing tracking
        # will disagree.
        bb_min_x, bb_max_x, bb_min_y, bb_max_y = local_bounding_box(sim, base)
        self.cx_local = (bb_min_x + bb_max_x) / 2.0 + center_offset_local[0]
        self.cy_local = (bb_min_y + bb_max_y) / 2.0 + center_offset_local[1]

        # `initial` is the CENTER's pose (origin + rotated local offset), not
        # the raw origin's -- world_pose()/apply_world_pose() track and place
        # the center throughout; only apply_world_pose() converts back to the
        # origin, right before writing the object matrix.
        c, s = math.cos(self.yaw0), math.sin(self.yaw0)
        cx0 = x0 + (c * self.cx_local - s * self.cy_local)
        cy0 = y0 + (s * self.cx_local + c * self.cy_local)
        self.initial = (cx0, cy0, self.yaw0)

        # base_matrix = m0 with the initial yaw removed, so the mesh keeps any
        # pitch/roll/structure while we drive yaw ourselves (same trick as
        # DiffDrive.set_pose).
        c, s = math.cos(-self.yaw0), math.sin(-self.yaw0)
        rz_inv = [c, -s, 0.0, 0.0,
                  s,  c, 0.0, 0.0,
                  0.0, 0.0, 1.0, 0.0]
        self.base_matrix = sim.multiplyMatrices(rz_inv, m0)

        self._wheel_angle = 0.0

    def shift_anchor(self, dx: float, dy: float) -> None:
        """Translate the anchor by (dx, dy) in the WORLD frame.

        `world_pose()` is compose(initial, rel), and compose applies `initial`'s
        rotation to `rel` before adding `initial`'s translation -- so moving the
        anchor's translation moves every pose it produces by exactly the same
        world vector, leaving heading and all relative motion untouched.

        This is how the 1-D range correction lands (see runner/range_fix.py).
        Correcting the ANCHOR rather than patching each pose is what makes the
        fix persistent: the statement being made is "our guess of where the kart
        started was off by this much", which stays true for every later odometry
        sample. Patching the output pose would be undone by the next telemetry
        frame, since that frame recomputes it from the unchanged anchor.
        """
        x0, y0, th0 = self.initial
        self.initial = (x0 + dx, y0 + dy, th0)

    def world_pose(self, odom_pose, odom_ref):
        """Odom telemetry pose -> world (x, y, theta) of the kart's CENTER,
        relative to the first odom sample (odom_ref) and anchored to the
        initial sim pose. Pure computation -- doesn't touch the sim, so
        callers that only need the pose (e.g. to drive the Raylib mirror or
        find the kart's grid cell) don't have to write it to Coppelia via
        mirror()."""
        rel = compose(inverse(odom_ref), odom_pose)      # motion since we started
        return compose(self.initial, rel)                # into the sim world frame

    def apply_world_pose(self, wx, wy, wtheta):
        """(wx, wy, wtheta): world pose of the kart's CENTER. Converts back to
        where the object's own origin must sit for the center to land there
        (see set_planar_pose's docstring for why)."""
        c, s = math.cos(wtheta), math.sin(wtheta)
        rz = [c, -s, 0.0, 0.0,
              s,  c, 0.0, 0.0,
              0.0, 0.0, 1.0, 0.0]
        m = self.sim.multiplyMatrices(rz, self.base_matrix)
        m[3] = wx - (c * self.cx_local - s * self.cy_local)
        m[7] = wy - (s * self.cx_local + c * self.cy_local)
        m[11] = self.z0
        self.sim.setObjectMatrix(self.base, self.sim.handle_world, m)

    def mirror(self, odom_pose, odom_ref):
        """Place the sim kart from a telemetry odom pose."""
        wx, wy, wth = self.world_pose(odom_pose, odom_ref)
        self.apply_world_pose(wx, wy, wth)

    def spin_wheels(self, v_left, v_right, dt):
        da_l = (v_left / self.r) * dt
        da_r = (v_right / self.r) * dt
        for j in self.left_joints:
            self.sim.setJointPosition(j, (self.sim.getJointPosition(j) + da_l)
                                      % (2 * math.pi))
        for j in self.right_joints:
            self.sim.setJointPosition(j, (self.sim.getJointPosition(j) + da_r)
                                      % (2 * math.pi))
