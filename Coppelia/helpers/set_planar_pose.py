import math

from Coppelia.sim import ObjectHandle
from Coppelia.helpers.local_bounding_box import local_bounding_box


def set_planar_pose(sim, handle: ObjectHandle,
                    x: float, y: float, theta: float,
                    z: float = None,
                    center_offset_local=(0.0, 0.0)) -> None:
    """Move `handle` so its GEOMETRIC CENTER (not its own coordinate origin)
    ends up at world (x, y) with world yaw `theta` (radians, CCW from +X),
    keeping whatever pitch/roll its mesh already had and its current Z
    (unless `z` is given).

    `handle`'s own origin is not necessarily at its footprint's center (e.g.
    kart_base's sits at a corner) -- placing the raw origin at (x, y) would
    put the model visibly off-center on its cell and, worse, make any later
    in-place rotation swing around that corner instead of the body's middle.
    So this reads the bounding box of `handle`'s whole model tree (same trick
    as Coppelia/drive.py's DiffDrive and shadow.sim_mirror.SimMirror, but
    covering descendants too -- kart_base itself is a dummy with no shape of
    its own, so its OWN bbox alone is degenerate) to find the fixed
    body-frame offset from the origin to the true center, rotates that
    offset by the REQUESTED heading, and places the origin far enough back
    that the center lands exactly on (x, y).

    `center_offset_local` (dx, dy) is an extra hand-tuned nudge added on top
    of that auto-computed offset, in `handle`'s own body frame (see
    Coppelia/helpers/kart_center_offset.py) -- for when an accessory skews
    the aggregated bounding box away from the body's true visual center in
    just one axis. Zero by default, i.e. trust the bbox alone."""
    m0 = sim.getObjectMatrix(handle, sim.handle_world)
    yaw0 = math.atan2(m0[4], m0[0])
    c0, s0 = math.cos(-yaw0), math.sin(-yaw0)
    rz_inv = [c0, -s0, 0.0, 0.0,
              s0,  c0, 0.0, 0.0,
              0.0, 0.0, 1.0, 0.0]
    base_matrix = sim.multiplyMatrices(rz_inv, m0)

    # Local-frame vector from the object's own origin to its geometric
    # center (bounding-box midpoint, plus the hand-tuned fudge) -- constant,
    # independent of world pose.
    bb_min_x, bb_max_x, bb_min_y, bb_max_y = local_bounding_box(sim, handle)
    cx_local = (bb_min_x + bb_max_x) / 2.0 + center_offset_local[0]
    cy_local = (bb_min_y + bb_max_y) / 2.0 + center_offset_local[1]

    c1, s1 = math.cos(theta), math.sin(theta)
    rz = [c1, -s1, 0.0, 0.0,
          s1,  c1, 0.0, 0.0,
          0.0, 0.0, 1.0, 0.0]
    m = sim.multiplyMatrices(rz, base_matrix)
    # (x, y) is where the CENTER must land; the origin sits
    # center - R(theta) @ (cx_local, cy_local) for that to hold.
    m[3] = x - (c1 * cx_local - s1 * cy_local)
    m[7] = y - (s1 * cx_local + c1 * cy_local)
    m[11] = base_matrix[11] if z is None else z
    sim.setObjectMatrix(handle, sim.handle_world, m)
