"""
A brute-force, per-pixel 3-D ray caster of the maze: floor, wall boxes of real
height, and the maze edge.

Independent of the analytic contact model on purpose. That model reasons on the
floor plane from the camera foot; this one traces every pixel's ray in 3-D
against the boxes' six faces' worth of slabs and the floor. The two are
compared in the tests, which is what pins the analytic model's claims (columns
that are not vertical planes, corners occluding contacts) -- a model must never
be tested against itself.

It also produces the per-pixel PREDICTED labels the colour model trains on
(camera_link/floor_segmenter.py), and the synthetic frames of the tests.
"""
from __future__ import annotations

import math

import numpy as np

from runner.wall_world import WallWorld

FLOOR = 0
WALL = 1
UNKNOWN = 2          # above the horizon, outside the maze, beyond the range


def render_labels(pinhole, pose, world: WallWorld, max_range: float = None,
                  stride: int = 1, extra_boxes: np.ndarray = None) -> np.ndarray:
    """Label image (H/stride, W/stride) int8 of what the camera of `pinhole`
    sees with the kart at `pose` (x, y, theta).

    `extra_boxes` are rendered as walls too (discovered obstacles are real
    boxes, so a frame of the real scene shows them). `max_range` (from the
    camera foot) turns further floor into UNKNOWN.
    """
    H, W = pinhole.height, pinhole.width
    us = np.arange(0, W, stride, dtype=float)
    vs = np.arange(0, H, stride, dtype=float)
    uu, vv = np.meshgrid(us, vs)
    d = pinhole.rays(uu, vv)                                   # centre frame
    x, y, th = pose
    c, s = math.cos(th), math.sin(th)
    dwx = c * d[..., 0] - s * d[..., 1]
    dwy = s * d[..., 0] + c * d[..., 1]
    dwz = d[..., 2]
    cx, cy, cz = pinhole.C
    ox = x + c * cx - s * cy
    oy = y + s * cx + c * cy
    oz = cz

    with np.errstate(divide="ignore", invalid="ignore"):
        t_floor = np.where(dwz < -1e-12, -oz / dwz, np.inf)

    boxes = world.walls
    if extra_boxes is not None and len(extra_boxes):
        boxes = np.vstack([boxes, extra_boxes])
    t_wall = np.full(uu.shape, np.inf)
    eps = 1e-12
    ix = 1.0 / np.where(np.abs(dwx) < eps, np.copysign(eps, dwx), dwx)
    iy = 1.0 / np.where(np.abs(dwy) < eps, np.copysign(eps, dwy), dwy)
    iz = 1.0 / np.where(np.abs(dwz) < eps, np.copysign(eps, dwz), dwz)
    zmin, zmax = 0.0, world.box_height
    for xmin, xmax, ymin, ymax in boxes:
        t1, t2 = (xmin - ox) * ix, (xmax - ox) * ix
        t3, t4 = (ymin - oy) * iy, (ymax - oy) * iy
        t5, t6 = (zmin - oz) * iz, (zmax - oz) * iz
        tmin = np.maximum(np.maximum(np.minimum(t1, t2), np.minimum(t3, t4)),
                          np.minimum(t5, t6))
        tmax = np.minimum(np.minimum(np.maximum(t1, t2), np.maximum(t3, t4)),
                          np.maximum(t5, t6))
        enter = np.maximum(tmin, 0.0)
        hit = tmax >= enter
        np.minimum(t_wall, np.where(hit, enter, np.inf), out=t_wall)

    labels = np.full(uu.shape, UNKNOWN, dtype=np.int8)
    wall = np.isfinite(t_wall) & (t_wall <= t_floor)
    labels[wall] = WALL
    floor = ~wall & np.isfinite(t_floor)
    qx = ox + dwx * t_floor
    qy = oy + dwy * t_floor
    xmin, xmax, ymin, ymax = world.bounds
    with np.errstate(invalid="ignore"):
        inside = (qx >= xmin) & (qx <= xmax) & (qy >= ymin) & (qy <= ymax)
    floor &= inside
    if max_range is not None:
        fx = x + c * pinhole.C[0] - s * pinhole.C[1]
        fy = y + s * pinhole.C[0] + c * pinhole.C[1]
        with np.errstate(invalid="ignore"):
            floor &= np.hypot(qx - fx, qy - fy) <= max_range
    labels[floor] = FLOOR
    return labels


def upsample_labels(labels: np.ndarray, stride: int, shape) -> np.ndarray:
    """Nearest-neighbour back to full resolution."""
    if stride == 1:
        return labels
    up = np.repeat(np.repeat(labels, stride, axis=0), stride, axis=1)
    return up[:shape[0], :shape[1]]
