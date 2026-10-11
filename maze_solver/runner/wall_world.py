"""
The maze as the camera model sees it: axis-aligned boxes on a floor.

Built from the ORIGINAL walls only, for the same reason as range_fix's trust
boundary: discovered obstacles are a belief derived from the pose, so letting
them explain an image would let a mislocalised kart confirm its own invention.
They are kept separately (`obstacles`) and are only ever used to VETO columns
that look through them -- they may remove evidence, never supply it.

`bounds` is the maze's outer rectangle. tcc.maze has no perimeter walls, so a
camera at a border cell looks out into the room; beyond `bounds` the model says
"unknown", not "floor".
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

import Coppelia.helpers as cop


class WallWorld(NamedTuple):
    walls: np.ndarray            # (N, 4) xmin, xmax, ymin, ymax
    bounds: np.ndarray           # (4,)   xmin, xmax, ymin, ymax
    obstacles: np.ndarray        # (M, 4)
    box_height: float


def _cell_box(origin, cell_length, cell, inset):
    x, y, _ = cop.cell_to_world(origin, cell_length, cell)
    h = cell_length / 2.0 - inset
    return (x - h, x + h, y - h, y + h)


def build_wall_world(view, origin, cell_length: float, inset: float = 0.0,
                     box_height: float = None) -> WallWorld:
    """`inset` shrinks every wall box on all sides -- boxes smaller than the
    cell they stand in, as the calibration may measure."""
    walls = [_cell_box(origin, cell_length, (r, c), inset)
             for r in range(view.rows) for c in range(view.cols)
             if view.is_original_wall((r, c))]
    obstacles = [_cell_box(origin, cell_length, cell, inset)
                 for cell in sorted(getattr(view, "obstacles", ()) or ())
                 if not view.is_original_wall(cell)]
    ox, oy, _ = origin
    half = cell_length / 2.0
    bounds = np.array([ox - half, ox + (view.cols - 0.5) * cell_length,
                       oy - (view.rows - 0.5) * cell_length, oy + half])
    return WallWorld(np.array(walls, dtype=float).reshape(-1, 4), bounds,
                     np.array(obstacles, dtype=float).reshape(-1, 4),
                     float(box_height if box_height is not None else cell_length))


def boxes_near(boxes: np.ndarray, x: float, y: float, radius: float) -> np.ndarray:
    """Boxes whose rectangle comes within `radius` of (x, y) -- the only ones a
    camera there can see, which keeps every raycast short."""
    if len(boxes) == 0:
        return boxes
    dx = np.maximum(0.0, np.maximum(boxes[:, 0] - x, x - boxes[:, 1]))
    dy = np.maximum(0.0, np.maximum(boxes[:, 2] - y, y - boxes[:, 3]))
    return boxes[np.hypot(dx, dy) <= radius]
