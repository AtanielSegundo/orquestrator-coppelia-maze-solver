"""
Where the camera pose fix v2 is worth its time: map-only judgement of a cell,
from the heading the kart arrives with, and the look-ahead that postpones a
due fix to such a cell -- the camera's NEXT_BEST_CLOSEST_AS_REFERENCE.

A good cell (`camera_cell`):
* the cell AHEAD is open: the lens sits ~13 cm ahead of the kart's centre, so
  a wall in the adjacent cell would fill the frame from 2 cm away;
* obstacles IN FRONT: at least CAMERA_V2_MIN_FRONT_OBSTACLES original walls
  among the cell ahead's two sides and the three cells two ahead -- what the
  centre pan's left and right half planes see, and what makes the
  longitudinal offset observable at all (a corridor's side walls alone slide
  past without changing);
* a CORRIDOR-like stretch (CAMERA_V2_REQUIRE_CORRIDOR): walls on both sides of
  the kart's cell or of the cell ahead -- what the side pans see, close and
  converging, the best evidence of yaw.

Original walls only, like every camera model here: a discovered obstacle is a
belief from the pose, not something to localise against. Cells outside the
maze count as open (tcc.maze has no perimeter: the room is out there).
"""
from __future__ import annotations

import math
from typing import NamedTuple, Optional, Tuple

from shadow import grid_heading_to_world_theta
from src.solver import heading_between

from runner.constants import (CAMERA_V2_MIN_FRONT_OBSTACLES,
                              CAMERA_V2_REQUIRE_CORRIDOR)


class CellQuality(NamedTuple):
    ok: bool
    front_obstacles: int
    corridor: bool
    reason: str


def _steps(theta: float) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """(forward, left) one-cell grid steps (drow, dcol) for a world heading
    snapped to the grid. World x = col * L, y = -row * L."""
    c, s = round(math.cos(theta)), round(math.sin(theta))
    return (-s, c), (-c, -s)


def camera_cell(view, cell, theta: float,
                min_front: int = CAMERA_V2_MIN_FRONT_OBSTACLES,
                require_corridor: bool = CAMERA_V2_REQUIRE_CORRIDOR) -> CellQuality:
    (fr, fc), (lr, lc) = _steps(theta)
    r, c = cell

    def rel(f, l):
        return (r + f * fr + l * lr, c + f * fc + l * lc)

    def wall(cc):
        return view.in_bounds(cc) and view.is_original_wall(cc)

    ahead = rel(1, 0)
    front = [rel(1, 1), rel(1, -1), rel(2, 1), rel(2, 0), rel(2, -1)]
    n_front = sum(1 for cc in front if wall(cc))
    corridor = ((wall(rel(0, 1)) and wall(rel(0, -1)))
                or (wall(rel(1, 1)) and wall(rel(1, -1))))
    if not view.in_bounds(ahead) or view.is_wall(ahead):
        return CellQuality(False, n_front, corridor, f"{ahead} fechada na frente")
    if n_front < min_front:
        return CellQuality(False, n_front, corridor,
                           f"{n_front} obstaculo(s) na frente (minimo {min_front})")
    if require_corridor and not corridor:
        return CellQuality(False, n_front, corridor, "sem corredor")
    return CellQuality(True, n_front, corridor, "")


def find_camera_cell(view, path, index: int, max_cells: int) -> Optional[int]:
    """First plan index AHEAD of `index` (at most `max_cells` past it) that is a
    good camera cell, judged with the heading the kart arrives there with.
    None when there is none."""
    for i in range(index + 1, min(len(path), index + 1 + max_cells)):
        heading = heading_between(path[i - 1], path[i])
        if heading is None:
            continue
        if camera_cell(view, path[i], grid_heading_to_world_theta(heading)).ok:
            return i
    return None
