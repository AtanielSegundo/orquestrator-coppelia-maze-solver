from typing import Tuple

# ---------------------------------------------------------------------------
# Grid <-> world plane mapping, shared by render_maze() and anyone else who
# needs to place an object on a maze cell (e.g. the CoppeliaSim maze-solving
# runner). A cell (r, c) sits at world (ox + c*cell_cube_size,
# oy - r*cell_cube_size): col grows with +X, row grows with -Y. The raylib
# MazeView draws row 0 at the TOP of the screen (row- = "north"), so north
# must be +Y for the world frame to be right-handed -- i.e. the same floor
# the real kart drives on, seen from above, with theta CCW from +X. Mapping
# row to +Y instead mirrors the maze: the sim stays self-consistent, but the
# real kart's turns come out swapped on the axis it did not start along.
# ---------------------------------------------------------------------------
def maze_origin(rows: int, cols: int,
                center: Tuple[float, float, float],
                cell_cube_size: float) -> Tuple[float, float, float]:
    ox, oy, oz = center
    ox -= (cols // 2) * cell_cube_size
    oy += (rows // 2) * cell_cube_size
    return ox, oy, oz
