from typing import Tuple


def cell_to_world(origin: Tuple[float, float, float],
                  cell_cube_size: float,
                  cell: Tuple[int, int]) -> Tuple[float, float, float]:
    """col grows with +X, row grows with -Y (row- = "north" = +Y), so the
    world frame is right-handed and matches the real floor seen from above --
    see Coppelia/helpers/maze_origin.py."""
    ox, oy, oz = origin
    r, c = cell
    return ox + c * cell_cube_size, oy - r * cell_cube_size, oz
