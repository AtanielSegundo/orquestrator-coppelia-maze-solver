from typing import Tuple


def cell_to_world(origin: Tuple[float, float, float],
                  cell_cube_size: float,
                  cell: Tuple[int, int]) -> Tuple[float, float, float]:
    ox, oy, oz = origin
    r, c = cell
    return ox + c * cell_cube_size, oy + r * cell_cube_size, oz
