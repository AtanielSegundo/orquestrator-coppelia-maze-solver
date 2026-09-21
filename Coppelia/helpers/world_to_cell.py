from typing import Tuple


def world_to_cell(origin: Tuple[float, float, float],
                  cell_cube_size: float,
                  x: float, y: float) -> Tuple[float, float]:
    """Inverse of cell_to_world(). Returns FLOAT (row, col) so a caller can
    tell how far off-center a world position is, before rounding to a cell."""
    ox, oy, _ = origin
    return (oy - y) / cell_cube_size, (x - ox) / cell_cube_size
