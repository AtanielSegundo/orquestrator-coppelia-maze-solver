from typing import Tuple

# ---------------------------------------------------------------------------
# Grid <-> world plane mapping, shared by render_maze() and anyone else who
# needs to place an object on a maze cell (e.g. the CoppeliaSim maze-solving
# runner). A cell (r, c) sits at world (ox + c*cell_cube_size,
# oy + r*cell_cube_size): col grows with +X, row grows with +Y, matching the
# raylib MazeView's (row, col) -> (screen_x, screen_y) mapping so both views
# agree on where a given cell physically is.
# ---------------------------------------------------------------------------
def maze_origin(rows: int, cols: int,
                center: Tuple[float, float, float],
                cell_cube_size: float) -> Tuple[float, float, float]:
    ox, oy, oz = center
    ox -= (cols // 2) * cell_cube_size
    oy -= (rows // 2) * cell_cube_size
    return ox, oy, oz
