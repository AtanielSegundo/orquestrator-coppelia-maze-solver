import math

from shadow.wrap import wrap


def grid_heading_to_world_theta(deg):
    """maze_solver's DIRECTIONS heading (0=col+, 90=row-, 180=col-, 270=row+,
    CCW degrees, 90 = screen "up"/north) -> world/odom theta (radians, CCW
    from +X). The world frame maps col to +X and row to -Y (see
    Coppelia/helpers/maze_origin.py), so north (row-1) is +Y and the two
    conventions coincide: grid heading 90 is world theta +90."""
    return wrap(math.radians(deg))
