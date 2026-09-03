import math

from shadow.wrap import wrap


def grid_heading_to_world_theta(deg):
    """maze_solver's DIRECTIONS heading (0=col+, 90=row-, 180=col-, 270=row+,
    CCW degrees -- a "compass" convention borrowed from screen-space "up") ->
    world/odom theta (radians, CCW from +X, where +Y is the SAME axis the
    grid's row grows along -- see Coppelia/helpers/cell_to_world.py). The
    grid's 90=row- already assumes a flipped Y, so converting to a real X/Y
    plane negates the angle: moving "up" (row-1, grid heading 90) is world
    direction (0,-1), i.e. world theta -90, not +90."""
    return wrap(-math.radians(deg))
