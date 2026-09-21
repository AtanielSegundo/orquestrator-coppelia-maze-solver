import math

from shadow.wrap import wrap


def world_theta_to_grid_heading(theta):
    """Inverse of grid_heading_to_world_theta() -- see its docstring."""
    return math.degrees(wrap(theta)) % 360.0
