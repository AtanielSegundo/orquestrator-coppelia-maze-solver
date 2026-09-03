def rgb_to_norm(t):
    """(0-255, 0-255, 0-255) -> (0-1, 0-1, 0-1), for sim.setShapeColor()."""
    return tuple(c / 255 for c in t)
