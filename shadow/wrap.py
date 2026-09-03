import math


def wrap(t):
    """Wrap an angle (radians) to (-pi, pi]."""
    return (t + math.pi) % (2 * math.pi) - math.pi
