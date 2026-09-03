import math


def compose(a, b):
    """Pose a applied to pose b: world = a (+) b. Both are (x, y, theta),
    theta in radians CCW from +X."""
    ca, sa = math.cos(a[2]), math.sin(a[2])
    return (a[0] + ca * b[0] - sa * b[1],
            a[1] + sa * b[0] + ca * b[1],
            a[2] + b[2])
