import math


def inverse(a):
    """Inverse of pose a = (x, y, theta), theta in radians CCW from +X."""
    ca, sa = math.cos(a[2]), math.sin(a[2])
    return (-(ca * a[0] + sa * a[1]),
            -(-sa * a[0] + ca * a[1]),
            -a[2])
