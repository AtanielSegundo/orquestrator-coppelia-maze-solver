import math


def apply_deadzone(value, dz=0.12):
    if abs(value) < dz:
        return 0.0
    return (value - math.copysign(dz, value)) / (1 - dz)
