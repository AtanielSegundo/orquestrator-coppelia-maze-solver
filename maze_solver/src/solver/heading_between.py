from typing import Optional, Tuple

from src.solver.constants import DIRECTIONS


def heading_between(a: Tuple[int, int], b: Tuple[int, int]) -> Optional[float]:
    delta = (b[0] - a[0], b[1] - a[1])
    for dr, dc, angle in DIRECTIONS:
        if delta == (dr, dc):
            return angle
    return None
