"""
2-D ray casting on the floor plane, vectorised over many origins x many
directions.

Why 2-D is exact here: a floor point Q is hidden from the lens C exactly when
the segment from C's FOOT (the floor point under C) to Q crosses a box -- the
3-D sight line descends from the lens height to 0 over that segment, so wherever
its shadow crosses a box footprint the line itself is below the box top, as
long as the boxes are taller than the camera (0.30 m boxes, 0.10 m lens).
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _safe_inv(d: np.ndarray) -> np.ndarray:
    return 1.0 / np.where(np.abs(d) < _EPS, np.copysign(_EPS, d), d)


def first_box_hits(ox, oy, dx, dy, boxes: np.ndarray) -> np.ndarray:
    """Distance along each ray to its first entry into any box, +inf if none.

    `ox, oy` broadcast against `dx, dy` (e.g. origins (B, 1), directions
    (1, M) -> (B, M)). Directions must be unit length for the result to be a
    distance. An origin inside a box gets 0 for that box.
    """
    ox = np.asarray(ox, dtype=float)
    oy = np.asarray(oy, dtype=float)
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    shape = np.broadcast(ox, oy, dx, dy).shape
    best = np.full(shape, np.inf)
    if len(boxes) == 0:
        return best
    ix = _safe_inv(dx)
    iy = _safe_inv(dy)
    for xmin, xmax, ymin, ymax in boxes:
        t1 = (xmin - ox) * ix
        t2 = (xmax - ox) * ix
        t3 = (ymin - oy) * iy
        t4 = (ymax - oy) * iy
        tmin = np.maximum(np.minimum(t1, t2), np.minimum(t3, t4))
        tmax = np.minimum(np.maximum(t1, t2), np.maximum(t3, t4))
        enter = np.maximum(tmin, 0.0)
        hit = tmax >= enter
        np.minimum(best, np.where(hit, enter, np.inf), out=best)
    return best


def exit_distance(ox, oy, dx, dy, rect) -> np.ndarray:
    """Distance along each ray from an origin to where it leaves `rect`
    (xmin, xmax, ymin, ymax). 0 for an origin already outside."""
    xmin, xmax, ymin, ymax = rect
    ox = np.asarray(ox, dtype=float)
    oy = np.asarray(oy, dtype=float)
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(dx > _EPS, (xmax - ox) / dx,
                      np.where(dx < -_EPS, (xmin - ox) / dx, np.inf))
        ty = np.where(dy > _EPS, (ymax - oy) / dy,
                      np.where(dy < -_EPS, (ymin - oy) / dy, np.inf))
    t = np.maximum(np.minimum(tx, ty), 0.0)
    inside = (ox >= xmin) & (ox <= xmax) & (oy >= ymin) & (oy <= ymax)
    return np.where(inside, t, 0.0)
