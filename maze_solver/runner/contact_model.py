"""
The analytic forward model: where each column's floor contact SHOULD be, for
many candidate kart poses at once.

For a candidate pose, the camera foot lands at a world point F. Along each
bearing from F the maze is raycast once (`floor_visibility`): R is the distance
to the first wall box, E the distance to where the floor leaves the maze. A
sample of a column (stored as distance r and bearing phi from the foot, see
view_geometry) is

    visible floor     if r < R and r < E
    hidden (wall)     if r >= R          -> the column's contact is here
    outside the maze  if r >= E          -> the map cannot say what is there

The first hidden sample gives the contact, interpolated between it and the
sample before on r - R, so the predicted row is continuous in the pose.

Every column then becomes an INTERVAL of rows the measured contact may fall in
(rows grow downward, so smaller is further):

    HIT       [v, v]
    NEAR      [v_near, +inf)     wall already at the bottom of the band
    FREE      (-inf, v_far]      floor out to the range limit
    UNKNOWN   (-inf, v_exit]     floor until the maze edge, then anything

and the residual against the measured interval is the gap between the two
(zero when they overlap). One rule covers every combination -- a measured
"nothing up to the range limit" against a predicted contact just short of it
costs exactly the few rows between them, rather than a whole-column penalty --
so the cost stays continuous in the pose.

Discovered obstacles are not part of the model. Columns that look through one
at the believed pose are vetoed for every candidate (`vetoed_columns`), so they
neither help nor hurt.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from camera_link.contact_profile import FREE, HIT, INVALID, NEAR
from runner.floor_visibility import exit_distance, first_box_hits
from runner.wall_world import WallWorld, boxes_near

UNKNOWN = 3


def _first_true(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(index of first True along the last axis, any True)."""
    return np.argmax(mask, axis=-1), mask.any(axis=-1)


def _interp_rows(geom, idx, g):
    """Row where g (sample-wise, negative = visible) crosses zero, between
    sample idx-1 and idx. `idx` must be >= 1 where used."""
    k1 = np.maximum(idx - 1, 0)[..., None]
    k2 = idx[..., None]
    g1 = np.take_along_axis(g, k1, axis=-1)[..., 0]
    g2 = np.take_along_axis(g, k2, axis=-1)[..., 0]
    rows = np.broadcast_to(geom.rows, g.shape)
    v1 = np.take_along_axis(rows, k1, axis=-1)[..., 0]
    v2 = np.take_along_axis(rows, k2, axis=-1)[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(np.isfinite(g1) & np.isfinite(g2) & (g2 != g1),
                     g1 / (g1 - g2), 1.0)
    t = np.clip(np.nan_to_num(t, nan=1.0), 0.0, 1.0)
    return v1 + t * (v2 - v1)


def predict_intervals(geom, xs: np.ndarray, ys: np.ndarray, theta: float,
                      world: WallWorld, walls: np.ndarray = None):
    """Model intervals for candidate kart centres (xs[i], ys[i]) sharing the
    heading `theta`. Returns (lo, hi, kind), each (B, n_columns).

    `walls` overrides world.walls (a pre-filtered subset near the kart).
    """
    with np.errstate(invalid="ignore"):
        return _predict(geom, xs, ys, theta, world, walls)


def _predict(geom, xs, ys, theta, world, walls):
    xs = np.asarray(xs, dtype=float).reshape(-1)
    ys = np.asarray(ys, dtype=float).reshape(-1)
    walls = world.walls if walls is None else walls
    c, s = math.cos(theta), math.sin(theta)
    fx, fy = geom.foot
    Fx = xs + c * fx - s * fy                                   # (B,)
    Fy = ys + s * fx + c * fy

    beta = theta + geom.phi0 + geom.dphi * np.arange(geom.n_bins)
    dx = np.cos(beta)[None, :]
    dy = np.sin(beta)[None, :]
    R = first_box_hits(Fx[:, None], Fy[:, None], dx, dy, walls)          # (B, M)
    E = exit_distance(Fx[:, None], Fy[:, None], dx, dy, world.bounds)

    Rs = R[:, geom.bin_idx]                                     # (B, n, K)
    Es = E[:, geom.bin_idx]
    r = geom.r[None]
    valid = geom.valid[None]
    blocked = valid & (r >= Rs)
    outside = valid & (r >= Es)

    kb, has_b = _first_true(blocked)
    ke, has_e = _first_true(outside)
    wall_first = has_b & (~has_e | (kb <= ke))
    unknown = has_e & ~wall_first
    usable = np.isfinite(geom.v_far)[None, :]

    B = len(xs)
    n = geom.n_columns
    lo = np.full((B, n), np.nan)
    hi = np.full((B, n), np.nan)
    kind = np.full((B, n), INVALID, dtype=np.int8)

    near = usable & wall_first & (kb == 0)
    hit = usable & wall_first & (kb > 0)
    free = usable & ~has_b & ~has_e
    unk = usable & unknown & (ke > 0)

    if np.any(hit):
        v_hit = _interp_rows(geom, kb, r - Rs)
        lo[hit] = hi[hit] = v_hit[hit]
        kind[hit] = HIT
    lo[near] = geom.v_near
    hi[near] = np.inf
    kind[near] = NEAR
    vfar = np.broadcast_to(geom.v_far[None, :], (B, n))
    lo[free] = -np.inf
    hi[free] = vfar[free]
    kind[free] = FREE
    if np.any(unk):
        v_exit = _interp_rows(geom, ke, r - Es)
        lo[unk] = -np.inf
        hi[unk] = v_exit[unk]
        kind[unk] = UNKNOWN
    return lo, hi, kind


def predict_at(geom, pose, world: WallWorld):
    """Intervals at one pose (x, y, theta) -> (lo, hi, kind), each (n,)."""
    lo, hi, kind = predict_intervals(geom, [pose[0]], [pose[1]], pose[2], world)
    return lo[0], hi[0], kind[0]


def vetoed_columns(geom, pose, world: WallWorld, margin_m: float = 0.05) -> np.ndarray:
    """Columns that look through a discovered obstacle (inflated by
    `margin_m`, the size of pose error the veto has to survive) before
    reaching their predicted contact or the range limit. (n,) bool."""
    n = geom.n_columns
    if len(world.obstacles) == 0:
        return np.zeros(n, dtype=bool)
    inflated = world.obstacles + np.array([-margin_m, margin_m, -margin_m, margin_m])
    with_obs = world._replace(walls=np.vstack([world.walls, inflated]))
    _lo, hi_obs, kind_obs = predict_at(geom, pose, with_obs)
    _lo, hi_wall, kind_wall = predict_at(geom, pose, world)
    # The obstacle changed what the column would see -> it is in the way.
    changed = (kind_obs != kind_wall) | (np.abs(np.nan_to_num(hi_obs, posinf=1e9)
                                                - np.nan_to_num(hi_wall, posinf=1e9)) > 0.5)
    return changed


def interval_gaps(m_lo, m_hi, p_lo, p_hi) -> np.ndarray:
    """Gap between measured [m_lo, m_hi] and predicted [p_lo, p_hi] intervals,
    0 where they overlap. Broadcasts; NaN where either side is undefined."""
    with np.errstate(invalid="ignore"):
        gap = np.maximum(np.maximum(m_lo - p_hi, p_lo - m_hi), 0.0)
    undefined = np.isnan(m_lo) | np.isnan(p_lo) | np.isnan(m_hi) | np.isnan(p_hi)
    return np.where(undefined, np.nan, gap)


def near_walls(world: WallWorld, pose, reach: float) -> np.ndarray:
    """Walls within `reach` of the kart -- everything a view can see from
    anywhere in the search window."""
    return boxes_near(world.walls, pose[0], pose[1], reach)
