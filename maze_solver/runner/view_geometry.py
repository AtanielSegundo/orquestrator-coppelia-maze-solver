"""
Everything about one camera view that does not depend on where the kart is.

For each sampled image column, walking UP from the top of the ignore band
(`v_near`) in `row_step` steps, every pixel centre is back-projected onto the
floor. Each sample is stored by its distance `r` and bearing `phi` from the
camera FOOT, in the kart's centre frame. Those two numbers do not change when
the kart translates, and rotating the kart only adds its heading to `phi` -- so
the model of any candidate pose only has to raycast the maze from the foot
along bearings, never touch the camera again.

Columns are not vertical planes once the camera is pitched: all column traces
on the floor fan out from the nadir point behind the foot, and the samples of
one column sit at different bearings from the foot. Storing (r, phi) per sample
rather than per column is what keeps the model exact at the image edges and
around box corners.

A column's samples stop at the floor-range limit (`max_range_m` from the foot)
or just below the horizon; `v_far` is the row of the last one. The measured
profile is scanned over exactly the same rows.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from camera_link.pinhole_view import PinholeView


class ViewGeometry(NamedTuple):
    name: str
    pinhole: PinholeView
    columns: np.ndarray          # (n,) pixel columns
    v_near: int
    rows: np.ndarray             # (n, K) sample rows (float, pixel centres)
    valid: np.ndarray            # (n, K) bool, contiguous from k = 0
    r: np.ndarray                # (n, K) distance from the foot, m
    phi: np.ndarray              # (n, K) bearing from the foot, rad, centre frame
    v_far: np.ndarray            # (n,) row of the last valid sample (nan: unusable)
    foot: np.ndarray             # (2,) foot in the centre frame
    phi0: float                  # first bearing bin
    dphi: float                  # bearing bin width
    n_bins: int
    bin_idx: np.ndarray          # (n, K) int bearing bin of each sample
    nuisance: bool
    max_range: float

    @property
    def n_columns(self) -> int:
        return len(self.columns)


def build_view_geometry(name: str, pinhole: PinholeView, ignore_bottom_px: int,
                        max_range_m: float, col_step: int, row_step: int,
                        bearing_step_deg: float, nuisance: bool = False) -> ViewGeometry:
    W, H = pinhole.width, pinhole.height
    columns = np.arange(col_step // 2, W, col_step, dtype=int)
    v_near = int(H - 1 - max(0, int(ignore_bottom_px)))
    sample_rows = np.arange(v_near, -1, -row_step, dtype=float)       # (K,)
    uu, vv = np.meshgrid(columns.astype(float), sample_rows, indexing="ij")
    pts = pinhole.floor_points(uu, vv)                                # (n, K, 3)
    fx, fy = pinhole.foot
    ddx = pts[..., 0] - fx
    ddy = pts[..., 1] - fy
    r = np.hypot(ddx, ddy)
    phi = np.arctan2(ddy, ddx)
    horizon = pinhole.horizon_rows(columns.astype(float))[:, None]
    ok = np.isfinite(r) & (r <= max_range_m) & (vv > horizon + 2.0)
    valid = np.cumprod(ok, axis=1).astype(bool)                       # contiguous
    count = valid.sum(axis=1)
    v_far = np.where(count >= 2, sample_rows[np.maximum(count - 1, 0)], np.nan)
    rows = np.broadcast_to(sample_rows, uu.shape).copy()

    dphi = math.radians(bearing_step_deg)
    if np.any(valid):
        lo = float(np.min(phi[valid])) - 2 * dphi
        hi = float(np.max(phi[valid])) + 2 * dphi
    else:
        lo, hi = -dphi, dphi
    n_bins = int(math.ceil((hi - lo) / dphi)) + 1
    bin_idx = np.clip(np.rint((np.where(valid, phi, lo) - lo) / dphi),
                      0, n_bins - 1).astype(np.int32)
    r = np.where(valid, r, np.inf)
    return ViewGeometry(name, pinhole, columns, v_near, rows, valid, r, phi,
                        v_far, np.array([fx, fy]), lo, dphi, n_bins, bin_idx,
                        bool(nuisance), float(max_range_m))
