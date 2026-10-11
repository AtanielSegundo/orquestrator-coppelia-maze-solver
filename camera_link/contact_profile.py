"""
The floor-contact profile: per image column, the row where the floor ends.

Why this feature
----------------
Scanning a column UP from the bottom of the image, pixels look at floor points
further and further away, until the first one that is not floor -- the bottom
edge of the nearest wall in that column, i.e. where that wall stands on the
floor. That row depends only on wall FOOTPRINTS (which is exactly what a .maze
file describes), not on box height, box texture, tape, labels or what is above
the boxes. Shading and colour enter only through the wall/floor segmentation,
and only at one place per column.

It is also cheap to compare: one number per column, the same extraction for a
real frame and for any model of what the camera should see.

Kinds of columns
----------------
    HIT      a contact was found between `v_near` and `v_far`
    FREE     floor all the way to `v_far` (the configured floor range)
    NEAR     already wall at `v_near` -- the contact, if any, is hidden in the
             ignore band at the bottom (kart body)
    INVALID  nothing to say (column outside the usable image)

`v_near` is the top of the bottom ignore band; `v_far` is per column, the row
where the floor range limit sits. Rows grow DOWNWARDS, so "further" is "smaller
row". The contact row is the boundary between the last floor pixel and the first
wall pixel: r + 0.5 for wall pixel r over floor pixel r + 1, refined to the
zero crossing of the log-likelihood ratio when one is given.
"""
from __future__ import annotations

from typing import NamedTuple, Optional, Sequence

import numpy as np

HIT = 0
FREE = 1
NEAR = 2
INVALID = -1


class ContactProfile(NamedTuple):
    columns: np.ndarray          # (n,) int pixel columns
    rows: np.ndarray             # (n,) float contact row for HIT, else NaN
    kind: np.ndarray             # (n,) int8 HIT / FREE / NEAR / INVALID
    v_near: int
    v_far: np.ndarray            # (n,) float

    @property
    def hits(self) -> int:
        return int(np.count_nonzero(self.kind == HIT))

    def intervals(self):
        """(lo, hi) rows allowed for the contact, per column. HIT is a point,
        FREE is "at v_far or further", NEAR is "at v_near or nearer". INVALID
        is (nan, nan) and must be skipped by the caller."""
        n = len(self.kind)
        lo = np.full(n, np.nan)
        hi = np.full(n, np.nan)
        hit = self.kind == HIT
        lo[hit] = hi[hit] = self.rows[hit]
        free = self.kind == FREE
        lo[free] = -np.inf
        hi[free] = self.v_far[free]
        near = self.kind == NEAR
        lo[near] = self.v_near
        hi[near] = np.inf
        return lo, hi


def extract_contact_profile(wall: np.ndarray, columns: Sequence[int],
                            v_near: int, v_far: np.ndarray,
                            min_run_px: int = 3,
                            llr: Optional[np.ndarray] = None,
                            invalid: Optional[np.ndarray] = None) -> ContactProfile:
    """Contact profile of a boolean wall mask (H, W).

    A contact needs `min_run_px` consecutive wall pixels, so an isolated
    misclassified speck on the floor does not end the column early.
    `invalid` (H, W) marks pixels with no image behind them (undistortion
    borders): a column crossing any between v_near and v_far is INVALID rather
    than guessed.
    """
    columns = np.asarray(columns, dtype=int)
    v_far = np.asarray(v_far, dtype=float)
    n = len(columns)
    kind = np.full(n, INVALID, dtype=np.int8)
    rows = np.full(n, np.nan)
    usable = np.isfinite(v_far) & (v_far < v_near)
    if not np.any(usable):
        return ContactProfile(columns, rows, kind, int(v_near), v_far)

    top = max(0, int(np.floor(np.nanmin(np.where(usable, v_far, np.inf)))))
    # index 0 = row v_near, growing index = going UP the image
    sub = wall[top:v_near + 1, :][:, columns][::-1].astype(np.int32)
    depth = sub.shape[0]
    run = max(1, int(min_run_px))
    if depth < run:
        return ContactProfile(columns, rows, kind, int(v_near), v_far)

    cs = np.vstack([np.zeros((1, n), np.int32), np.cumsum(sub, axis=0)])
    starts = np.arange(depth - run + 1)
    run_ok = (cs[starts + run] - cs[starts]) == run            # (S, n)
    # A run must START at or below this column's v_far.
    limit = np.where(usable, v_near - v_far, -1.0)            # max index
    run_ok &= starts[:, None] <= limit[None, :]

    has = run_ok.any(axis=0)
    first = np.argmax(run_ok, axis=0)
    kind[usable & ~has] = FREE
    kind[usable & has & (first == 0)] = NEAR
    hit = usable & has & (first > 0)
    kind[hit] = HIT

    r = v_near - first[hit]                    # wall pixel row
    est = r + 0.5
    if llr is not None:
        cols = columns[hit]
        a = llr[r + 1, cols].astype(float)     # floor side, expected < 0
        b = llr[r, cols].astype(float)         # wall side, expected > 0
        ok = (a < 0) & (b > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            zc = r + 1 + a / (b - a)
        est = np.where(ok, np.clip(zc, r, r + 1), est)
    rows[hit] = est

    if invalid is not None:
        bad = np.zeros(n, dtype=bool)
        for i in np.flatnonzero(usable):
            lo = int(np.floor(v_far[i]))
            if invalid[max(0, lo):v_near + 1, columns[i]].any():
                bad[i] = True
        kind[bad] = INVALID
        rows[bad] = np.nan
    return ContactProfile(columns, rows, kind, int(v_near), v_far)


def median_profile(profiles: Sequence[ContactProfile]) -> ContactProfile:
    """Combine the profiles of several frames of the same view.

    Majority vote on the kind, median of the HIT rows among the frames that
    voted HIT. One blurred or half-decoded frame then cannot move a column,
    and with three frames neither can a passing hand.
    """
    if len(profiles) == 1:
        return profiles[0]
    kinds = np.stack([p.kind for p in profiles])
    rows = np.stack([p.rows for p in profiles])
    base = profiles[0]
    n = kinds.shape[1]
    out_kind = np.full(n, INVALID, dtype=np.int8)
    out_rows = np.full(n, np.nan)
    need = len(profiles) // 2 + 1
    for k in (HIT, FREE, NEAR):
        votes = np.count_nonzero(kinds == k, axis=0)
        out_kind[votes >= need] = k
    hit = out_kind == HIT
    if np.any(hit):
        masked = np.where(kinds[:, hit] == HIT, rows[:, hit], np.nan)
        out_rows[hit] = np.nanmedian(masked, axis=0)
    return ContactProfile(base.columns, out_rows, out_kind, base.v_near, base.v_far)
