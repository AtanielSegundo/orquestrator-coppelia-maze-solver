"""
Fit a 2-DOF pose correction to an angular sweep of range readings.

What the sweep buys
-------------------
A single forward reading observes ONE degree of freedom. Pointed down a
corridor at a wall, it constrains the distance to that wall and says nothing
about lateral offset -- which is why `range_fix` corrects longitudinally only.

Sweeping the heading fixes that. Readings taken at several angles see the maze
from different directions, so the lateral coordinate stops being invisible:
move the kart sideways in a corridor and the profile of ranges-versus-angle
shifts in a way a single reading cannot show. That is the entire technical
reason to spend the rotations.

The model
---------
For a candidate correction (dx, dy) applied to the recorded pose, each sample
predicts a range by raycasting the ORIGINAL maze walls from the corrected
position along that sample's own measured heading:

    r_pred_i(dx, dy) = raycast(px + dx, py + dy, theta_i)
    residual_i       = r_meas_i - r_pred_i

and the fit minimises the sum of squared residuals over (dx, dy).

Solved by direct search rather than Gauss-Newton on purpose. The cost surface
is genuinely non-smooth: a raycast against a grid is piecewise-linear with
jumps where the ray slides past a wall corner, so a derivative-based step can
land in a discontinuity and diverge. Over a window of +-15 cm at centimetre
resolution the whole grid is a few thousand raycasts -- some milliseconds --
and it cannot diverge, which matters more here than elegance.

What is done with the result
-----------------------------
Longitudinal (along the heading) and lateral (perpendicular) are separated,
because they are actionable in different ways. The kart can be DRIVEN
longitudinally to cancel that part; it has no way to translate sideways, so
the lateral part corrects the belief only.

Robustness
----------
Samples are fitted with a trimmed cost: the worst `SCAN_FIT_TRIM_FRACTION` of
residuals are dropped before summing. One sample looking through a doorway the
map does not model, or catching a corner, would otherwise drag the whole fit.
"""
from __future__ import annotations

import math
from typing import List, NamedTuple, Optional, Sequence, Tuple

from runner.constants import (
    SCAN_FIT_COARSE_STEP_M, SCAN_FIT_FINE_STEP_M, SCAN_FIT_MAX_SHIFT_M,
    SCAN_FIT_MIN_SAMPLES, SCAN_FIT_TRIM_FRACTION, RANGE_FIX_MAX_PREDICTED_M,
)
from runner.range_fix import predict_range


class ScanSample(NamedTuple):
    """One reading of the sweep.

    `theta` is the heading the kart ACTUALLY had when the reading was taken,
    read from telemetry (gyro-backed), not the angle that was commanded. A turn
    of 15 degrees that came out as 12 must be compared against 12, or the
    rotation error is laundered into the position fit.
    """
    theta: float          # rad, world frame
    measured: float       # m, sensor to surface


class ScanFit(NamedTuple):
    dx: float             # m, world
    dy: float             # m, world
    longitudinal: float   # m, along the recorded heading
    lateral: float        # m, perpendicular (+ is to the left of the heading)
    rms_before: float
    rms_after: float
    used: int             # samples that survived trimming
    total: int


def _trimmed_cost(residuals: List[float], trim: float) -> Tuple[float, int]:
    """Sum of squares over the best (1 - trim) fraction of residuals.

    A plain sum lets one bad sample dominate: a reading through an opening the
    map does not model, or one that grazed a corner, is off by a whole cell
    while the rest are off by centimetres. Squaring makes that sample worth
    hundreds of good ones.
    """
    if not residuals:
        return float("inf"), 0
    keep = max(1, int(round(len(residuals) * (1.0 - trim))))
    best = sorted(residuals, key=abs)[:keep]
    return sum(r * r for r in best), keep


def _residuals(view, origin, cell_length, px, py,
               samples: Sequence[ScanSample]) -> Optional[List[float]]:
    """Measured minus predicted for every sample, from the position (px, py).

    Returns None if too few samples can be predicted at all: a candidate
    position from which the map foresees no wall is not a candidate whose cost
    can be compared against one where it does. Scoring it on a subset would
    make "predicts nothing" look cheap and let the search run away to it.
    """
    out = []
    for s in samples:
        predicted, _hit = predict_range(view, origin, cell_length,
                                        px, py, s.theta,
                                        RANGE_FIX_MAX_PREDICTED_M)
        if predicted is None:
            continue
        out.append(s.measured - predicted)
    if len(out) < SCAN_FIT_MIN_SAMPLES:
        return None
    return out


def _search(view, origin, cell_length, px, py, samples, centre, half, step):
    """Grid search over (dx, dy) in [centre +- half], returning the best."""
    best = None
    n = int(round(half / step))
    for i in range(-n, n + 1):
        dx = centre[0] + i * step
        for j in range(-n, n + 1):
            dy = centre[1] + j * step
            residuals = _residuals(view, origin, cell_length,
                                   px + dx, py + dy, samples)
            if residuals is None:
                continue
            cost, used = _trimmed_cost(residuals, SCAN_FIT_TRIM_FRACTION)
            if best is None or cost < best[0]:
                best = (cost, dx, dy, used, len(residuals))
    return best


def fit_scan(view, origin, cell_length: float,
             px: float, py: float, heading: float,
             samples: Sequence[ScanSample]) -> Optional[ScanFit]:
    """Best (dx, dy) correction for the recorded pose, or None.

    `px, py, heading` are the pose believed when the sweep started -- the one
    every sample's `theta` is expressed against, and the one being corrected.

    Two passes: a coarse grid over the whole search window, then a fine grid
    around the coarse winner. One fine pass over the full window would be
    ~900x the raycasts for the same answer.
    """
    if len(samples) < SCAN_FIT_MIN_SAMPLES:
        return None

    base = _residuals(view, origin, cell_length, px, py, samples)
    if base is None:
        return None
    rms_before = math.sqrt(sum(r * r for r in base) / len(base))

    coarse = _search(view, origin, cell_length, px, py, samples,
                     (0.0, 0.0), SCAN_FIT_MAX_SHIFT_M, SCAN_FIT_COARSE_STEP_M)
    if coarse is None:
        return None
    fine = _search(view, origin, cell_length, px, py, samples,
                   (coarse[1], coarse[2]), SCAN_FIT_COARSE_STEP_M,
                   SCAN_FIT_FINE_STEP_M)
    best = fine if fine is not None and fine[0] < coarse[0] else coarse

    _cost, dx, dy, used, total = best
    final = _residuals(view, origin, cell_length, px + dx, py + dy, samples)
    if final is None:
        return None
    rms_after = math.sqrt(sum(r * r for r in final) / len(final))

    # Split into the two components, because they are actionable differently:
    # the kart can be driven along its heading, but it cannot translate
    # sideways, so the lateral part can only ever move the belief.
    c, s = math.cos(heading), math.sin(heading)
    longitudinal = dx * c + dy * s
    lateral = -dx * s + dy * c

    return ScanFit(dx=dx, dy=dy, longitudinal=longitudinal, lateral=lateral,
                   rms_before=rms_before, rms_after=rms_after,
                   used=used, total=total)
