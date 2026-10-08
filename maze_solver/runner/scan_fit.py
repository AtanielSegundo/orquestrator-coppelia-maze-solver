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

They are also separated because only ONE of them is reliably observable, which
is the subtlety that makes this module dangerous if used naively. See
`longitudinal_is_observable`: in a corridor with no wall ahead the cost surface
is exactly flat along the heading (measured, not supposed), so the search
reports an arbitrary value -- on tcc.maze, -16 cm against a true +2 cm. That
component therefore comes back as None unless the geometry actually pinned it,
while lateral, which a sweep of side walls constrains well, is always reported.

Robustness
----------
Not by trimming. `SCAN_FIT_TRIM_FRACTION` is 0.0 and its comment in constants.py
carries the measurement: a sweep yields ~4 usable samples and a 2-DOF fit needs
2, so discarding the worst 25% throws away a reading that was pinning the
answer. Trimming took the p90 lateral error from 0.82 cm to 8.68 cm and gross
failures from 0 to 42 (of 289 fits). The machinery is kept because it is right
for a denser scan; the fraction is zero because this scan is sparse.

What protects the fit instead is refusing to produce one: SCAN_FIT_MIN_SAMPLES
(4, where the error tail disappears), the observability gate above, and the
caller's own ceilings on how large a correction it will apply. All three fail
toward leaving the belief alone.
"""
from __future__ import annotations

import math
import time
from typing import List, NamedTuple, Optional, Sequence, Tuple

from runner.constants import (
    SCAN_FIT_COARSE_STEP_M, SCAN_FIT_FINE_STEP_M, SCAN_FIT_MAX_SHIFT_M,
    SCAN_FIT_MIN_SAMPLES, SCAN_FIT_OBSERVABLE_MIN_RISE, SCAN_FIT_YIELD_S,
    SCAN_FIT_OBSERVABLE_PROBE_M, SCAN_FIT_TRIM_FRACTION,
    RANGE_FIX_MAX_PREDICTED_M,
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
    longitudinal: Optional[float]   # m along the heading, or None if unobservable
    lateral: Optional[float]        # m perpendicular (+ left), or None if so
    rms_before: float
    rms_after: float
    used: int             # samples that survived trimming
    total: int

    @property
    def longitudinal_observable(self) -> bool:
        """Whether the sweep's geometry constrained the along-heading axis.

        False in a straight corridor with no wall ahead in range, where the side
        walls look the same wherever the kart sits along them.
        """
        return self.longitudinal is not None

    @property
    def lateral_observable(self) -> bool:
        """Whether the sweep's geometry constrained the across-heading axis.

        False when the sweep was taken facing ACROSS a corridor: the beams go
        out through the open ends and nothing pins the kart sideways. Usually
        True -- a sweep sees side walls best -- but not something a caller may
        assume. See `axis_is_observable`.
        """
        return self.lateral is not None


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


def _search(view, origin, cell_length, px, py, samples, centre, half, step,
            yield_every: int = 0):
    """Grid search over (dx, dy) in [centre +- half], returning the best.

    `yield_every` parks the thread briefly every that many rows when non-zero.
    That is not politeness: CPython holds the GIL through a loop like this, so
    running the search on a worker thread still stalls the main thread -- which
    defeats the purpose of being off-thread at all.

    The sleep is SCAN_FIT_YIELD_S, not zero, and that detail is measured rather
    than assumed. With a CPU-bound worker running, the worst wait seen by a main
    thread trying to tick was:

        no yield          17.5 ms
        time.sleep(0)     26.0 ms   (worse than nothing)
        time.sleep(5e-4)   2.8 ms

    `sleep(0)` does not release the GIL to a waiting thread in CPython 3.14 --
    it returns too fast to force a switch, and the extra syscall makes it worse
    than not trying. An actual (tiny) sleep does yield. The cost is a few
    percent of fit latency, paid while the kart is stationary and waiting
    anyway. See runner/sweep.py, which passes it.
    """
    best = None
    n = int(round(half / step))
    for row, i in enumerate(range(-n, n + 1)):
        if yield_every and row % yield_every == 0:
            time.sleep(SCAN_FIT_YIELD_S)
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


def axis_is_observable(view, origin, cell_length, px, py, samples,
                       dx: float, dy: float,
                       ux: float, uy: float) -> bool:
    """Did this sweep's geometry constrain the pose along the unit axis (ux, uy)?

    The question is not academic, and it applies to BOTH axes -- which is the
    part that is easy to get wrong. Measured cases:

    * Facing ALONG a corridor with no wall ahead inside
      RANGE_FIX_MAX_PREDICTED_M, the side walls present the same profile
      wherever the kart sits along it, so the cost is flat LONGITUDINALLY. The
      search then returns whichever end of the window it touched: on tcc.maze,
      about -16 cm against a true +2 cm.
    * Facing ACROSS a corridor, the sweep looks out through the open ends and
      the cost is flat LATERALLY instead -- the fit runs to the window edge and
      reports a confident 12 cm that is pure artefact.

    So neither axis gets a free pass. Probe the cost at the optimum, then
    SCAN_FIT_OBSERVABLE_PROBE_M either side of it along the axis, and require
    the cost to rise in BOTH directions. A real minimum sits in a basin; an
    unobservable axis climbs on neither side, and a one-sided climb means the
    optimum is pinned against the window edge rather than found.

    What survives this is a correction the readings actually support. What does
    not comes back as None, so a caller cannot act on it by accident.
    """
    def cost_at(t: float) -> Optional[float]:
        r = _residuals(view, origin, cell_length,
                       px + dx + ux * t, py + dy + uy * t, samples)
        if r is None:
            return None
        return _trimmed_cost(r, SCAN_FIT_TRIM_FRACTION)[0]

    base = cost_at(0.0)
    if base is None:
        return False
    probe = SCAN_FIT_OBSERVABLE_PROBE_M
    rises = 0
    for t in (+probe, -probe):
        c_off = cost_at(t)
        if c_off is None:
            # The map foresees no wall from there, so that side carries no
            # evidence either way -- not a rise.
            continue
        if c_off - base >= SCAN_FIT_OBSERVABLE_MIN_RISE:
            rises += 1
    return rises == 2


def longitudinal_is_observable(view, origin, cell_length, px, py, heading,
                               samples, dx: float, dy: float) -> bool:
    """`axis_is_observable` along the heading. See it for why this gate exists."""
    return axis_is_observable(view, origin, cell_length, px, py, samples,
                              dx, dy, math.cos(heading), math.sin(heading))


def lateral_is_observable(view, origin, cell_length, px, py, heading,
                          samples, dx: float, dy: float) -> bool:
    """`axis_is_observable` perpendicular to the heading (+ is to the left).

    Needed for the same reason as the longitudinal gate and in a different
    geometry: a sweep taken facing across a corridor sees the open ends, not
    the side walls, and then lateral is the blind axis.
    """
    return axis_is_observable(view, origin, cell_length, px, py, samples,
                              dx, dy, -math.sin(heading), math.cos(heading))


def fit_scan(view, origin, cell_length: float,
             px: float, py: float, heading: float,
             samples: Sequence[ScanSample],
             yield_every: int = 0) -> Optional[ScanFit]:
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
                     (0.0, 0.0), SCAN_FIT_MAX_SHIFT_M, SCAN_FIT_COARSE_STEP_M,
                     yield_every=yield_every)
    if coarse is None:
        return None
    fine = _search(view, origin, cell_length, px, py, samples,
                   (coarse[1], coarse[2]), SCAN_FIT_COARSE_STEP_M,
                   SCAN_FIT_FINE_STEP_M, yield_every=yield_every)
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

    # ...and because only one of them is always observable. An unconstrained
    # along-heading axis gets None rather than the number the search landed on,
    # so a caller cannot act on it by accident -- the failure mode here is not
    # a slightly-off correction, it is a confident 16 cm one.
    # Gate each axis, then PROJECT the offset onto the axes that survived.
    #
    # The projection is not tidying-up, it is load-bearing. When one axis is
    # unobservable the optimum slides freely along it and parks at the window
    # edge, and the gate for the OTHER axis would then be probing from that
    # bogus position -- where the cost does rise, because being 12 cm sideways
    # in a corridor really is constrained. The second axis would pass its gate
    # while validating an artefact. Zeroing the dead component first means both
    # gates judge the pose a caller would actually apply.
    long_ok = longitudinal_is_observable(view, origin, cell_length,
                                         px, py, heading, samples, dx, dy)
    lat_ok = lateral_is_observable(view, origin, cell_length,
                                   px, py, heading, samples, dx, dy)
    if not long_ok:
        longitudinal = None
    if not lat_ok:
        lateral = None

    if longitudinal is None and lateral is None:
        # Nothing was measured: the readings are consistent with the pose being
        # anywhere in the window. Returning a fit here would hand the caller two
        # Nones to special-case; returning None says the same thing once.
        return None

    if longitudinal is None or lateral is None:
        # Rebuild (dx, dy) from the surviving component alone, and re-verify it
        # from there -- the first gate was measured at the unprojected optimum.
        keep_long = longitudinal if longitudinal is not None else 0.0
        keep_lat = lateral if lateral is not None else 0.0
        dx = c * keep_long - s * keep_lat
        dy = s * keep_long + c * keep_lat
        # Re-verify from the PROJECTED offset: the earlier verdict was measured
        # at the unprojected optimum, which sat off in the dead axis.
        if longitudinal is not None and not longitudinal_is_observable(
                view, origin, cell_length, px, py, heading, samples, dx, dy):
            return None
        if lateral is not None and not lateral_is_observable(
                view, origin, cell_length, px, py, heading, samples, dx, dy):
            return None
        final = _residuals(view, origin, cell_length, px + dx, py + dy, samples)
        if final is None:
            return None
        rms_after = math.sqrt(sum(r * r for r in final) / len(final))

    return ScanFit(dx=dx, dy=dy, longitudinal=longitudinal, lateral=lateral,
                   rms_before=rms_before, rms_after=rms_after,
                   used=used, total=total)
