"""
1-D longitudinal pose correction from the front distance sensor.

The idea
--------
Odometry closes the loop on rotation (the MPU6050 yaw is ground truth) but NOT
on translation: `d_center` comes from the encoders alone, so every centimetre of
wheel slip goes straight into the pose and never comes out. The error grows
without bound along a corridor.

The maze file, however, is a known map. So the walls are an exteroceptive
reference we already own: raycast from the believed pose against the ORIGINAL
maze walls, compare the predicted range with what the real sensor reports, and
the residual is a direct measurement of how wrong the pose is *along the beam*.

    predicted  = raycast(believed_pose, original_walls)
    measured   = front_distance_cm / 100
    residual   = measured - predicted

Sign, carefully: moving FORWARD along the heading SHORTENS the predicted range.
So a believed pose that has run ahead of the truth by d predicts a range that is
short by d, and `residual` comes out at +d. The residual therefore *is* the
forward drift, and the correction to apply along the heading is `-residual`.
(Getting this backwards doubles the error instead of removing it, which is why
`tests/test_range_fix.py` asserts the corrected pose is closer to the truth
rather than merely that a correction happened.)

Why 1-D, and only 1-D
---------------------
One scalar reading can correct at most one degree of freedom, and which one
depends on geometry. Pointed down a corridor at a wall, the reading says a great
deal about the distance TO that wall and essentially nothing about the kart's
lateral offset within the corridor. So this module corrects the longitudinal
coordinate only -- the component along the current heading -- and leaves lateral
and heading alone. That is the DOF the measurement actually observes, and it is
also the one that is unobservable by any other sensor on the kart.

Recovering the lateral DOF needs a *profile* of ranges taken at several
headings (rotate ~15 degrees, sample, repeat), which is a strictly bigger
feature; this is the 80/20 that removes the unbounded error.

The trust boundary
------------------
Ranges are predicted against `is_original_wall` ONLY -- never against obstacles
discovered at runtime. This is not a detail, it is what keeps the feature from
becoming a divergent SLAM loop:

    pose -> projects a reading -> marks an obstacle -> raycast hits it
         -> "confirms" the pose -> ...

A kart that mislocalises would invent an obstacle at the wrong place and then
localise against its own invention, confirming the error with growing
confidence. File walls are a priori map (immutable, trustworthy); discovered
obstacles are a belief *derived from* the pose and can never be used to correct
it.

Sensor validity and trust
-------------------------
A correction is attempted only when the reading is (a) a real measurement of a
real surface and (b) close enough to be trusted. Those are two separate gates:

  `is_valid_reading`    rejects the several ways the stack says "no
                        measurement" -- None / 255 / >=300 / <=0 / blind zone.
  `is_trusted_reading`  additionally caps the range at RANGE_FIX_MAX_TRUSTED_M.
                        A far reading is still a real measurement; it is just
                        not one to move a pose on, because the sensor's cone
                        widens with distance and an off-axis heading costs more
                        the further the wall is.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import Coppelia.helpers as cop

from runner.constants import (
    RANGE_FIX_ALIGN_TOLERANCE_DEG, RANGE_FIX_MAX_CORRECTION_M,
    RANGE_FIX_MAX_PREDICTED_M, RANGE_FIX_MAX_TRUSTED_M,
    RANGE_FIX_MIN_CORRECTION_M, RANGE_FIX_MIN_PREDICTED_M,
    SENSOR_FORWARD_OFFSET_M, SENSOR_INVALID_CM, SENSOR_MAX_RANGE_CM,
    SENSOR_MIN_RANGE_CM,
)


def is_valid_reading(dist_cm) -> bool:
    """Is this front reading a real measurement of a real surface?

    Every rejected case is a way the kart reports "no measurement", and taking
    any of them as a distance would corrupt the pose rather than fix it:

      * ``None``      -- the server's `front_distance_cm` expired (nothing valid
                         for `front_distance_ttl_s`), or there is no sensor. The
                         web UI shows this as "-- cm".
      * ``255``       -- the RPi.GPIO fallback's "failed read" sentinel. Not a
                         distance at all; 255 cm would read as a clear road.
      * ``>= 300``    -- the pigpio reader's max range, reported for far/clear
                         AND for a latched sensor fault. Either way it is "I see
                         nothing", not "there is a wall at 3 m".
      * ``<= 0``      -- nonsense.
      * below the HC-SR04's minimum -- inside the blind zone, where the echo
        overlaps the trigger burst and the number is unreliable.
    """
    if dist_cm is None:
        return False
    try:
        cm = float(dist_cm)
    except (TypeError, ValueError):
        return False
    if cm != cm:                                  # NaN
        return False
    if cm in SENSOR_INVALID_CM:
        return False
    return SENSOR_MIN_RANGE_CM <= cm < SENSOR_MAX_RANGE_CM


def is_trusted_reading(dist_cm) -> bool:
    """Is this reading close enough to CORRECT with?

    Deliberately separate from :func:`is_valid_reading`, and stricter. A 1.5 m
    reading is a perfectly real measurement -- it is just not one to move a pose
    (or a kart) on. Folding the two together would also break the other
    consumers of validity, such as obstacle mapping, for which a long reading is
    exactly the useful case.

    See RANGE_FIX_MAX_TRUSTED_M for why the ceiling sits where it does: the
    HC-SR04's cone widens with distance, so a far echo may have come off a side
    wall rather than the surface the ray points at, and the cost of an
    off-axis heading grows with range too.
    """
    if not is_valid_reading(dist_cm):
        return False
    return float(dist_cm) / 100.0 <= RANGE_FIX_MAX_TRUSTED_M


def predict_range(view, origin, cell_length: float,
                  wx: float, wy: float, wth: float,
                  max_range_m: float) -> Tuple[Optional[float], Optional[tuple]]:
    """Distance from the SENSOR to the first original maze wall along `wth`.

    Returns ``(metres, cell)``, or ``(None, None)`` when no wall is reached
    inside `max_range_m` -- which is a real answer, not a failure: it means the
    map predicts "clear", and a clear prediction cannot be compared against a
    measured range.

    Marched analytically over the grid rather than through CoppeliaSim's
    proximity sensor, for three reasons: the sim scene contains the discovered
    obstacles this must ignore, each sim raycast is a ZMQ round trip, and a grid
    march is testable with no simulator running at all.
    """
    # The reading is measured from the sensor face, which sits ahead of the
    # tracked center -- so the ray starts there too, or every predicted range
    # comes out one sensor-offset too long.
    sx = wx + math.cos(wth) * SENSOR_FORWARD_OFFSET_M
    sy = wy + math.sin(wth) * SENSOR_FORWARD_OFFSET_M

    cx, cy = math.cos(wth), math.sin(wth)
    # A quarter cell keeps the march from stepping over a one-cell-thick wall
    # while staying cheap (a few hundred samples at most).
    step = cell_length * 0.25
    steps = max(1, int(max_range_m / step))

    for i in range(steps + 1):
        t = i * step
        rf, cf = cop.world_to_cell(origin, cell_length, sx + cx * t, sy + cy * t)
        cell = (round(rf), round(cf))
        if not view.in_bounds(cell):
            return None, None            # left the map: nothing to measure to
        if view.is_original_wall(cell):
            # Refine to the actual face of that wall cell instead of reporting
            # the sample position, which is quantised to `step` and biases every
            # prediction by up to a quarter cell.
            return _refine_hit(view, origin, cell_length, sx, sy, cx, cy,
                               t, step), cell
    return None, None


def _refine_hit(view, origin, cell_length, sx, sy, cx, cy,
                t_hit: float, step: float) -> float:
    """Bisect between the last clear sample and the first blocked one, so the
    returned range is the wall's face rather than a quantised step boundary."""
    lo, hi = max(0.0, t_hit - step), t_hit
    for _ in range(24):                  # ~step/2^24: far below sensor noise
        mid = (lo + hi) / 2.0
        rf, cf = cop.world_to_cell(origin, cell_length, sx + cx * mid,
                                   sy + cy * mid)
        cell = (round(rf), round(cf))
        if view.in_bounds(cell) and view.is_original_wall(cell):
            hi = mid
        else:
            lo = mid
    return hi


def heading_alignment_error_deg(wth: float) -> float:
    """How far the heading is from the nearest grid axis, in degrees.

    The correction assumes the beam is roughly along a maze axis. Off-axis, the
    ray grazes wall corners and the predicted range becomes acutely sensitive to
    small heading errors -- a degree of yaw error can move the predicted hit by
    a whole cell, so the residual would be measuring heading error, not
    translation error, and "correcting" the pose with it makes things worse.
    """
    return abs(((math.degrees(wth) + 45.0) % 90.0) - 45.0)


class RangeFix:
    """Decides, and applies, the 1-D longitudinal correction.

    Stateless with respect to the pose: it is handed a believed pose and a
    reading, and returns a correction to apply (or None). The caller owns the
    pose. Keeping it that way means it can be unit-tested against a plain grid
    with no kart, no sim and no telemetry.
    """

    def __init__(self, view, origin, cell_length: float):
        self.view = view
        self.origin = origin
        self.cell_length = cell_length
        # Diagnostics, for the HUD -- the operator needs to see WHY a correction
        # was or was not applied, otherwise this is an invisible black box
        # silently moving the kart.
        self.applied = 0
        self.rejected_reason = "nao avaliado"
        self.last_residual_m = None
        self.last_predicted_m = None
        self.last_measured_m = None
        self.last_correction_m = 0.0
        self.total_correction_m = 0.0

    def evaluate(self, wx: float, wy: float, wth: float,
                 dist_cm, slipping: bool = False,
                 count: bool = True) -> Optional[float]:
        """Signed correction to ADD along the heading, in metres, or None.

        The caller applies it as ``(wx, wy) += correction * (cos wth, sin wth)``
        -- the sign returned is already the one to add.

        It is ``-residual``: moving forward shortens the predicted range, so a
        positive residual (measured further than predicted) means the believed
        pose had run AHEAD of the truth and must come back. See the module
        docstring.

        ``count=False`` runs the identical decision without touching the
        ``applied`` / ``total_correction_m`` counters. The physical recentring
        loop (runner/recenter.py) calls this several times per session just to
        MEASURE, and counting those would turn the diagnostics into a tally of
        measurements rather than of corrections actually applied.
        """
        self.last_residual_m = None
        self.last_predicted_m = None
        self.last_measured_m = None
        self.last_correction_m = 0.0

        if not is_valid_reading(dist_cm):
            self.rejected_reason = "sensor sem leitura valida"
            return None

        # Trust ceiling, reported separately from validity so the HUD says
        # which gate refused: "there is no reading" and "the reading is too far
        # to act on" call for different things from the operator.
        if not is_trusted_reading(dist_cm):
            self.rejected_reason = (f"leitura {float(dist_cm):.0f} cm acima do "
                                    f"limite de confianca "
                                    f"({RANGE_FIX_MAX_TRUSTED_M * 100:.0f} cm)")
            return None

        # A reading taken while the wheels are scrubbing is still a valid
        # DISTANCE -- the ultrasonic does not care -- but the POSE it would be
        # compared against was integrated from those same slipping wheels, so
        # the residual mixes the error being measured with the error being
        # corrected. Wait for traction before trusting the comparison.
        if slipping:
            self.rejected_reason = "derrapagem detectada"
            return None

        align = heading_alignment_error_deg(wth)
        if align > RANGE_FIX_ALIGN_TOLERANCE_DEG:
            self.rejected_reason = f"fora de eixo ({align:.0f} deg)"
            return None

        predicted, _cell = predict_range(self.view, self.origin,
                                         self.cell_length, wx, wy, wth,
                                         RANGE_FIX_MAX_PREDICTED_M)
        if predicted is None:
            self.rejected_reason = "mapa nao preve parede no alcance"
            return None

        measured = float(dist_cm) / 100.0
        self.last_predicted_m = predicted
        self.last_measured_m = measured

        # Too close: inside a cell of the wall the geometry is dominated by
        # which part of the cell face the beam catches. Too far: the predicted
        # range is long enough that a small heading error dominates the residual.
        if not (RANGE_FIX_MIN_PREDICTED_M <= predicted
                <= RANGE_FIX_MAX_PREDICTED_M):
            self.rejected_reason = f"parede prevista a {predicted:.2f} m (fora da faixa)"
            return None

        residual = measured - predicted
        self.last_residual_m = residual

        if abs(residual) < RANGE_FIX_MIN_CORRECTION_M:
            self.rejected_reason = "residual dentro do ruido"
            return None

        # A residual bigger than this is not drift, it is a disagreement about
        # WHICH wall the beam hit -- a wrong cell, an unmapped object in the
        # corridor, or a pose already lost by more than this module can fix.
        # Applying it would teleport the kart; refusing leaves the existing
        # (merely drifting) pose, which is the lesser error.
        if abs(residual) > RANGE_FIX_MAX_CORRECTION_M:
            self.rejected_reason = (f"residual {residual:+.2f} m grande demais "
                                    f"(parede errada?)")
            return None

        correction = -residual          # see the module docstring on the sign
        self.rejected_reason = ""
        self.last_correction_m = correction
        if count:
            self.applied += 1
            self.total_correction_m += abs(correction)
        return correction

    def status_line(self) -> str:
        """One line for the HUD."""
        if self.last_predicted_m is None:
            return f"range-fix: {self.rejected_reason or 'ocioso'} " \
                   f"({self.applied} aplicada(s))"
        base = (f"range-fix: prev={self.last_predicted_m:.2f} m "
                f"med={self.last_measured_m:.2f} m")
        if self.rejected_reason:
            return f"{base} -> {self.rejected_reason} ({self.applied} aplicada(s))"
        return (f"{base} -> corrigiu {self.last_correction_m:+.3f} m "
                f"({self.applied} aplicada(s), "
                f"acum {self.total_correction_m:.2f} m)")
