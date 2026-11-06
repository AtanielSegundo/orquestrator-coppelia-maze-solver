"""
What to DO with a camera fix: how much of each component goes into the belief,
whether the kart drives back to the centre, and whether the bigger corrector
has to step in.

Pure, so the decision table is tested without a kart.

Small corrections, per axis
---------------------------
The camera runs LAST at a boundary, after Recenter and after the closest
obstacle reference / sweep when one was due (see LiveMazeRunner's camera
block). Those sonar correctors make the big TRANSLATION corrections; the
camera mops up what they leave. The two axes kinds are treated differently
because only one of them has anybody else to fall back on:

* Translation (lateral, longitudinal) is shared with the sonar correctors.
  Inside CAMERA_FIX_SMALL_LATERAL_M / _LONG_M it is applied whole --
  longitudinal included, since Recenter has already driven that axis against
  the sonar and what is left is a residual. Beyond them it is NOT applied and
  is ESCALATED: the caller runs the big corrector now. Only if that cannot run
  here (no usable side wall, or a better one scheduled ahead) does `fallback`
  apply -- the same correction clamped to the small range, so a known error is
  never left standing just because the right tool is out of reach, and a wrong
  fit can never move the belief by more than the camera's own range.

* Yaw is the camera's alone: no sonar corrector observes heading. So it is
  always applied, clamped to CAMERA_FIX_SMALL_YAW_DEG. Escalating it would be
  escalating to nobody -- measured in the closed loop, a yaw error the camera
  refused to apply kept growing until even the fit's plausibility gate refused
  the (accurate) reading, and from then on nothing could correct it.

Step mode (CAMERA_FIX_STEP_MODE, the default) replaces all of the above
-----------------------------------------------------------------------
The camera only nudges: CAMERA_FIX_STEP_GAIN_* of each component, the
translation clamped to a norm of CAMERA_FIX_STEP_M and the yaw to
CAMERA_FIX_STEP_YAW_DEG, whatever the fit's confidence (camera_fit then does
not gate, see CameraFitter._judge). Never an escalation, never a centring
move: the gross errors are the reference swapping's, and a step that small
cannot by itself take a kart out of the 2 cm radius it counts as localised.
"""
from __future__ import annotations

import math
from typing import NamedTuple, Optional, Tuple

from runner.camera_fit import CORRECTED, CameraFit
from runner.constants import (CAMERA_FIX_LATERAL_MOVE_M, CAMERA_FIX_MOVE_POLICY,
                              CAMERA_FIX_SMALL_LATERAL_M, CAMERA_FIX_SMALL_LONG_M,
                              CAMERA_FIX_SMALL_YAW_DEG, CAMERA_FIX_STEP_GAIN_LAT,
                              CAMERA_FIX_STEP_GAIN_LONG, CAMERA_FIX_STEP_GAIN_YAW,
                              CAMERA_FIX_STEP_M, CAMERA_FIX_STEP_MODE,
                              CAMERA_FIX_STEP_YAW_DEG, CAMERA_FIX_STEP_YAW_SIGN)

MEASURE = "medicao"
VERIFY = "verificacao"
START = "inicio"


class CameraDecision(NamedTuple):
    longitudinal: float              # m to apply now, along the capture heading
    lateral: float                   # m to apply now, + = left
    yaw: float                       # rad to apply now
    move: bool                       # drive the kart back to the centre line
    escalate: bool                   # translation too big: big corrector now
    fallback: Tuple[float, float]    # (long, lat) to apply if it cannot run

    @property
    def applies(self) -> bool:
        return bool(self.longitudinal or self.lateral or self.yaw)


NOTHING = CameraDecision(0.0, 0.0, 0.0, False, False, (0.0, 0.0))


def _clamp(v: float, lim: float) -> float:
    return max(-lim, min(lim, v))


def plan_after_camera(fit: Optional[CameraFit], offset_lat_m: float,
                      pass_name: str, next_turn_deg: float, at_goal: bool,
                      policy: str = CAMERA_FIX_MOVE_POLICY,
                      move_threshold_m: float = CAMERA_FIX_LATERAL_MOVE_M,
                      small_lat_m: float = CAMERA_FIX_SMALL_LATERAL_M,
                      small_long_m: float = CAMERA_FIX_SMALL_LONG_M,
                      small_yaw_deg: float = CAMERA_FIX_SMALL_YAW_DEG,
                      step_mode: bool = CAMERA_FIX_STEP_MODE) -> CameraDecision:
    """`offset_lat_m` is the CORRECTED pose's lateral offset from the cell
    centre (not the correction itself); `next_turn_deg` the in-place turn the
    next leg starts with."""
    if fit is None or fit.outcome != CORRECTED:
        return NOTHING
    if step_mode:
        return step_after_camera(fit)
    lat = fit.lateral or 0.0
    lng = fit.longitudinal or 0.0
    yaw = _clamp(fit.yaw, math.radians(small_yaw_deg)) if fit.yaw is not None else 0.0

    if abs(lat) > small_lat_m or abs(lng) > small_long_m:
        return CameraDecision(0.0, 0.0, yaw, False, True,
                              (_clamp(lng, small_long_m), _clamp(lat, small_lat_m)))

    move = False
    if (pass_name in (MEASURE, START) and fit.lateral is not None
            and abs(offset_lat_m) >= move_threshold_m):
        if policy == "always":
            move = True
        elif policy == "before_turns":
            move = abs(next_turn_deg) >= 45.0 or at_goal
    return CameraDecision(lng, lat, yaw, move, False, (0.0, 0.0))


def step_after_camera(fit: CameraFit,
                      gain_long: float = CAMERA_FIX_STEP_GAIN_LONG,
                      gain_lat: float = CAMERA_FIX_STEP_GAIN_LAT,
                      gain_yaw: float = CAMERA_FIX_STEP_GAIN_YAW,
                      step_m: float = CAMERA_FIX_STEP_M,
                      step_yaw_deg: float = CAMERA_FIX_STEP_YAW_DEG,
                      yaw_sign: float = CAMERA_FIX_STEP_YAW_SIGN) -> CameraDecision:
    """The bounded nudge of step mode: a fraction of each component, the
    translation's NORM clamped to `step_m` (direction kept), yaw to
    `step_yaw_deg`, its sign multiplied by `yaw_sign` (CAMERA_FIX_STEP_YAW_
    SIGN). A component the fit left as None counts as zero."""
    if fit is None or fit.outcome != CORRECTED:
        return NOTHING
    lng = gain_long * (fit.longitudinal or 0.0)
    lat = gain_lat * (fit.lateral or 0.0)
    norm = math.hypot(lng, lat)
    if norm > step_m > 0.0:
        lng, lat = lng * step_m / norm, lat * step_m / norm
    yaw = yaw_sign * _clamp(gain_yaw * (fit.yaw or 0.0), math.radians(step_yaw_deg))
    return CameraDecision(lng, lat, yaw, False, False, (0.0, 0.0))
