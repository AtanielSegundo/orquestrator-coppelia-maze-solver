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
"""
from __future__ import annotations

import math
from typing import NamedTuple, Optional, Tuple

from runner.camera_fit import CORRECTED, CameraFit
from runner.constants import (CAMERA_FIX_LATERAL_MOVE_M, CAMERA_FIX_MOVE_POLICY,
                              CAMERA_FIX_SMALL_LATERAL_M, CAMERA_FIX_SMALL_LONG_M,
                              CAMERA_FIX_SMALL_YAW_DEG)

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
                      small_yaw_deg: float = CAMERA_FIX_SMALL_YAW_DEG) -> CameraDecision:
    """`offset_lat_m` is the CORRECTED pose's lateral offset from the cell
    centre (not the correction itself); `next_turn_deg` the in-place turn the
    next leg starts with."""
    if fit is None or fit.outcome != CORRECTED:
        return NOTHING
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
