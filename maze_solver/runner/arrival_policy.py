"""
What to DO with a camera fix: which components go into the belief now, which
wait for the 1-D recentring, and whether the kart drives back to the centre.

Pure, so the decision table is tested without a kart.

Longitudinal is the subtle one. `Recenter` drives the kart until the sonar reads
the range predicted from the BELIEVED pose, frozen at its start. If the camera
had just moved the belief longitudinally onto the kart, Recenter would find
nothing to do and the kart would stay off-centre. So the camera's longitudinal
is held back ("stashed") and applied only if Recenter takes no measurement
there (no wall ahead in range) -- except when a centring move follows, because
that goto re-centres every axis and must aim from the fully corrected belief.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

from runner.camera_fit import CORRECTED, CameraFit
from runner.constants import CAMERA_FIX_LATERAL_MOVE_M, CAMERA_FIX_MOVE_POLICY

MEASURE = "medicao"
VERIFY = "verificacao"
START = "inicio"


class CameraDecision(NamedTuple):
    apply_lat: bool
    apply_yaw: bool
    apply_long_now: bool
    stash_long: bool
    move: bool


NOTHING = CameraDecision(False, False, False, False, False)


def plan_after_camera(fit: Optional[CameraFit], offset_lat_m: float,
                      pass_name: str, next_turn_deg: float, at_goal: bool,
                      policy: str = CAMERA_FIX_MOVE_POLICY,
                      move_threshold_m: float = CAMERA_FIX_LATERAL_MOVE_M) -> CameraDecision:
    """`offset_lat_m` is the CORRECTED pose's lateral offset from the cell
    centre (not the correction itself); `next_turn_deg` the in-place turn the
    next leg starts with."""
    if fit is None or fit.outcome != CORRECTED:
        return NOTHING
    apply_lat = fit.lateral is not None
    apply_yaw = fit.yaw is not None
    has_long = fit.longitudinal is not None

    move = False
    if pass_name in (MEASURE, START) and apply_lat and abs(offset_lat_m) >= move_threshold_m:
        if policy == "always":
            move = True
        elif policy == "before_turns":
            move = abs(next_turn_deg) >= 45.0 or at_goal
    return CameraDecision(apply_lat, apply_yaw,
                          apply_long_now=move and has_long,
                          stash_long=(not move) and has_long,
                          move=move)
