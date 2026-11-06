"""
Camera calibration for the pose fix: the hardware defaults from
runner/constants.py, overridden by the calibration file when there is one, and
the per-view geometry built from it.

The defaults are a complete, usable calibration (nominal Pi Camera rev 1.3
optics, the measured lens position, the commanded pan/tilt) -- good enough to
run, and exactly what `camera_calibrate.py` refines.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from camera_link.camera_calibration import (CameraCalibration, TiltCalibration,
                                            ViewCalibration)
from camera_link.pinhole_view import PinholeView

from runner.constants import (
    CAMERA_CALIB_DEFAULT_PATH, CAMERA_FIX_BEARING_STEP_DEG,
    CAMERA_FIX_COLUMN_STEP_PX, CAMERA_FIX_IGNORE_BOTTOM_PX,
    CAMERA_FIX_MAX_FLOOR_RANGE_M, CAMERA_FIX_PAN_NUISANCE_SIGMA_DEG,
    CAMERA_FIX_ROW_STEP_PX, CAMERA_FIX_TILT_PCT, CAMERA_FIX_VIEWS,
    CAMERA_FOCAL_MM, CAMERA_LENS_FORWARD_M, CAMERA_LENS_HEIGHT_M,
    CAMERA_LENS_LATERAL_M, CAMERA_PAN_ARM_M, CAMERA_PAN_HOME_PCT,
    CAMERA_PAN_LEFT_IS_HIGHER, CAMERA_PIXEL_UM, CAMERA_SENSOR_CROP_PX,
    CAMERA_SERVO_DEG_PER_PCT, CAMERA_STREAM_SIZE, CAMERA_TILT_DOWN_IS_LOWER,
    CAMERA_TILT_HOME_PCT, CAMERA_V2_PAN_ORDER, CAMERA_V2_TILT_ORDER,
    CAMERA_V2_TILT_UP_PCT,
)
from runner.view_geometry import ViewGeometry, build_view_geometry

ORQUESTRATOR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def nominal_focal_px(width: int = CAMERA_STREAM_SIZE[0]) -> float:
    """3.6 mm lens over the OV5647 window the 640x480 mode reads, scaled to the
    stream width -- see CAMERA_SENSOR_CROP_PX."""
    pixel_mm = CAMERA_SENSOR_CROP_PX * CAMERA_PIXEL_UM * 1e-3 / width
    return CAMERA_FOCAL_MM / pixel_mm


def fix_pitch_deg(tilt_pct: int = CAMERA_FIX_TILT_PCT) -> float:
    """Downward pitch the commanded fix tilt corresponds to, from the level
    home tilt and one servo degree per unit."""
    sign = 1.0 if CAMERA_TILT_DOWN_IS_LOWER else -1.0
    return sign * (CAMERA_TILT_HOME_PCT - tilt_pct) * CAMERA_SERVO_DEG_PER_PCT


def default_calibration() -> CameraCalibration:
    w, h = CAMERA_STREAM_SIZE
    f = nominal_focal_px(w)
    views = [ViewCalibration(name=name, pan_deg=float(pan),
                             ignore_bottom_px=CAMERA_FIX_IGNORE_BOTTOM_PX,
                             # The forward view defines the kart's yaw; only
                             # the side views get a free pan.
                             nuisance=(abs(pan) > 1e-6
                                       and CAMERA_FIX_PAN_NUISANCE_SIGMA_DEG > 0))
             for name, pan in CAMERA_FIX_VIEWS]
    return CameraCalibration(
        width=w, height=h, fx=f, fy=f, cx=(w - 1) / 2.0, cy=(h - 1) / 2.0,
        lens_xyz=(CAMERA_LENS_FORWARD_M, CAMERA_LENS_LATERAL_M, CAMERA_LENS_HEIGHT_M),
        pan_arm_m=CAMERA_PAN_ARM_M, pitch_deg=fix_pitch_deg(),
        tilt_pct=CAMERA_FIX_TILT_PCT, pan_home_pct=CAMERA_PAN_HOME_PCT,
        deg_per_pct=CAMERA_SERVO_DEG_PER_PCT,
        pan_left_is_higher=CAMERA_PAN_LEFT_IS_HIGHER, views=views,
        tilt_deg_per_pct=CAMERA_SERVO_DEG_PER_PCT,
        tilt_down_is_higher=not CAMERA_TILT_DOWN_IS_LOWER,
        # v2's two tilts: "baixo" IS the held tilt (whatever the file says it
        # is), "cima" is derived from it until `extrinsics` measures it.
        tilts=[TiltCalibration("baixo"),
               TiltCalibration("cima", tilt_pct=CAMERA_V2_TILT_UP_PCT)])


def resolve_calib_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        path = CAMERA_CALIB_DEFAULT_PATH
    if not os.path.isabs(path):
        path = os.path.join(ORQUESTRATOR, path)
    return path


def load_camera_calibration(path: Optional[str] = None) -> Tuple[CameraCalibration, str]:
    """(calibration, human-readable source). A missing file falls back to the
    defaults; a broken one too, with the reason in the source string."""
    base = default_calibration()
    full = resolve_calib_path(path)
    if full is None or not os.path.exists(full):
        return base, "padrao (sem arquivo de calibracao)"
    try:
        return CameraCalibration.load(full, base), full
    except Exception as exc:                                     # noqa: BLE001
        return base, f"padrao (calibracao invalida: {exc})"


def build_view_geometries(calib: CameraCalibration,
                          max_range_m: float = CAMERA_FIX_MAX_FLOOR_RANGE_M,
                          col_step: int = CAMERA_FIX_COLUMN_STEP_PX,
                          row_step: int = CAMERA_FIX_ROW_STEP_PX,
                          **extra) -> Dict[str, ViewGeometry]:
    out = {}
    for view in calib.views:
        pin = PinholeView.from_calibration(calib, view, **extra)
        out[view.name] = build_view_geometry(
            view.name, pin, view.ignore_bottom_px, max_range_m, col_step,
            row_step, CAMERA_FIX_BEARING_STEP_DEG, nuisance=view.nuisance)
    return out


# -- camera pose fix v2 -------------------------------------------------------
def v2_views(calib: CameraCalibration,
             pans=CAMERA_V2_PAN_ORDER, tilts=CAMERA_V2_TILT_ORDER) -> List[ViewCalibration]:
    """The v2 views in VISITING order: for each pan, each tilt (up, then
    down). A pan or tilt the calibration does not have is skipped."""
    names = {v.name for v in calib.views}
    have = {t.name for t in calib.tilts}
    return [calib.head_view(calib.view(p), t)
            for p in pans if p in names for t in tilts if t in have]


def v2_pinholes(calib: CameraCalibration, views: Optional[List[ViewCalibration]] = None,
                **extra) -> Dict[str, PinholeView]:
    views = v2_views(calib) if views is None else views
    return {v.name: PinholeView.from_calibration(calib, v, **extra) for v in views}

