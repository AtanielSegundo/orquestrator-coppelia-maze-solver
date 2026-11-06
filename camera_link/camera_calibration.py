"""
Everything needed to turn a pixel of the kart's camera into a ray on the floor:
intrinsics, the undistorted "ideal" pinhole the rest of the pipeline works in,
and where the lens sits and points for each pan/tilt VIEW the pose fix uses.

Frames
------
* Centre frame: origin on the floor under the kart's odometry centre, x
  forward, y left, z up -- the frame the pose estimate lives in.
* Camera frame: OpenCV's, x right, y down, z along the optical axis.

`pose_center_cam(view)` gives (R, C): R takes a direction from the camera frame
to the centre frame, C is the lens position in the centre frame.

The pan/tilt head
-----------------
The pan servo rotates the camera about a vertical axis that sits `pan_arm_m`
BEHIND the lens, so a side view does not only turn the camera, it also moves it
sideways by roughly arm*sin(pan). At 2.5 cm and 45 deg that is 1.8 cm, which is
the size of the error being corrected, so it is modelled rather than ignored.
The camera pose fix v1 holds the tilt at one value for the whole run
(`tilt_pct`); `lens_xyz` is measured AT that tilt. The v2 fix (runner/
camera_fix_v2.py) also looks from other tilts -- `tilts`, e.g. "cima" to see
the box tops and "baixo" (the held tilt) to see the floor contacts. Each has
its own pitch (measured by `extrinsics`, or derived from the held tilt's pitch
and the tilt scale until then), and when the lens sits `tilt_arm_m` in front
of the tilt axis, tilting also moves it: that is modelled from `lens_xyz`.

A v2 VIEW is one pan view combined with one tilt (`head_view`): the pan biases
belong to the pan view, the pitch to the tilt -- the two servos' own errors.

Servo units: Server/servo.py drives `500 + (angle + 10) / 0.09` us, i.e. one
CMD_SERVO unit is NOMINALLY one servo degree. Real servos land short of that,
so `deg_per_pct` (pan) is measured by `camera_calibrate.py servo`, which also
measures the tilt (`tilt_deg_per_pct`, `tilt_level_pct`) and from it the
`pitch_deg` at `tilt_pct`. The "percent" naming is CameraMirror's, kept so
both talk about the same number.

Every number here can be overridden by a JSON file (`load`), which is what the
calibration tool writes. Nothing in this module knows about mazes.
"""
from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# Camera axes (x right, y down, z forward) expressed in the centre frame for a
# camera looking straight ahead, level: columns are the images of x, y, z.
_R0 = np.array([[0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0]])


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


@dataclass
class ViewCalibration:
    """One pan position of the head.

    `pan_deg` is the COMMANDED pan relative to home (+ = left, CCW seen from
    above). The biases are what the calibration tool measured on top of it:
    how far the real head lands from where the command says.

    `nuisance` lets the fit treat this view's pan as uncertain by a bounded
    amount (CAMERA_FIX_PAN_NUISANCE_*). Servo repeatability on the side views
    is the realistic case; the forward view defines the kart's yaw and so
    should not be free, or yaw becomes unobservable.
    """
    name: str
    pan_deg: float
    pan_bias_deg: float = 0.0
    tilt_bias_deg: float = 0.0
    roll_deg: float = 0.0
    ignore_bottom_px: int = 20
    nuisance: bool = True
    # v2 only (see `CameraCalibration.head_view`): the tilt this view looks
    # from ("" = the calibration's own `tilt_pct` / `pitch_deg`), and the pan
    # view it was made from. Never saved for the pan views themselves.
    tilt: str = ""
    pan_name: str = ""


@dataclass
class TiltCalibration:
    """One tilt position of the head the v2 fix looks from.

    `tilt_pct` None is the calibration's own `tilt_pct` (the tilt held during
    a run). `pitch_deg` None is DERIVED from the calibrated `pitch_deg` at
    `tilt_pct` and the tilt scale (`tilt_deg_per_pct`, `tilt_down_is_higher`)
    -- an estimate until `extrinsics` measures it and writes it here.
    """
    name: str
    tilt_pct: Optional[int] = None
    pitch_deg: Optional[float] = None


@dataclass
class CameraCalibration:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    # Lens position at home pan, in the centre frame.
    lens_xyz: Tuple[float, float, float] = (0.13, 0.0, 0.10)
    pan_arm_m: float = 0.025
    # Pitch DOWN of the optical axis at `tilt_pct`, degrees.
    pitch_deg: float = 25.0
    tilt_pct: int = 62
    pan_home_pct: int = 50
    # Pan degrees per servo unit.
    deg_per_pct: float = 1.0
    # Tilt mapping. v1 only uses `pitch_deg` (which `servo` derives from these
    # at `tilt_pct`); v2 uses the scale to derive the pitch of a tilt that
    # `extrinsics` has not measured yet (TiltCalibration.pitch_deg None).
    tilt_deg_per_pct: float = 1.0
    tilt_level_pct: float = 37.0
    pan_left_is_higher: bool = True
    # A HIGHER tilt unit looks further down (measured on the real kart).
    tilt_down_is_higher: bool = True
    # Lens in front of the tilt axis, m. 0 = tilting does not move the lens.
    tilt_arm_m: float = 0.0
    views: List[ViewCalibration] = field(default_factory=list)
    tilts: List[TiltCalibration] = field(default_factory=list)
    box_height_m: float = 0.30
    wall_inset_m: float = 0.0
    source: str = "padrao"

    # -- intrinsics --------------------------------------------------------
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]])

    @property
    def f_ideal(self) -> float:
        """Focal length of the undistorted pinhole everything downstream uses.
        Square pixels, principal point at the image centre -- so a model only
        ever needs one number."""
        return 0.5 * (self.fx + self.fy)

    @property
    def c_ideal(self) -> Tuple[float, float]:
        return (self.width - 1) / 2.0, (self.height - 1) / 2.0

    def K_ideal(self) -> np.ndarray:
        cx, cy = self.c_ideal
        f = self.f_ideal
        return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])

    def needs_remap(self) -> bool:
        """False when the stream already IS the ideal pinhole (no distortion,
        centred principal point): the remap would then be a costly identity."""
        cx, cy = self.c_ideal
        return (any(abs(d) > 1e-9 for d in self.dist)
                or abs(self.cx - cx) > 0.25 or abs(self.cy - cy) > 0.25
                or abs(self.fx - self.fy) > 0.25)

    # -- views -------------------------------------------------------------
    def view(self, name: str) -> ViewCalibration:
        for v in self.views:
            if v.name == name:
                return v
        raise KeyError(name)

    def _pan_source(self, view: ViewCalibration) -> ViewCalibration:
        """The pan view whose pan and biases `view` uses: itself, or for a v2
        view the calibration's CURRENT pan view -- not the copy made when the
        view was, which would miss every bias fitted since."""
        if view.pan_name:
            for v in self.views:
                if v.name == view.pan_name:
                    return v
        return view

    def pan_pct(self, view: ViewCalibration) -> int:
        sign = 1.0 if self.pan_left_is_higher else -1.0
        return int(round(self.pan_home_pct
                         + sign * self._pan_source(view).pan_deg / self.deg_per_pct))

    # -- tilts (v2) ----------------------------------------------------------
    def tilt(self, name: str) -> TiltCalibration:
        for t in self.tilts:
            if t.name == name:
                return t
        raise KeyError(name)

    def tilt_pct_of(self, t: TiltCalibration) -> int:
        return int(self.tilt_pct if t.tilt_pct is None else t.tilt_pct)

    def tilt_pitch_deg(self, t: TiltCalibration) -> float:
        """Down-pitch of the optical axis at tilt `t` (negative = above level)."""
        if t.pitch_deg is not None:
            return float(t.pitch_deg)
        sign = 1.0 if self.tilt_down_is_higher else -1.0
        return float(self.pitch_deg + sign * (self.tilt_pct_of(t) - self.tilt_pct)
                     * self.tilt_deg_per_pct)

    def view_tilt(self, view: ViewCalibration) -> Tuple[int, float]:
        """(tilt units, down-pitch deg) the head uses for `view`."""
        if not view.tilt:
            return int(self.tilt_pct), float(self.pitch_deg)
        t = self.tilt(view.tilt)
        return self.tilt_pct_of(t), self.tilt_pitch_deg(t)

    def head_view(self, pan_view: ViewCalibration, tilt_name: str) -> ViewCalibration:
        """The v2 view of pan view `pan_view` at tilt `tilt_name`, named
        "<pan>_<tilt>". Its pan and biases are always read from the pan view
        of whichever calibration places it (`_pan_source`)."""
        self.tilt(tilt_name)                       # KeyError on an unknown tilt
        v = copy.copy(pan_view)
        v.name = f"{pan_view.name}_{tilt_name}"
        v.tilt = tilt_name
        v.pan_name = pan_view.name
        return v

    def pose_center_cam(self, view: ViewCalibration,
                        extra_pan_deg: float = 0.0,
                        extra_pitch_deg: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """(R, C) for `view`: R maps camera-frame directions into the centre
        frame, C is the lens position in the centre frame.

        The `extra_*` arguments perturb the pose without editing the stored
        calibration -- the calibration tool searches over them.
        """
        _tilt_pct, base_pitch = self.view_tilt(view)
        src = self._pan_source(view)
        pan = math.radians(src.pan_deg + src.pan_bias_deg + extra_pan_deg)
        pitch = math.radians(base_pitch + src.tilt_bias_deg + extra_pitch_deg)
        roll = math.radians(src.roll_deg)
        R = _rot_z(pan) @ _rot_y(pitch) @ _R0 @ _rot_z(roll)
        ref = np.asarray(self.lens_xyz, dtype=float)       # at home pan, held tilt
        lens = ref
        if self.tilt_arm_m:
            # The lens swings about the tilt axis, `tilt_arm_m` behind it along
            # the optical axis; `lens_xyz` is where it is at the held pitch.
            p0 = math.radians(self.pitch_deg)
            lens = ref + self.tilt_arm_m * np.array([math.cos(pitch) - math.cos(p0), 0.0,
                                                     -(math.sin(pitch) - math.sin(p0))])
        pivot = ref - np.array([self.pan_arm_m, 0.0, 0.0])
        C = pivot + _rot_z(pan) @ (lens - pivot)
        return R, C

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("source", None)
        d["dist"] = list(self.dist)
        d["lens_xyz"] = list(self.lens_xyz)
        # Pan views never carry the v2-only fields: a file written here still
        # loads on a branch whose ViewCalibration does not know them.
        for v in d["views"]:
            if not v.get("tilt"):
                v.pop("tilt", None)
            if not v.get("pan_name"):
                v.pop("pan_name", None)
        return d

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fp:
            json.dump({"version": 1, **self.to_dict()}, fp, indent=2)

    @classmethod
    def load(cls, path: str, base: "CameraCalibration") -> "CameraCalibration":
        """`base` with every field present in the JSON at `path` replaced.

        Merging over a base (the hardware defaults) rather than requiring a
        complete file means a partial calibration -- say, intrinsics only --
        is still usable. A stream size different from the base's is refused:
        intrinsics measured at one resolution are wrong at another.
        """
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        out = copy.deepcopy(base)
        for key, value in data.items():
            if key in ("version", "views", "tilts"):
                continue
            if hasattr(out, key):
                setattr(out, key, tuple(value) if isinstance(value, list) else value)
        if "views" in data:
            known = ViewCalibration.__dataclass_fields__
            out.views = [ViewCalibration(**{k: val for k, val in v.items() if k in known})
                         for v in data["views"]]
        if "tilts" in data:
            # Merged by name over the base's: a file from before a tilt existed
            # keeps that tilt's default instead of losing it.
            known = TiltCalibration.__dataclass_fields__
            by_name = {t.name: t for t in out.tilts}
            for t in data["tilts"]:
                by_name[t["name"]] = TiltCalibration(**{k: v for k, v in t.items()
                                                        if k in known})
            out.tilts = list(by_name.values())
        if (out.width, out.height) != (base.width, base.height):
            raise ValueError(f"calibracao para {out.width}x{out.height}, "
                             f"stream e {base.width}x{base.height}")
        out.source = os.path.basename(path)
        return out
