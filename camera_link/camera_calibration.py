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
Tilt is held at one value for the whole run (`tilt_pct`), so the lens position
is measured AT that tilt and tilting is not modelled at all.

Servo units: Server/servo.py drives `500 + (angle + 10) / 0.09` us, i.e. one
CMD_SERVO unit is one servo degree (`deg_per_pct`). The "percent" naming is
CameraMirror's, kept so both talk about the same number.

Every number here can be overridden by a JSON file (`load`), which is what the
calibration tool writes. Nothing in this module knows about mazes.
"""
from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import List, Tuple

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
    tilt_pct: int = 12
    pan_home_pct: int = 50
    deg_per_pct: float = 1.0
    pan_left_is_higher: bool = True
    views: List[ViewCalibration] = field(default_factory=list)
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

    def pan_pct(self, view: ViewCalibration) -> int:
        sign = 1.0 if self.pan_left_is_higher else -1.0
        return int(round(self.pan_home_pct + sign * view.pan_deg / self.deg_per_pct))

    def pose_center_cam(self, view: ViewCalibration,
                        extra_pan_deg: float = 0.0,
                        extra_pitch_deg: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """(R, C) for `view`: R maps camera-frame directions into the centre
        frame, C is the lens position in the centre frame.

        The `extra_*` arguments perturb the pose without editing the stored
        calibration -- the calibration tool searches over them.
        """
        pan = math.radians(view.pan_deg + view.pan_bias_deg + extra_pan_deg)
        pitch = math.radians(self.pitch_deg + view.tilt_bias_deg + extra_pitch_deg)
        roll = math.radians(view.roll_deg)
        R = _rot_z(pan) @ _rot_y(pitch) @ _R0 @ _rot_z(roll)
        lens = np.asarray(self.lens_xyz, dtype=float)
        arm = np.array([self.pan_arm_m, 0.0, 0.0])
        pivot = lens - arm
        C = pivot + _rot_z(pan) @ arm
        return R, C

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("source", None)
        d["dist"] = list(self.dist)
        d["lens_xyz"] = list(self.lens_xyz)
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
            if key in ("version", "views"):
                continue
            if hasattr(out, key):
                setattr(out, key, tuple(value) if isinstance(value, list) else value)
        if "views" in data:
            out.views = [ViewCalibration(**v) for v in data["views"]]
        if (out.width, out.height) != (base.width, base.height):
            raise ValueError(f"calibracao para {out.width}x{out.height}, "
                             f"stream e {base.width}x{base.height}")
        out.source = os.path.basename(path)
        return out
