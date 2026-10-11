"""
The ideal (undistorted) pinhole of one view, placed in the kart's centre frame.

Pure geometry over numpy arrays: project centre-frame points to pixels,
back-project pixels to rays, intersect rays with the floor. Pixel centres are
at integer coordinates (OpenCV's convention), and so is every row this module
returns -- the contact profile is defined on the same grid.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


class PinholeView:
    def __init__(self, f: float, cx: float, cy: float, width: int, height: int,
                 R: np.ndarray, C: np.ndarray):
        self.f = float(f)
        self.cx = float(cx)
        self.cy = float(cy)
        self.width = int(width)
        self.height = int(height)
        self.R = np.asarray(R, dtype=float)          # camera -> centre
        self.C = np.asarray(C, dtype=float)          # lens, centre frame

    @classmethod
    def from_calibration(cls, calib, view, **extra) -> "PinholeView":
        R, C = calib.pose_center_cam(view, **extra)
        cx, cy = calib.c_ideal
        return cls(calib.f_ideal, cx, cy, calib.width, calib.height, R, C)

    @property
    def foot(self) -> Tuple[float, float]:
        """The point of the floor straight under the lens. Visibility on the
        floor is decided from here: a floor point is hidden exactly when the
        segment from the foot to it crosses a box taller than the camera."""
        return float(self.C[0]), float(self.C[1])

    def rays(self, u, v) -> np.ndarray:
        """Centre-frame ray directions (not normalised) for pixels (u, v)."""
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        d_cam = np.stack([(u - self.cx) / self.f, (v - self.cy) / self.f,
                          np.ones(np.broadcast(u, v).shape)], axis=-1)
        return d_cam @ self.R.T

    def floor_points(self, u, v) -> np.ndarray:
        """Where the rays of pixels (u, v) meet the floor (z = 0), centre
        frame. NaN for rays at or above the horizon."""
        d = self.rays(u, v)
        dz = d[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(dz < -1e-9, -self.C[2] / dz, np.nan)
        return self.C + d * t[..., None]

    def project(self, pts) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Centre-frame points -> (u, v, depth). Depth <= 0 is behind."""
        p = np.asarray(pts, dtype=float) - self.C
        cam = p @ self.R                             # R^T applied to rows
        z = cam[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.cx + self.f * cam[..., 0] / z
            v = self.cy + self.f * cam[..., 1] / z
        return u, v, z

    def horizon_rows(self, u) -> np.ndarray:
        """Row of the horizon in each column `u` (rays with zero z component).
        Rows below (larger v) look at the floor."""
        u = np.asarray(u, dtype=float)
        r20, r21, r22 = self.R[2]
        x = (u - self.cx) / self.f
        with np.errstate(divide="ignore", invalid="ignore"):
            y = -(r20 * x + r22) / r21
        return self.cy + self.f * y
