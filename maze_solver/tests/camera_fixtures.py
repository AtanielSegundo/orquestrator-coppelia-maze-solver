"""
Shared fixtures for the camera pose fix tests: stub mazes, the default camera,
and synthetic measurements and frames rendered by the INDEPENDENT per-pixel ray
caster (runner/reference_renderer.py) -- never by the analytic model under
test, which must not be checked against itself.
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MAZE_SOLVER = os.path.dirname(HERE)
REPO = os.path.dirname(MAZE_SOLVER)
for _p in (REPO, MAZE_SOLVER):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2                                                        # noqa: E402
import numpy as np                                                # noqa: E402

import Coppelia.helpers as cop                                    # noqa: E402
from camera_link.contact_profile import extract_contact_profile   # noqa: E402
from camera_link.pinhole_view import PinholeView                  # noqa: E402
from runner.camera_fit import ViewMeasurement                     # noqa: E402
from runner.camera_setup import (build_view_geometries,           # noqa: E402
                                 default_calibration)
from runner.reference_renderer import FLOOR, WALL, render_labels  # noqa: E402

CELL = 0.30
ORIGIN = (0.0, 0.0, 0.0)
TCC = os.path.join(REPO, "tcc.maze")

CALIB = default_calibration()
GEOMS = build_view_geometries(CALIB)


class Stub:
    rows, cols = 3, 12
    obstacles = ()
    walls = frozenset()

    def in_bounds(self, c):
        return 0 <= c[0] < self.rows and 0 <= c[1] < self.cols

    def is_original_wall(self, c):
        if not self.in_bounds(c):
            return True
        return self._wall(c)

    def is_wall(self, c):
        return self.is_original_wall(c) or c in set(self.obstacles)

    def _wall(self, c):
        return c in self.walls


class Corridor(Stub):
    """Walls along rows 0 and 2: a corridor along +X, open at both ends."""
    def _wall(self, c):
        return c[0] in (0, 2)


class Pillars(Stub):
    rows, cols = 5, 8
    walls = frozenset({(1, 3), (3, 3), (2, 5), (0, 1), (4, 1), (1, 6)})


class OpenField(Stub):
    rows, cols = 7, 7

    def _wall(self, c):
        return False


def centre(cell):
    x, y, _ = cop.cell_to_world(ORIGIN, CELL, cell)
    return x, y


def offset_pose(cell, heading, along, lateral, yaw_deg):
    x, y = centre(cell)
    c, s = math.cos(heading), math.sin(heading)
    return (x + c * along - s * lateral, y + s * along + c * lateral,
            heading + math.radians(yaw_deg))


def measure(world, truth, pan_err=None, row_noise_px=0.0, rng=None,
            calib=CALIB, geoms=GEOMS):
    """Measured profiles of every view at `truth`, rendered per pixel, with an
    optional real pan error per view and Gaussian noise on the contact rows."""
    pan_err = pan_err or {}
    out = []
    for v in calib.views:
        pin = PinholeView.from_calibration(calib, v, extra_pan_deg=pan_err.get(v.name, 0.0))
        lab = render_labels(pin, truth, world)
        g = geoms[v.name]
        prof = extract_contact_profile(lab == WALL, g.columns, g.v_near, g.v_far, 3)
        if row_noise_px and rng is not None:
            prof = prof._replace(rows=prof.rows + rng.normal(0.0, row_noise_px,
                                                             prof.rows.shape))
        out.append(ViewMeasurement(g, prof, np.zeros(g.n_columns, dtype=bool)))
    return out


WALL_BGR = (97, 131, 178)        # CAIXA_COR, cardboard
FLOOR_BGR = (150, 150, 140)
ROOM_BGR = (70, 75, 80)


def synth_frame(labels, rng, body_rows=15, noise=6.0):
    """A camera-like BGR frame from a label image: flat colours, a vertical
    shading gradient, noise, a dark kart body at the bottom, JPEG q90."""
    h, w = labels.shape
    img = np.empty((h, w, 3), np.float32)
    img[:] = ROOM_BGR
    img[labels == FLOOR] = FLOOR_BGR
    img[labels == WALL] = WALL_BGR
    img *= np.linspace(0.85, 1.1, h)[:, None, None]
    img += rng.normal(0.0, noise, img.shape)
    if body_rows:
        img[h - body_rows:, :] = (20, 20, 20)
    img = np.clip(img, 0, 255).astype(np.uint8)
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return cv2.imdecode(jpg, cv2.IMREAD_COLOR)
