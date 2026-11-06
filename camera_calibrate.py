"""
Bench tool for the camera pose fix (maze_solver/runner/camera_fix.py). Talks to
the kart directly (MJPEG + CMD_SERVO over HTTP) -- no CoppeliaSim, no runner.

    python camera_calibrate.py intrinsics  [-i HOST] [--board 9x6 --square 0.025]
    python camera_calibrate.py servo       [-i HOST] [--board 9x6 --square 0.025]
    python camera_calibrate.py check       [-i HOST] --maze tcc.maze --cell 3 6 --heading 180
    python camera_calibrate.py extrinsics  [-i HOST] --maze tcc.maze --cell 3 6 --heading 180
                                           [--altura]

`check` and `extrinsics` work on the camera pose fix v2 views (runner/
camera_fix_v2.py): every pan view at every tilt -- "cima" for the box tops,
"baixo" (the held tilt) for the floor contacts -- compared by the box EDGES
(runner/edge_match.py). `--v1` runs the old floor-contact versions instead.
Pick a cell the v2 schedule would pick (runner/camera_cells.py): the cell ahead
open, walls two cells ahead and on the sides -- e.g. tcc.maze (3, 6) facing
180; `check` says so when the cell is a poor one.

intrinsics
    Live view of the stream; SPACE grabs a frame when the chessboard is found,
    C calibrates (10+ grabs, spread over the whole image and a few tilts), Q
    quits. Writes fx, fy, cx, cy and the distortion to the calibration file.
    Without this the nominal Pi Camera rev 1.3 optics are used (f ~ 402 px).

servo
    Measures what a servo unit really turns the camera by. Stand the chessboard
    UPRIGHT with its edges level (taped to a box face), ~50-70 cm in front of
    the kart at camera height, battery charged. The head sweeps pan, then tilt,
    as far as the board stays in view, both ways; solvePnP gives the camera's
    real rotation at each step and a line through them gives pan deg/unit and
    direction, tilt deg/unit, the level tilt and the pitch at the fix tilt.
    Resets the per-view biases: run `extrinsics` afterwards.

check
    Place the kart by hand on the CENTRE of the given cell, facing the given
    grid heading (0 right/east, 90 up/north, 180 left/west, 270 down/south),
    exactly as for 'ok' in the runner. Runs ONE measurement of the v2 fix there
    (nothing applied, the kart does not turn) and prints the estimate -- ideally
    close to zero -- and each group's deltas: the centre pan's left and right
    half planes, each side pan, tops and floor contacts apart. Debug PNGs per
    view: the real box boundary (green), the simulated edges at the informed
    pose (red) and at the estimate (cyan); and *_sim_<view>.png: the real frame
    | the simulated one | the real one with the simulated outline at the
    informed (red) and estimated (cyan) poses. A mirrored or vertically
    shifted model there means a sign or a tilt is wrong.

extrinsics
    Same placement, then fits what the v2 views need to line their simulated
    edges up with the real ones, the pose held at the informed one: the pitch
    of each tilt, the pan offset of each pan view, and with --altura the lens
    height too (box tops and floor contacts together separate it from the
    pitch). A coarse grid per parameter first (+-span), then the joint fit.
    Writes the calibration file: the held tilt's pitch as `pitch_deg`, the
    others into `tilts`, the pan offsets into each pan view. Place the kart as
    carefully as possible -- its yaw goes into the centre pan's offset.
    Repeat at 2-3 cells and keep the run whose `check` reads closest to zero.
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import shutil
import types
import re
import socket
import subprocess
import sys
import time

import cv2
import numpy as np
import pyray as pr

_HERE = os.path.dirname(os.path.abspath(__file__))
_MAZE_SOLVER = os.path.join(_HERE, "maze_solver")
for p in (_HERE, _MAZE_SOLVER):
    if p not in sys.path:
        sys.path.insert(0, p)

import Coppelia.helpers as cop                                   # noqa: E402
from camera_link.mjpeg_camera import MjpegCamera                 # noqa: E402
from camera_link.pinhole_view import PinholeView                 # noqa: E402
from camera_link.servo_head import ServoHead                     # noqa: E402
from shadow import grid_heading_to_world_theta                   # noqa: E402
from src.renders import MazeView                                 # noqa: E402
from src.solver import resolve_maze                              # noqa: E402

from runner.arrival_policy import plan_after_camera             # noqa: E402
from runner.camera_fit import FitParams, _rho                    # noqa: E402
from runner.camera_cells import camera_cell                      # noqa: E402
from runner.camera_fix import CameraFix                          # noqa: E402
from runner.camera_fix_v2 import (CameraFixV2, PoseParams,       # noqa: E402
                                  make_segmenter, measure_prepared)
from runner.camera_setup import (build_view_geometries,          # noqa: E402
                                 load_camera_calibration, resolve_calib_path,
                                 v2_views)
from runner.edge_match import matching_cost, real_edges, solve_edges  # noqa: E402
from runner.constants import (CAMERA_FIX_BEARING_STEP_DEG,       # noqa: E402
                              CAMERA_FIX_COLUMN_STEP_PX,
                              CAMERA_FIX_MAX_FLOOR_RANGE_M,
                              CAMERA_FIX_ROW_STEP_PX, CAMERA_FIX_STEP_MODE,
                              CAMERA_TILT_HOME_PCT, CAMERA_V2_SPLIT_PAN,
                              KART_HOST, KART_PORT)
from runner.contact_model import interval_gaps, predict_at       # noqa: E402
from runner.reference_renderer import FLOOR, WALL, render_labels  # noqa: E402
from runner.view_geometry import build_view_geometry             # noqa: E402
from runner.wall_world import build_wall_world                   # noqa: E402

CELL = 0.30
ORIGIN = (0.0, 0.0, 0.0)
LOG_DIR = os.path.join(_MAZE_SOLVER, "logs", "camera")


def _resolve(host: str) -> str:
    """IPv4 of `host`. mDNS names (pikart.local) do not always resolve through
    the normal resolver on Windows, so fall back to `ping`, which does -- the
    same trick get_cam.py uses."""
    try:
        return socket.getaddrinfo(host, None, socket.AF_INET)[0][4][0]
    except OSError:
        pass
    out = subprocess.run(["ping", host, "-4", "-n", "1"], capture_output=True, text=True)
    m = re.search(r"\[(\d{1,3}(?:\.\d{1,3}){3})\]", out.stdout)
    if m:
        return m.group(1)
    raise SystemExit(f"[ERROR] nao consegui resolver '{host}' -- passe o IP com -i")


def _urls(args):
    base = f"http://{args.ip}:{args.port}"
    return base + "/video", base + "/command"


def _open_stream(args) -> MjpegCamera:
    """Connect to the kart's MJPEG stream, failing fast and clearly.

    A plain TCP connect first (3 s): a wrong IP or a kart that is off then says
    so at once, instead of after the stream reader's 10 s urlopen timeout. The
    first frame is given longer -- the server only starts Picamera2 when the
    first viewer arrives.
    """
    ip = _resolve(args.ip)
    if ip != args.ip:
        print(f"[INFO] {args.ip} -> {ip}")
    args.ip = ip
    print(f"[INFO] kart em http://{ip}:{args.port}")
    try:
        socket.create_connection((ip, args.port), timeout=3.0).close()
    except OSError as exc:
        raise SystemExit(
            f"[ERROR] sem conexao com {ip}:{args.port} ({exc}).\n"
            f"        O kart esta ligado e o servidor rodando? O IP certo aparece no "
            f"OLED do kart;\n        passe com -i <ip> (ou -i pikart.local). "
            f"O padrao vem de KART_HOST em runner/constants.py.")
    video_url, _ = _urls(args)
    stream = MjpegCamera(video_url)
    t_end = time.monotonic() + 15.0
    while time.monotonic() < t_end:
        frame, _count, _t = stream.latest_stamped()
        if frame is not None:
            return stream
        time.sleep(0.05)
    # Stop the reader before exiting: a daemon thread still printing while the
    # interpreter shuts down makes CPython abort.
    stream.stop()
    raise SystemExit(f"[ERROR] conectou em {ip}:{args.port} mas nenhum frame chegou "
                     f"de {video_url} em 15 s (camera ligada no kart?)")


def _find_board(gray, size):
    found, corners = cv2.findChessboardCornersSB(gray, size)
    return (corners if found else None)


# ---------------------------------------------------------------------------
def cmd_intrinsics(args):
    calib, source = load_camera_calibration(args.calib)
    cols, rows = (int(v) for v in args.board.lower().split("x"))
    stream = _open_stream(args)
    first, _c, _t = stream.latest_stamped()
    h, w = first.shape[:2]
    if (w, h) != (calib.width, calib.height):
        print(f"[WARN] stream {w}x{h}, calibracao espera {calib.width}x{calib.height}")
    # --board counts INNER corners, which a printed board with N x M squares
    # has (N-1) x (M-1) of. Passing the square count is the usual mistake, so
    # the smaller size is tried too and adopted if that is what is in view.
    sizes = [(cols, rows)]
    if cols > 3 and rows > 3:
        sizes.append((cols - 1, rows - 1))
    size = None
    obj_pts, img_pts = [], []
    print(f"tabuleiro: {cols}x{rows} cantos internos (ou {cols - 1}x{rows - 1}, se "
          f"{cols}x{rows} for o numero de casas), casa de {args.square * 100:.1f} cm")
    print("SPACE captura | C calibra | Q sai")
    # The window is raylib's, not cv2.imshow: opencv-python-headless (pulled in
    # by vllm) shadows the GUI build of cv2 and has no highgui.
    pr.set_trace_log_level(pr.TraceLogLevel.LOG_WARNING)
    pr.set_target_fps(30)
    pr.init_window(2 * w, 2 * h, "intrinsics")
    blank = pr.gen_image_color(w, h, pr.BLACK)
    texture = pr.load_texture_from_image(blank)
    pr.unload_image(blank)
    try:
        while not pr.window_should_close():
            frame, _c, _t = stream.latest_stamped()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = None
            for cand in ([size] if size else sizes):
                corners = _find_board(gray, cand)
                if corners is not None:
                    if size is None:
                        size = cand
                        print(f"[INFO] tabuleiro detectado com {cand[0]}x{cand[1]} "
                              f"cantos internos")
                    break
            found = corners is not None
            show = frame.copy()
            if found:
                cv2.drawChessboardCorners(show, size, corners, found)
            cv2.putText(show, f"{len(img_pts)} capturas", (6, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            rgba = np.ascontiguousarray(cv2.cvtColor(show, cv2.COLOR_BGR2RGBA))
            pr.update_texture(texture, pr.ffi.cast("void *", pr.ffi.from_buffer(rgba)))
            pr.begin_drawing()
            pr.clear_background(pr.BLACK)
            pr.draw_texture_pro(texture, pr.Rectangle(0, 0, w, h),
                                pr.Rectangle(0, 0, pr.get_screen_width(),
                                             pr.get_screen_height()),
                                pr.Vector2(0, 0), 0.0, pr.WHITE)
            pr.end_drawing()
            if pr.is_key_pressed(pr.KeyboardKey.KEY_SPACE) and found:
                objp = np.zeros((size[0] * size[1], 3), np.float32)
                objp[:, :2] = (np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)
                               * args.square)
                obj_pts.append(objp)
                img_pts.append(corners)
                print(f"  captura {len(img_pts)}")
            elif pr.is_key_pressed(pr.KeyboardKey.KEY_C) and len(img_pts) >= 10:
                rms, K, dist, _r, _tv = cv2.calibrateCamera(
                    obj_pts, img_pts, (w, h), None, None,
                    flags=cv2.CALIB_FIX_K3)
                print(f"[INFO] rms {rms:.3f} px\nK=\n{K}\ndist={dist.ravel()}")
                calib.fx, calib.fy = float(K[0, 0]), float(K[1, 1])
                calib.cx, calib.cy = float(K[0, 2]), float(K[1, 2])
                calib.dist = tuple(float(v) for v in dist.ravel()[:5])
                path = resolve_calib_path(args.calib)
                calib.save(path)
                print(f"[INFO] salvo em {path}")
                if rms > 0.5:
                    print("[WARN] rms acima de 0.5 px -- capture mais vistas variadas")
            elif pr.is_key_pressed(pr.KeyboardKey.KEY_Q):
                break
    finally:
        stream.stop()
        pr.unload_texture(texture)
        pr.close_window()


# ---------------------------------------------------------------------------
class _StaticLink:
    """No telemetry: the servo ack is skipped and the kart is taken as still."""
    def servo_telemetry(self):
        return {}


def _placement(args):
    maze = resolve_maze(args.maze)
    if maze is None:
        raise SystemExit(f"[ERROR] maze nao encontrado: {args.maze}")
    view = MazeView(maze)
    cell = (int(args.cell[0]), int(args.cell[1]))
    if not view.in_bounds(cell) or view.is_original_wall(cell):
        raise SystemExit(f"[ERROR] celula {cell} invalida")
    x, y, _ = cop.cell_to_world(ORIGIN, CELL, cell)
    return view, (x, y, grid_heading_to_world_theta(args.heading))


def _run_fix_v2(args, calib, view, pose, measure_only=False, margin_px=None):
    """One v2 measurement at `pose` (nothing applied, no turns)."""
    stream = _open_stream(args)
    _video_url, command_url = _urls(args)
    head = ServoHead(command_url)
    world = build_wall_world(view, ORIGIN, CELL, inset=calib.wall_inset_m,
                             box_height=calib.box_height_m)
    quality = camera_cell(view, _cell_of(pose), pose[2])
    if not quality.ok:
        print(f"[WARN] celula ruim para a camera v2: {quality.reason} -- o runner "
              f"nao rodaria aqui")
    fix = CameraFixV2(calib, head, stream, _StaticLink(), world, odom_ref=(0.0, 0.0, 0.0),
                      purpose="bancada", act=False, measure_only=measure_only,
                      dump_dir=None if measure_only else LOG_DIR)
    if margin_px is not None:
        fix.segmenter.margin_px = margin_px
    still = {"goal_active": False}
    try:
        while fix.poll(still, (0.0, 0.0, 0.0), *pose) == "pending":
            time.sleep(1 / 60)
    finally:
        stream.stop()
    return fix, world


def _cell_of(pose):
    rf, cf = cop.world_to_cell(ORIGIN, CELL, pose[0], pose[1])
    return int(round(rf)), int(round(cf))


def _as_geoms(views):
    """{name: (pinhole, pose)} -> what _save_sim takes: ({name: .pinhole},
    {name: pose})."""
    return ({n: types.SimpleNamespace(pinhole=pin) for n, (pin, _p) in views.items()},
            {n: pose for n, (_pin, pose) in views.items()})


def _run_fix(args, calib, geoms, view, pose, margin_px=None):
    stream = _open_stream(args)
    _video_url, command_url = _urls(args)
    head = ServoHead(command_url)
    world = build_wall_world(view, ORIGIN, CELL, inset=calib.wall_inset_m,
                             box_height=calib.box_height_m)
    fix = CameraFix(calib, geoms, head, stream, _StaticLink(), world,
                    odom_ref=(0.0, 0.0, 0.0), purpose="bancada", dump_dir=LOG_DIR)
    if margin_px is not None:
        fix.segmenter.margin_px = margin_px
    still = {"goal_active": False}
    try:
        while fix.poll(still, (0.0, 0.0, 0.0), *pose) == "pending":
            time.sleep(1 / 60)
    finally:
        stream.stop()
    return fix, world


_SIM_COLORS = {FLOOR: (170, 160, 150), WALL: (70, 110, 160)}   # BGR: floor, cardboard


def _wall_outline(labels):
    wall = (labels == WALL).astype(np.uint8)
    return cv2.morphologyEx(wall, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0


def _caption(img, text):
    cv2.putText(img, text, (6, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1)
    return img


def _save_sim(fix, world, sim_geoms, sim_pose, overlays, tag):
    """One PNG per view: the real frame | the maze ray caster's simulated frame
    | the real frame with simulated wall outlines on top. `overlays` is a list
    of (geoms, {view: pose}, BGR colour, legend), one outline each."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    paths = []
    for name, geom in sim_geoms.items():
        sim = render_labels(geom.pinhole, sim_pose, world)
        sim_img = np.full(sim.shape + (3,), 40, np.uint8)
        for k, c in _SIM_COLORS.items():
            sim_img[sim == k] = c
        panels = [_caption(sim_img, f"{name}: simulado")]
        real = fix.frames.get(name)
        if real is not None:
            over = real.copy()
            for geoms, poses, color, _legend in overlays:
                over[_wall_outline(render_labels(geoms[name].pinhole, poses[name],
                                                 world))] = color
            legend = " / ".join(lg for *_r, lg in overlays)
            panels = [_caption(real.copy(), f"{name}: real")] + panels + \
                     [_caption(over, legend)]
        path = os.path.join(LOG_DIR, f"{stamp}_{tag}_sim_{name}.png")
        if cv2.imwrite(path, np.hstack(panels)):
            paths.append(path)
    return paths


def cmd_check(args):
    if not args.v1:
        return cmd_check_v2(args)
    calib, source = load_camera_calibration(args.calib)
    print(f"[INFO] calibracao: {source}")
    view, pose = _placement(args)
    geoms = build_view_geometries(calib)
    fix, world = _run_fix(args, calib, geoms, view, pose)
    res = fix.result
    print(f"[INFO] {res.summary()}  ({res.duration_s:.1f} s)")
    informed = {name: fix.capture_pose for name in geoms}
    overlays = [(geoms, informed, (0, 0, 255), "pose informada (verm.)")]
    if res.fit is not None:
        a, b, psi = res.fit.estimate
        print(f"       estimativa bruta: long {a * 100:+.1f} cm, lat {b * 100:+.1f} cm, "
              f"yaw {math.degrees(psi):+.2f} deg, pans "
              + ", ".join(f"{k} {math.degrees(v):+.1f}" for k, v in res.fit.etas.items()))
        x0, y0, th0 = fix.capture_pose
        c, s = math.cos(th0), math.sin(th0)
        x, y, th = x0 + c * a - s * b, y0 + s * a + c * b, th0 + psi
        corrected = {name: (x, y, th + res.fit.etas.get(name, 0.0)) for name in geoms}
        overlays.append((geoms, corrected, (255, 255, 0), "corrigida (ciano)"))
        step = plan_after_camera(res.fit, 0.0, "medicao", 0.0, False)
        if CAMERA_FIX_STEP_MODE:
            print(f"       passo que o runner aplicaria: long {step.longitudinal * 100:+.1f} cm, "
                  f"lat {step.lateral * 100:+.1f} cm, yaw {math.degrees(step.yaw):+.2f} deg"
                  + ("" if step.applies else " (nada: confirmado/recusado)"))
            if step.applies:
                sa, sb = step.longitudinal, step.lateral
                stepped = (x0 + c * sa - s * sb, y0 + s * sa + c * sb, th0 + step.yaw)
                overlays.append((geoms, {n: stepped for n in geoms}, (0, 255, 0),
                                 "apos o passo (verde)"))
    for p in res.debug_paths:
        print(f"       {p}")
    for p in _save_sim(fix, world, geoms, fix.capture_pose, overlays, "bancada"):
        print(f"[SIM]  {p}")


def cmd_check_v2(args):
    calib, source = load_camera_calibration(args.calib)
    print(f"[INFO] calibracao: {source}")
    view, pose = _placement(args)
    fix, world = _run_fix_v2(args, calib, view, pose)
    res = fix.result
    print(f"[INFO] {res.summary()}  ({res.duration_s:.1f} s)")
    for name, why in fix.dropped.items():
        print(f"       {name}: descartada ({why})")
    for line in fix.group_lines():
        print(f"       {line}")
    if fix.fit is not None:
        for name, why in fix.fit.dropped.items():
            print(f"       grupo {name} fora: {why}")
    for p in res.debug_paths:
        print(f"       {p}")
    if fix.params is None:
        return
    g0, p0 = _as_geoms(fix.params.build(fix.params.x0))
    overlays = [(g0, p0, (0, 0, 255), "pose informada (verm.)")]
    if fix.fit is not None:
        g1, p1 = _as_geoms(fix.params.build(fix.fit.x))
        overlays.append((g1, p1, (255, 255, 0), "estimada (ciano)"))
    for p in _save_sim(fix, world, g0, pose, overlays, "bancada"):
        print(f"[SIM]  {p}")


class _ExtrinsicParams:
    """x = (pitch offset per tilt deg, pan offset per SIDE pan deg, tilt bias
    per side pan deg[, lens height offset m], kart yaw deg, kart long m, kart
    lat m).

    The side pans' tilt biases are there because the head does not land on
    the same pitch at every pan: on the real kart the two side views' floor
    contacts sat 3 and 6-7 deg apart from the model that fitted the centre.
    The centre pan has none -- its pitch IS the tilt's.

    The last three are the kart's PLACEMENT, free with a prior and never
    saved: a kart placed 5 cm off the centre, held at the informed pose, bent
    every angle to make up for it (measured on the real kart: the front box's
    corners said 37 cm where the model had 32). The centre pan has no pan
    offset for the same reason -- it would be the kart's yaw twice."""

    PLACEMENT = ("kart_yaw", "kart_long", "kart_lat")

    def __init__(self, calib, views, pose, fit_height, split_pan=CAMERA_V2_SPLIT_PAN):
        self.calib, self.views, self.pose = calib, list(views), pose
        self.tilts = list(dict.fromkeys(v.tilt for v in self.views))
        self.pans = list(dict.fromkeys(v.pan_name for v in self.views))
        self.side = [p for p in self.pans if p != split_pan]
        self.names = ([f"pitch_{t}" for t in self.tilts] + [f"pan_{p}" for p in self.side]
                      + [f"tilt_{p}" for p in self.side] + (["altura"] if fit_height else [])
                      + list(self.PLACEMENT))
        self.fit_height = fit_height
        n_ang = len(self.tilts) + 2 * len(self.side)
        h = [0.03] if fit_height else []
        self.x0 = np.zeros(len(self.names))
        # Placement: the kart is put on the centre by hand, good to a couple
        # of cm / degrees. Looser, it trades against the pitch (kart further
        # back + camera less pitched gives the same floor contact): measured
        # on the real kart, a 5 cm prior let it slide 15 cm.
        self.sigma = np.array([15.0] * n_ang + h + [2.0, 0.02, 0.02])
        self.steps = np.array([0.2] * n_ang + ([0.002] if fit_height else [])
                              + [0.2, 0.005, 0.005])
        self.max_step = np.array([1.0] * n_ang + ([0.005] if fit_height else [])
                                 + [1.0, 0.01, 0.01])
        # Coarse grid per parameter: (half width, step); None = the --span ones.
        self.grid = {"kart_yaw": (3.0, 1.0), "kart_long": (0.04, 0.01),
                     "kart_lat": (0.03, 0.01), "altura": None}

    def _index(self, kind, key=None):
        name = kind if key is None else f"{kind}_{key}"
        return self.names.index(name) if name in self.names else None

    def placed(self, x):
        """The kart pose the parameters put it at."""
        x0, y0, th0 = self.pose
        a, b = float(x[self._index("kart_long")]), float(x[self._index("kart_lat")])
        c, s = math.cos(th0), math.sin(th0)
        return (x0 + c * a - s * b, y0 + s * a + c * b,
                th0 + math.radians(float(x[self._index("kart_yaw")])))

    def build(self, x):
        out = {}
        pose = self.placed(x)
        for v in self.views:
            pitch = float(x[self._index("pitch", v.tilt)])
            pan = 0.0
            if v.pan_name in self.side:
                pitch += float(x[self._index("tilt", v.pan_name)])
                pan = float(x[self._index("pan", v.pan_name)])
            pin = PinholeView.from_calibration(self.calib, v, extra_pitch_deg=pitch,
                                               extra_pan_deg=pan)
            if self.fit_height:
                pin = PinholeView(pin.f, pin.cx, pin.cy, pin.width, pin.height, pin.R,
                                  pin.C + np.array([0.0, 0.0, float(x[self._index("altura")])]))
            out[v.name] = (pin, pose)
        return out

    def affected(self, i):
        """Views parameter i moves."""
        name = self.names[i]
        kind, _, key = name.partition("_")
        if kind == "pitch":
            return [v.name for v in self.views if v.tilt == key]
        if kind in ("pan", "tilt"):
            return [v.name for v in self.views if v.pan_name == key]
        return [v.name for v in self.views]

    def applied(self, x):
        """A COPY of the calibration with x folded in (placement left out) --
        the original is never touched, so 'before' can still be drawn."""
        c = copy.deepcopy(self.calib)
        new_pitch = {t: c.tilt_pitch_deg(c.tilt(t)) + float(x[self._index("pitch", t)])
                     for t in self.tilts}
        held = [t for t in self.tilts
                if c.tilt(t).tilt_pct is None and c.tilt(t).pitch_deg is None]
        for t in held:                          # the held tilt IS pitch_deg
            c.pitch_deg = new_pitch[t]
        for t in self.tilts:
            if t not in held:
                c.tilt(t).pitch_deg = new_pitch[t]
        for p in self.side:
            c.view(p).pan_bias_deg += float(x[self._index("pan", p)])
            c.view(p).tilt_bias_deg += float(x[self._index("tilt", p)])
        if self.fit_height:
            lx, ly, lz = c.lens_xyz
            c.lens_xyz = (lx, ly, lz + float(x[self._index("altura")]))
        return c


def _per_view_start(par, reals, world, real_pts, span, span_tilt, log):
    """Starting x from each view ON ITS OWN: a 2-D grid of (pitch, pan) offsets
    per view (coarse 3 deg, then 1 deg around the best), then folded into the
    parameters -- each tilt's pitch from the centre pan, each side pan's pan
    and tilt bias from its two views. Independent views cannot hold each
    other back the way a shared pitch does in coordinate descent (measured on
    the real kart: the held tilt's pitch would not move to the floor contact
    while the side views' biases were still wrong)."""
    best_of = {}
    for name, real in reals.items():
        v = next(vv for vv in par.views if vv.name == name)

        def cost(dp, dn):
            pin = PinholeView.from_calibration(par.calib, v, extra_pitch_deg=dp,
                                               extra_pan_deg=dn)
            return matching_cost({name: real}, world, {name: (pin, par.pose)}, 30.0, 6.0,
                                 obstacles=world.obstacles, real_pts=real_pts)
        pan_lim = 0.0 if v.pan_name not in par.side else span
        best = (math.inf, 0.0, 0.0)
        for dp in np.arange(-span_tilt, span_tilt + 1e-9, 3.0):
            for dn in np.arange(-pan_lim, pan_lim + 1e-9, 3.0):
                c = cost(dp, dn)
                if c < best[0]:
                    best = (c, dp, dn)
        c0, p0, n0 = best
        for dp in p0 + np.arange(-2.0, 2.0 + 1e-9, 1.0):
            for dn in (n0 + np.arange(-2.0, 2.0 + 1e-9, 1.0)) if pan_lim else [0.0]:
                c = cost(dp, dn)
                if c < best[0]:
                    best = (c, dp, dn)
        best_of[name] = best
        log(f"  {name}: pitch {best[1]:+.1f}, pan {best[2]:+.1f} (custo {best[0]:.1f})")
    x = par.x0.copy()
    centre = {v.tilt: best_of[v.name][1] for v in par.views
              if v.pan_name not in par.side and v.name in best_of}
    for t in par.tilts:
        if t in centre:
            x[par._index("pitch", t)] = centre[t]
        else:                                  # no centre view at this tilt
            vals = [best_of[v.name][1] for v in par.views if v.tilt == t and v.name in best_of]
            x[par._index("pitch", t)] = float(np.mean(vals)) if vals else 0.0
    for p in par.side:
        mine = [v for v in par.views if v.pan_name == p and v.name in best_of]
        if not mine:
            continue
        x[par._index("pan", p)] = float(np.mean([best_of[v.name][2] for v in mine]))
        x[par._index("tilt", p)] = float(np.mean(
            [best_of[v.name][1] - x[par._index("pitch", v.tilt)] for v in mine]))
    return x


def _extrinsics_round(calib, views, prepared, pose, world, args, span, span_tilt,
                      step, log, span_tilt_side=None, per_view=False):
    """One round: segment with `calib`, coarse grids (two sweeps of coordinate
    descent on the SYMMETRIC cost), then the joint fit. Returns (params, x,
    fit, reals, cost before, cost after, edge hits)."""
    seg = make_segmenter()
    seg.margin_px = 25          # the calibration being fitted may be degrees off
    reals = {}
    for v in views:
        if v.name not in prepared:
            continue
        real, why, _img = measure_prepared(calib, v, prepared[v.name], pose, world, seg,
                                           cluster=True)
        if real is None:
            log(f"       {v.name}: descartada ({why})")
        else:
            reals[v.name] = real
    if not reals:
        return None
    use = [v for v in views if v.name in reals]
    par = _ExtrinsicParams(calib, use, pose, args.altura)
    real_pts = {n: real_edges(r) for n, r in reals.items()}
    if span_tilt_side is None:
        span_tilt_side = min(span, 10.0)
    search = 30.0

    def cost(x, names=None):
        sub = {n: r for n, r in reals.items() if names is None or n in names}
        built = par.build(x)
        return matching_cost(sub, world, {n: built[n] for n in sub}, search, 6.0,
                             obstacles=world.obstacles, real_pts=real_pts)

    x = par.x0.copy()
    before = cost(x)
    edge_hits = []
    if per_view:
        x = _per_view_start(par, reals, world, real_pts, span, span_tilt, log)
    for sweep in range(1 if per_view else 0, 2):
        # A side pan's pan and tilt first, on a JOINT grid: on a side view a
        # box edge runs diagonally and moves almost the same for either, so
        # one at a time settles in the wrong valley (measured on the real
        # kart: the left pan went -7 deg where its corners asked for +10).
        for p in par.side:
            ip, it = par._index("pan", p), par._index("tilt", p)
            if sweep == 0:
                pans = np.arange(-span, span + 1e-9, 1.5 * step)
                tilts = np.arange(-span_tilt_side, span_tilt_side + 1e-9, 1.5 * step)
            else:
                pans = x[ip] + np.arange(-2 * step, 2 * step + 1e-9, step / 2)
                tilts = x[it] + np.arange(-2 * step, 2 * step + 1e-9, step / 2)
            best = (math.inf, x[ip], x[it])
            for vp in pans:
                for vt in tilts:
                    xc = x.copy()
                    xc[ip], xc[it] = vp, vt
                    c = cost(xc, par.affected(ip))
                    if c < best[0]:
                        best = (c, vp, vt)
            x[ip], x[it] = best[1], best[2]
            if sweep == 0:
                if abs(abs(best[1]) - pans[-1]) < 1e-6:
                    edge_hits.append(f"pan_{p}")
                if abs(abs(best[2]) - tilts[-1]) < 1e-6:
                    edge_hits.append(f"tilt_{p}")
        for i, name in enumerate(par.names):
            if name in par.grid and par.grid[name] is None:
                continue                      # the height: joint fit only
            if name.startswith(("pan_", "tilt_")):
                continue                      # done above, jointly
            if name in par.grid:
                lim, st = par.grid[name]
            else:
                lim, st = (span_tilt if name.startswith("pitch_") else span), step
            if sweep == 0:
                grid = np.arange(-lim, lim + 1e-9, st)
            else:                             # second sweep: finer, around the first
                grid = x[i] + np.arange(-3 * st, 3 * st + 1e-9, st / 2)
            best = (math.inf, x[i])
            for val in grid:
                xc = x.copy()
                xc[i] = val
                c = cost(xc, par.affected(i))
                if c < best[0]:
                    best = (c, val)
            x[i] = best[1]
            if sweep == 0 and abs(abs(best[1]) - lim) < 1e-6:
                edge_hits.append(name)
        log(f"  grade {sweep + 1}: " + ", ".join(
            f"{n} {v * (100 if n in ('kart_long', 'kart_lat') else 1):+.1f}"
            for n, v in zip(par.names, x)))
    fit = solve_edges(reals, world, par.build, x, par.sigma, par.steps,
                      obstacles=world.obstacles, group_max_residual=math.inf,
                      iterations=args.passes, max_step=par.max_step)
    after = cost(fit.x)
    return par, fit.x, fit, reals, before, after, edge_hits


def cmd_extrinsics_v2(args):
    calib, source = load_camera_calibration(args.calib)
    print(f"[INFO] calibracao de partida: {source}")
    view, pose = _placement(args)
    fix, world = _run_fix_v2(args, calib, view, pose, measure_only=True, margin_px=25)
    if not fix.prepared:
        raise SystemExit(f"[ERROR] nenhuma vista capturada: {fix.result.reason}")
    views = list(fix.views)

    # Three rounds, each re-segmenting every view with the previous round's
    # calibration and starting from the previous round's placement: colours a
    # model degrees off could not tell apart usually can be once it is closer.
    # Rounds 1-2 start from per-view grids (wide); round 3 refines narrowly.
    # Measured on the real kart: with two rounds the up tilt's pitch stayed
    # 11 deg off -- right segmentation only came in round 2, whose window was
    # too narrow to reach it.
    rounds = [(args.span, args.span_tilt, args.step, True, None),
              (min(args.span, 10.0), 15.0, args.step, True, None),
              (4.0, 4.0, 0.5, False, 4.0)]
    cur_calib, cur_pose = calib, pose
    total = {}
    for k, (sp, spt, st, per_view, sts) in enumerate(rounds, 1):
        print(f"[INFO] rodada {k}")
        r = _extrinsics_round(cur_calib, views, fix.prepared, cur_pose, world, args,
                              sp, spt, st, print, span_tilt_side=sts, per_view=per_view)
        if r is None:
            raise SystemExit(f"[ERROR] nenhuma vista utilizavel na rodada {k}")
        par2, x2, fit2, reals2, before2, after2, hits2 = r
        print(f"[INFO] rodada {k}: custo {before2:.1f} -> {after2:.1f}")
        if k == 1:
            before, after1, hits = before2, after2, []
        for name, v in zip(par2.names, x2):
            total[name] = total.get(name, 0.0) + float(v)
        cur_calib, cur_pose = par2.applied(x2), par2.placed(x2)
    final = cur_calib

    std = dict(zip(par2.names, fit2.std))
    for name, val in total.items():
        k_, unit = ((100.0, "cm") if name in ("altura", "kart_long", "kart_lat")
                    else (1.0, "deg"))
        sd = std.get(name, math.nan) * k_
        print(f"  {name}: {val * k_:+.2f} +- {sd:.2f} {unit}")
    print(f"[INFO] {len(reals2)} vistas, {fit2.n_points} pontos, rms {fit2.rms_px:.2f} px; "
          f"custo ultima rodada {before2:.1f} -> {after2:.1f}")

    problems = []
    if hits:
        problems.append("na borda da busca: " + ", ".join(hits)
                        + " (aumente --span/--span-tilt ou confira a cena)")
    if hits2:
        problems.append("ultima rodada na borda: " + ", ".join(hits2))
    loose = [n for n, sd in std.items()
             if n not in _ExtrinsicParams.PLACEMENT
             and sd > (0.015 if n == "altura" else 1.5)]
    # Placement is a nuisance, but a large one is worth knowing: the kart was
    # not where it was said to be -- or the lens is not where the file says.
    a, b, yw = (total.get("kart_long", 0.0), total.get("kart_lat", 0.0),
                total.get("kart_yaw", 0.0))
    print(f"[INFO] posicao estimada do kart: long {a * 100:+.1f} cm, lat {b * 100:+.1f} cm, "
          f"yaw {yw:+.1f} deg em relacao a pose informada")
    if abs(a) > 0.03 or abs(b) > 0.03 or abs(yw) > 3.0:
        print("[WARN] longe da pose informada: confira o posicionamento (ou lens_xyz)")
    if loose:
        problems.append("indeterminados: " + ", ".join(loose))
    if len(reals2) < 4:
        problems.append(f"so {len(reals2)} vistas utilizaveis")
    if after2 > before2 + 1e-6 or after1 > before:
        problems.append("o custo nao caiu")

    # Before: the file's calibration as loaded, at the informed pose; after:
    # the fitted COPY at the FITTED placement (what the fit says the frames
    # show). Neither applies anything twice.
    before_par = _ExtrinsicParams(calib, par2.views, pose, False)
    after_par = _ExtrinsicParams(final, par2.views, cur_pose, False)
    g0, p0 = _as_geoms(before_par.build(before_par.x0))
    g1, p1 = _as_geoms(after_par.build(after_par.x0))
    overlays = [(g0, p0, (0, 0, 255), "antes (verm.)"), (g1, p1, (255, 255, 0), "depois (ciano)")]
    for p in _save_sim(fix, world, g1, after_par.pose, overlays, "extrinsics"):
        print(f"[SIM]  {p}")

    if problems and not args.forcar:
        print("[WARN] NAO salvei: " + " | ".join(problems))
        print("       (--forcar salva mesmo assim)")
        return
    path = resolve_calib_path(args.calib)
    if os.path.exists(path):
        backup = f"{path}.bak-{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.copy(path, backup)
        print(f"[INFO] calibracao anterior em {backup}")
    final.save(path)
    print(f"[INFO] pitch por tilt: "
          + ", ".join(f"{t} {final.tilt_pitch_deg(final.tilt(t)):+.2f}" for t in par2.tilts)
          + f"; altura da lente {final.lens_xyz[2] * 100:.1f} cm; salvo em {path}")
    print("[INFO] rode `check` de novo (e em outra celula) para conferir")


def _view_cost(calib, view_cal, meas, world, pose, d_pitch, d_pan):
    pin = PinholeView.from_calibration(calib, view_cal, extra_pan_deg=d_pan,
                                       extra_pitch_deg=d_pitch)
    geom = build_view_geometry(view_cal.name, pin, view_cal.ignore_bottom_px,
                               CAMERA_FIX_MAX_FLOOR_RANGE_M, CAMERA_FIX_COLUMN_STEP_PX,
                               CAMERA_FIX_ROW_STEP_PX, CAMERA_FIX_BEARING_STEP_DEG)
    m_lo, m_hi = meas.profile.intervals()
    p_lo, p_hi, _ = predict_at(geom, pose, world)
    g = interval_gaps(m_lo, m_hi, p_lo, p_hi)
    p = FitParams()
    return float(np.nansum(_rho(g / p.sigma_px, p.huber_k, p.trunc_k)))


def cmd_extrinsics(args):
    if not args.v1:
        return cmd_extrinsics_v2(args)
    calib, source = load_camera_calibration(args.calib)
    print(f"[INFO] calibracao de partida: {source}")
    view, pose = _placement(args)
    geoms = build_view_geometries(calib)
    # A wider training margin: the extrinsics being calibrated may still be
    # off by degrees, which moves the predicted boundaries by more than the
    # runtime margin allows for.
    fix, world = _run_fix(args, calib, geoms, view, pose, margin_px=25)
    measurements = list(fix._fitter.views) if fix._fitter is not None else []
    if not measurements:
        raise SystemExit(f"[ERROR] nenhuma vista utilizavel: {fix.result.reason}")
    span = np.arange(-args.span, args.span + 1e-9, args.step)
    found = {}
    for meas in measurements:
        vc = calib.view(meas.name)
        best = (math.inf, 0.0, 0.0)
        for dp in span:
            for dn in span:
                c = _view_cost(calib, vc, meas, world, pose, dp, dn)
                if c < best[0]:
                    best = (c, dp, dn)
        base = _view_cost(calib, vc, meas, world, pose, 0.0, 0.0)
        found[meas.name] = best
        print(f"  {meas.name}: pitch {best[1]:+.1f} deg, pan {best[2]:+.1f} deg "
              f"(custo {base:.1f} -> {best[0]:.1f})")
    # Common pitch into the calibration's pitch, the rest per view.
    common = float(np.mean([v[1] for v in found.values()]))
    calib.pitch_deg += common
    for name, (_c, dp, dn) in found.items():
        vc = calib.view(name)
        vc.tilt_bias_deg += dp - common
        vc.pan_bias_deg += dn
    path = resolve_calib_path(args.calib)
    calib.save(path)
    print(f"[INFO] pitch comum {calib.pitch_deg:.2f} deg; salvo em {path}")
    # Same frames, model before (red) and after (cyan) the new extrinsics.
    new_geoms = build_view_geometries(calib)
    at_pose = {name: pose for name in geoms}
    overlays = [(geoms, at_pose, (0, 0, 255), "antes (verm.)"),
                (new_geoms, at_pose, (255, 255, 0), "depois (ciano)")]
    for p in _save_sim(fix, world, new_geoms, pose, overlays, "extrinsics"):
        print(f"[SIM]  {p}")
    print("[INFO] rode `check` de novo (e em outra celula) para conferir")


# ---------------------------------------------------------------------------
def _board_sizes(board: str):
    """Inner-corner sizes to try for --board: as given, and one less each way
    (the usual mistake is passing the square count)."""
    cols, rows = (int(v) for v in board.lower().split("x"))
    sizes = [(cols, rows)]
    if cols > 3 and rows > 3:
        sizes.append((cols - 1, rows - 1))
    return sizes


def _board_rotation(gray, sizes, K, dist, square):
    """(R, size, corners) of the chessboard in view, R taking board-frame
    directions to the camera frame; None when no board is found."""
    for size in sizes:
        corners = _find_board(gray, size)
        if corners is None:
            continue
        objp = np.zeros((size[0] * size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2) * square
        ok, rvec, _tvec = cv2.solvePnP(objp, corners, K, dist)
        if ok:
            return cv2.Rodrigues(rvec)[0], size, corners
    return None


def _board_down(R) -> np.ndarray:
    """The in-plane board axis pointing down, in the board frame: the one the
    camera's own down (+y) is closest to. Needs the board upright with its
    edges level, and the camera's roll well under 45 deg."""
    y_cam = R.T @ np.array([0.0, 1.0, 0.0])
    axis = np.eye(3)[int(np.argmax(np.abs(y_cam[:2])))]
    return axis if y_cam @ axis > 0 else -axis


def _board_angles(R, down, ref_axis=None):
    """(pitch_down, yaw, roll) of the camera in degrees against an upright
    board. Yaw is the turn of the optical axis about the vertical from
    `ref_axis` (+ = left, CCW seen from above); roll + = image right side down."""
    axis = R.T @ np.array([0.0, 0.0, 1.0])
    x_cam = R.T @ np.array([1.0, 0.0, 0.0])
    pitch = math.degrees(math.asin(float(np.clip(axis @ down, -1.0, 1.0))))
    roll = math.degrees(math.asin(float(np.clip(x_cam @ down, -1.0, 1.0))))
    flat = axis - (axis @ down) * down
    flat /= np.linalg.norm(flat)
    yaw = 0.0
    if ref_axis is not None:
        yaw = math.degrees(math.atan2(float(np.cross(ref_axis, flat) @ -down),
                                      float(ref_axis @ flat)))
    return pitch, yaw, roll, flat


def _fresh_frame(stream, settle_s):
    """A frame decoded after the servo had `settle_s` to get there."""
    time.sleep(settle_s)
    t0 = time.monotonic()
    while True:
        frame, _c, t = stream.latest_stamped()
        if t > t0:
            return frame
        if time.monotonic() - t0 > 3.0:
            raise SystemExit("[ERROR] o stream parou de mandar quadros")
        time.sleep(0.02)


def _line_fit(samples):
    """Least-squares angle = a + b * pct over (pct, angle, direction) samples:
    (a, b, rms, hysteresis), hysteresis being the mean residual of the
    increasing pass minus the decreasing one."""
    pct = np.array([s[0] for s in samples], float)
    ang = np.array([s[1] for s in samples], float)
    b, a = np.polyfit(pct, ang, 1)
    res = ang - (a + b * pct)
    up = np.array([s[2] > 0 for s in samples])
    hyst = float(res[up].mean() - res[~up].mean()) if up.any() and (~up).any() else 0.0
    return float(a), float(b), float(np.sqrt(np.mean(res ** 2))), hyst


def cmd_servo(args):
    calib, source = load_camera_calibration(args.calib)
    print(f"[INFO] calibracao de partida: {source}")
    if not any(abs(d) > 1e-9 for d in calib.dist) and source == "padrao":
        print("[WARN] sem intrinsecos medidos -- rode `intrinsics` antes, o angulo "
              "depende da focal")
    K, dist = calib.K(), np.asarray(calib.dist, float)
    sizes = _board_sizes(args.board)
    stream = _open_stream(args)
    _video_url, command_url = _urls(args)
    head = ServoHead(command_url)
    pan_home, tilt_home = calib.pan_home_pct, CAMERA_TILT_HOME_PCT
    state = {}

    def measure(pan, tilt):
        head.set_percents(pan, tilt)
        frame = _fresh_frame(stream, args.settle)
        found = _board_rotation(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), sizes, K,
                                dist, args.square)
        if found is None:
            return None
        R, size, corners = found
        if "down" not in state:
            state["down"] = _board_down(R)
            state["size"] = size
            show = frame.copy()
            cv2.drawChessboardCorners(show, size, corners, True)
            os.makedirs(LOG_DIR, exist_ok=True)
            state["png"] = os.path.join(LOG_DIR, time.strftime("%Y%m%d_%H%M%S_servo_home.png"))
            cv2.imwrite(state["png"], show)
        pitch, yaw, roll, flat = _board_angles(R, state["down"], state.get("ref"))
        state.setdefault("ref", flat)
        return pitch, yaw, roll

    def sweep(name, home, set_pct, pick):
        """Out from home in --step units while the board stays in view, then
        one increasing and one decreasing pass over that range (servo
        hysteresis shows as the offset between them)."""
        lo = hi = home
        while hi + args.step <= min(100, home + args.max_units) \
                and measure(*set_pct(hi + args.step)) is not None:
            hi += args.step
        while lo - args.step >= max(0, home - args.max_units) \
                and measure(*set_pct(lo - args.step)) is not None:
            lo -= args.step
        values = list(range(lo, hi + 1, args.step))
        if len(values) < 3:
            raise SystemExit(f"[ERROR] {name}: o tabuleiro so ficou visivel em "
                             f"{values} -- afaste o tabuleiro do kart")
        samples = []
        measure(*set_pct(max(0, lo - args.step)))        # approach from below
        for direction, seq in ((+1, values), (-1, values[::-1])):
            for v in seq:
                got = measure(*set_pct(v))
                if got is not None:
                    samples.append((v, pick(got), direction))
        print(f"  {name}: {len(samples)} medidas de {lo} a {hi}")
        return samples, lo, hi

    try:
        if measure(pan_home, tilt_home) is None:
            raise SystemExit(
                f"[ERROR] tabuleiro {args.board} nao encontrado com pan {pan_home} e "
                f"tilt {tilt_home}. Prenda o tabuleiro EM PE (bordas niveladas) a "
                "~50-70 cm na frente do kart, na altura da camera, e confira com o "
                "/video no navegador")
        print(f"[INFO] tabuleiro {state['size'][0]}x{state['size'][1]} -> {state['png']}")
        pan_s, pan_lo, pan_hi = sweep("pan", pan_home, lambda v: (v, tilt_home),
                                      lambda g: g[1])
        tilt_s, tilt_lo, tilt_hi = sweep("tilt", tilt_home, lambda v: (pan_home, v),
                                         lambda g: g[0])
        rolls = [measure(pan_home, tilt_home)]
    finally:
        head.set_percents(pan_home, calib.tilt_pct)
        time.sleep(0.5)                    # the post runs on a daemon thread
        stream.stop()

    pa, pb, prms, physt = _line_fit(pan_s)
    ta, tb, trms, thyst = _line_fit(tilt_s)
    print(f"[INFO] pan : {abs(pb):.3f} deg/unidade ({'mais' if pb > 0 else 'menos'} = "
          f"esquerda), rms {prms:.2f} deg, histerese {physt:+.2f} deg")
    print(f"[INFO] tilt: {abs(tb):.3f} deg/unidade ({'mais' if tb > 0 else 'menos'} = "
          f"baixo), rms {trms:.2f} deg, histerese {thyst:+.2f} deg")
    if abs(pb) < 0.3 or abs(tb) < 0.3:
        raise SystemExit("[ERROR] o servo quase nao mexeu a camera -- bateria? nada salvo")
    level = -ta / tb
    pitch_fix = ta + tb * calib.tilt_pct
    print(f"[INFO] tilt nivelado em {level:.1f}; tilt {calib.tilt_pct} = "
          f"{pitch_fix:.1f} deg para baixo (antes: {calib.pitch_deg:.1f})")
    if not tilt_lo - args.step <= calib.tilt_pct <= tilt_hi + args.step:
        print(f"[WARN] tilt {calib.tilt_pct} fora da faixa medida ({tilt_lo}-{tilt_hi}): "
              "pitch extrapolado")
    if rolls[0] is not None:
        print(f"[INFO] roll da camera {rolls[0][2]:+.1f} deg (+ = lado direito para baixo)")

    calib.deg_per_pct = abs(pb)
    calib.pan_left_is_higher = pb > 0
    calib.tilt_deg_per_pct = abs(tb)
    calib.tilt_level_pct = float(level)
    calib.pitch_deg = float(pitch_fix)
    reach = min(calib.pan_home_pct, 100 - calib.pan_home_pct) * calib.deg_per_pct
    for v in calib.views:
        # Biases fitted over the old mapping mean nothing over the new one.
        v.pan_bias_deg = v.tilt_bias_deg = 0.0
        if abs(v.pan_deg) > reach:
            new = math.copysign(math.floor(reach), v.pan_deg)
            print(f"[WARN] vista {v.name}: pan {v.pan_deg:+.0f} deg alem do curso do "
                  f"servo ({reach:.0f} deg); usando {new:+.0f}")
            v.pan_deg = new
    for v in calib.views:
        print(f"       {v.name}: pan {v.pan_deg:+.0f} deg = unidade {calib.pan_pct(v)}")
    path = resolve_calib_path(args.calib)
    calib.save(path)
    print(f"[INFO] salvo em {path}; vieses das vistas zerados -- rode `extrinsics` "
          "de novo")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--ip", default=KART_HOST,
                   help="IP ou nome (ex: pikart.local) do kart (default: %(default)s)")
    p.add_argument("-p", "--port", type=int, default=KART_PORT)
    p.add_argument("--calib", default=None, help="arquivo de calibracao")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("intrinsics")
    s.add_argument("--board", default="9x6",
                   help="cantos INTERNOS, CxL (tabuleiro de 8x8 casas = 7x7)")
    s.add_argument("--square", type=float, default=0.025, help="lado do quadrado, m")
    s.set_defaults(fn=cmd_intrinsics)

    s = sub.add_parser("servo")
    s.add_argument("--board", default="9x6",
                   help="cantos INTERNOS, CxL (tabuleiro de 8x8 casas = 7x7)")
    s.add_argument("--square", type=float, default=0.025, help="lado do quadrado, m")
    s.add_argument("--step", type=int, default=4, help="passo da varredura, unidades")
    s.add_argument("--max-units", type=int, default=40,
                   help="ate quantas unidades do centro varrer")
    s.add_argument("--settle", type=float, default=0.8,
                   help="espera apos cada movimento do servo, s")
    s.set_defaults(fn=cmd_servo)

    for name, fn in (("check", cmd_check), ("extrinsics", cmd_extrinsics)):
        s = sub.add_parser(name)
        s.add_argument("--maze", default="tcc.maze")
        s.add_argument("--cell", nargs=2, type=int, required=True, metavar=("LIN", "COL"))
        s.add_argument("--heading", type=float, required=True,
                       help="0 direita, 90 cima, 180 esquerda, 270 baixo")
        s.add_argument("--v1", action="store_true",
                       help="versao antiga (contatos com o chao, um tilt so)")
        if name == "extrinsics":
            s.add_argument("--span", type=float, default=15.0,
                           help="busca +-graus (v2: pans e vieses de tilt)")
            s.add_argument("--span-tilt", type=float, default=35.0,
                           help="v2: busca +-graus do pitch de cada tilt")
            s.add_argument("--forcar", action="store_true",
                           help="v2: salva mesmo com parametros na borda/indeterminados")
            s.add_argument("--step", type=float, default=1.0,
                           help="passo da grade, graus (v1: 0.5)")
            s.add_argument("--altura", action="store_true",
                           help="v2: ajusta tambem a altura da lente")
            s.add_argument("--passes", type=int, default=8,
                           help="v2: passadas do ajuste fino")
        s.set_defaults(fn=fn)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
