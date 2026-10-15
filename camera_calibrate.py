"""
Bench tool for the camera pose fix (maze_solver/runner/camera_fix.py). Talks to
the kart directly (MJPEG + CMD_SERVO over HTTP) -- no CoppeliaSim, no runner.

    python camera_calibrate.py intrinsics  [-i HOST] [--board 9x6 --square 0.025]
    python camera_calibrate.py check       [-i HOST] --maze tcc.maze --cell 2 0 --heading 90
    python camera_calibrate.py extrinsics  [-i HOST] --maze tcc.maze --cell 2 0 --heading 90

intrinsics
    Live view of the stream; SPACE grabs a frame when the chessboard is found,
    C calibrates (10+ grabs, spread over the whole image and a few tilts), Q
    quits. Writes fx, fy, cx, cy and the distortion to the calibration file.
    Without this the nominal Pi Camera rev 1.3 optics are used (f ~ 402 px).

check
    Place the kart by hand on the CENTRE of the given cell, facing the given
    grid heading (0 right/east, 90 up/north, 180 left/west, 270 down/south),
    exactly as for 'ok' in the runner. Runs one full camera fix there and prints
    what it measured -- ideally close to zero. Debug PNGs (frame, wall mask,
    measured vs predicted contacts) go to maze_solver/logs/camera/. This is the
    first thing to look at: a mirrored or vertically shifted model in those
    images means a sign or the tilt is wrong in runner/constants.py.

extrinsics
    Same placement, then searches each view's pitch and pan offsets that make
    the predicted contacts land on the measured ones, and writes them to the
    calibration file (common pitch + per-view tilt/pan biases). Place the kart
    as carefully as possible -- any placement error goes into the biases.
    Repeat at 2-3 different cells and keep the run whose `check` reads closest
    to zero everywhere.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import socket
import subprocess
import sys
import time

import cv2
import numpy as np

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

from runner.camera_fit import FitParams, _rho                    # noqa: E402
from runner.camera_fix import CameraFix                          # noqa: E402
from runner.camera_setup import (build_view_geometries,          # noqa: E402
                                 load_camera_calibration, resolve_calib_path)
from runner.constants import (CAMERA_FIX_BEARING_STEP_DEG,       # noqa: E402
                              CAMERA_FIX_COLUMN_STEP_PX,
                              CAMERA_FIX_MAX_FLOOR_RANGE_M,
                              CAMERA_FIX_ROW_STEP_PX, KART_HOST, KART_PORT)
from runner.contact_model import interval_gaps, predict_at       # noqa: E402
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
    try:
        while True:
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
            cv2.imshow("intrinsics", cv2.resize(show, None, fx=2, fy=2))
            key = cv2.waitKey(30) & 0xFF
            if key == ord(" ") and found:
                objp = np.zeros((size[0] * size[1], 3), np.float32)
                objp[:, :2] = (np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)
                               * args.square)
                obj_pts.append(objp)
                img_pts.append(corners)
                print(f"  captura {len(img_pts)}")
            elif key == ord("c") and len(img_pts) >= 10:
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
            elif key == ord("q"):
                break
    finally:
        stream.stop()
        cv2.destroyAllWindows()


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


def cmd_check(args):
    calib, source = load_camera_calibration(args.calib)
    print(f"[INFO] calibracao: {source}")
    view, pose = _placement(args)
    geoms = build_view_geometries(calib)
    fix, _world = _run_fix(args, calib, geoms, view, pose)
    res = fix.result
    print(f"[INFO] {res.summary()}  ({res.duration_s:.1f} s)")
    if res.fit is not None:
        a, b, psi = res.fit.estimate
        print(f"       estimativa bruta: long {a * 100:+.1f} cm, lat {b * 100:+.1f} cm, "
              f"yaw {math.degrees(psi):+.2f} deg, pans "
              + ", ".join(f"{k} {math.degrees(v):+.1f}" for k, v in res.fit.etas.items()))
    for p in res.debug_paths:
        print(f"       {p}")


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
    print("[INFO] rode `check` de novo (e em outra celula) para conferir")


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

    for name, fn in (("check", cmd_check), ("extrinsics", cmd_extrinsics)):
        s = sub.add_parser(name)
        s.add_argument("--maze", default="tcc.maze")
        s.add_argument("--cell", nargs=2, type=int, required=True, metavar=("LIN", "COL"))
        s.add_argument("--heading", type=float, required=True,
                       help="0 direita, 90 cima, 180 esquerda, 270 baixo")
        if name == "extrinsics":
            s.add_argument("--span", type=float, default=6.0, help="busca +-graus")
            s.add_argument("--step", type=float, default=0.5, help="passo, graus")
        s.set_defaults(fn=fn)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
