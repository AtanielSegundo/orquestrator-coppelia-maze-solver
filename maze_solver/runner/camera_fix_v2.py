"""
Camera pose fix v2 at one cell: measure, step, measure again -- polled once per
frame like CameraFix, Recenter and the closest obstacle reference.

One iteration
-------------
    settle_kart -> home -> [aim -> wait_servo -> settle_servo -> capture] x views
                -> processing -> accepted | step -> [turn -> settle] -> next iteration

The views are the pan x tilt head positions of runner/camera_setup.v2_views,
always in the same order -- centre up, centre down, right up, right down, left
up, left down -- starting from the home pan at the held tilt, so each servo
reaches each position from the same side every time, as `extrinsics` did.

The worker thread turns each view's frames into a wallness field as they
arrive (undistortion, colour model trained on the simulated labels: box vs
everything else), and once all are in, fits (yaw, longitudinal, lateral, one
pan nuisance per side pan) to the box edges (runner/edge_match.py).

The step
--------
* Yaw is corrected by TURNING. The belief is first re-anchored to the measured
  heading (`on_step`), then the kart turns back onto the grid axis -- odometry
  tracks the turn, so without the re-anchor the turn would only move the
  belief error into the kart's real heading.
* The longitudinal offset only moves the belief (`on_step`, same call).
* Lateral is estimated (it would otherwise leak into yaw) but never applied:
  that is the closest obstacle reference's axis.
An axis whose standard deviation is too large is not stepped. Steps are
gain x estimate, clamped per iteration. Then the next iteration measures the
result, until both readings are under CAMERA_V2_ACCEPT_* (accepted) or
CAMERA_V2_MAX_ITERATIONS have run. A yaw reading that GROWS after a step stops
the loop ("divergiu"): a wrong sign or a wrong match would only be amplified.

`act=False` measures once and applies nothing (bench `check`, the runner's
`camfix`); `measure_only` stops after the wallness fields (bench
`extrinsics`, which fits the calibration to them instead of the pose).
"""
from __future__ import annotations

import math
import os
import queue
import threading
import time
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import cv2
import numpy as np

from camera_link.floor_segmenter import FLOOR_LABEL, WALL_LABEL, FloorSegmenter
from camera_link.pinhole_view import PinholeView
from runner.camera_setup import v2_views
from runner.closest_obstacle import snap_to_axis
from runner.constants import (
    CAMERA_FIX_FRAME_LATENCY_S, CAMERA_FIX_FRAMES_PER_VIEW,
    CAMERA_FIX_KART_MOVED_DEG, CAMERA_FIX_KART_MOVED_M, CAMERA_FIX_KART_SETTLE_S,
    CAMERA_FIX_SERVO_ACK_TIMEOUT_S, CAMERA_FIX_SERVO_SETTLE_S,
    CAMERA_FIX_VIEW_TIMEOUT_S, CAMERA_SEG_L_WEIGHT, CAMERA_SEG_LABEL_MARGIN_PX,
    CAMERA_SEG_MIN_ACCURACY, CAMERA_SEG_MIN_PIXELS, CAMERA_SEG_REFINE_ITERS,
    CAMERA_V2_ACCEPT_LONG_M, CAMERA_V2_ACCEPT_YAW_DEG, CAMERA_V2_DIVERGE_DEG,
    CAMERA_V2_LAT_PRIOR_M, CAMERA_V2_LONG_GAIN, CAMERA_V2_LONG_PRIOR_M,
    CAMERA_V2_MAX_ITERATIONS, CAMERA_V2_MAX_LONG_STEP_M, CAMERA_V2_MAX_STD_LONG_M,
    CAMERA_V2_MAX_STD_YAW_DEG, CAMERA_V2_MAX_YAW_STEP_DEG, CAMERA_V2_MIN_TURN_DEG,
    CAMERA_V2_COARSE_YAW_DEG, CAMERA_V2_COARSE_YAW_STEP_DEG,
    CAMERA_V2_PAN_NUISANCE_SIGMA_DEG, CAMERA_V2_PITCH_NUISANCE_SIGMA_DEG,
    CAMERA_V2_SEG_COMPONENTS, CAMERA_V2_SPLIT_PAN,
    CAMERA_V2_YAW_GAIN, CAMERA_V2_YAW_PRIOR_DEG, CAMERA_V2_YAW_SIGN,
    SWEEP_SETTLE_S, SWEEP_START_GRACE_S,
)
from runner.edge_match import (EdgeFit, GroupStat, RealView, ViewSpec, describe_groups,
                               sim_edges, solve_edges, wall_field)
from runner.reference_renderer import WALL, render_labels, upsample_labels
from runner.wall_world import WallWorld

from shadow.compose import compose
from shadow.inverse import inverse

ACCEPTED = "aceito"
CORRECTED = "corrigido"
DIVERGED = "divergiu"
REFUSED = "recusado"
MEASURED = "medido"

_LABEL_STRIDE = 2
_NARROW_MARGIN_PX = 8
_THIN_PX = 11
_STRIP_PX = 25


def _wrap(t: float) -> float:
    return (t + math.pi) % (2 * math.pi) - math.pi


def _clamp(v: float, lim: float) -> float:
    return max(-lim, min(lim, v))


def undistort(calib, frame: np.ndarray):
    """Frame -> (image in the ideal pinhole, invalid-pixel mask or None). The
    maps are cached on the calibration, shared with CameraFix."""
    W, H = calib.width, calib.height
    if frame.shape[1] != W or frame.shape[0] != H:
        if abs(frame.shape[1] / frame.shape[0] - W / H) > 0.01:
            raise ValueError(f"frame {frame.shape[1]}x{frame.shape[0]} "
                             f"nao bate com a calibracao {W}x{H}")
        frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
    if not calib.needs_remap():
        return frame, None
    maps = getattr(calib, "_maps", None)
    if maps is None:
        m1, m2 = cv2.initUndistortRectifyMap(
            calib.K(), np.asarray(calib.dist, dtype=float), None,
            calib.K_ideal(), (W, H), cv2.CV_32FC1)
        ones = np.full((H, W), 255, np.uint8)
        invalid = cv2.remap(ones, m1, m2, cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0) == 0
        maps = (m1, m2, invalid)
        calib._maps = maps
    m1, m2, invalid = maps
    return cv2.remap(frame, m1, m2, cv2.INTER_LINEAR), invalid


def view_spec(view) -> ViewSpec:
    return ViewSpec(view.name, view.pan_name or view.name, view.tilt, view.ignore_bottom_px)


def make_segmenter() -> FloorSegmenter:
    return FloorSegmenter(components=CAMERA_V2_SEG_COMPONENTS,
                          l_weight=CAMERA_SEG_L_WEIGHT,
                          margin_px=CAMERA_SEG_LABEL_MARGIN_PX,
                          min_pixels=CAMERA_SEG_MIN_PIXELS,
                          min_accuracy=CAMERA_SEG_MIN_ACCURACY,
                          refine_iters=CAMERA_SEG_REFINE_ITERS)


def measure_view(calib, view, frames: List[np.ndarray], pose, world: WallWorld,
                 segmenter: FloorSegmenter):
    """(RealView or None, why, first undistorted frame) of one view's frames:
    the colour model is seeded by the simulated labels at `pose` -- box versus
    EVERYTHING else, since above the boxes there is the room, not floor."""
    return measure_prepared(calib, view, [undistort(calib, f) for f in frames],
                            pose, world, segmenter)


def measure_prepared(calib, view, prepared, pose, world: WallWorld,
                     segmenter: FloorSegmenter, cluster: bool = False):
    """measure_view on frames already undistorted: [(image, invalid or None)].
    `extrinsics` re-segments with these once it has a better calibration.

    `cluster`: seed the colour model from colour clusters the prediction only
    votes on (FloorSegmenter.cluster_labels) instead of the eroded prediction
    -- for a calibration that may be degrees off."""
    img0, invalid = prepared[0]
    H, W = img0.shape[:2]
    pin = PinholeView.from_calibration(calib, view)
    labels = render_labels(pin, pose, world, stride=_LABEL_STRIDE,
                           extra_boxes=world.obstacles)
    labels = upsample_labels(labels, _LABEL_STRIDE, (H, W))
    seed = np.where(labels == WALL, WALL_LABEL, FLOOR_LABEL).astype(np.int8)
    band = np.zeros((H, W), dtype=bool)
    if view.ignore_bottom_px:
        band[H - 1 - int(view.ignore_bottom_px) + 1:, :] = True
    exclude = band | (invalid if invalid is not None else False)
    clustered = segmenter.cluster_labels(img0, seed, exclude) if cluster else None
    if clustered is not None:
        seed = clustered
        margin = segmenter.margin_px
        segmenter.margin_px = 2           # colour boundaries: no pose error to cover
        try:
            model, why = segmenter.train(img0, seed, exclude)
        finally:
            segmenter.margin_px = margin
        if model is None:
            return None, why, img0
        return _finish_measure(calib, view, prepared, segmenter, model, band, invalid, img0)
    # Looking up at a near box, the room above it can be a thin strip that the
    # training margin erodes away entirely: retry with a narrow margin.
    margin = segmenter.margin_px
    model, why = segmenter.train(img0, seed, exclude)
    if model is None and margin > _NARROW_MARGIN_PX:
        segmenter.margin_px = _NARROW_MARGIN_PX
        try:
            model, why = segmenter.train(img0, seed, exclude)
        finally:
            segmenter.margin_px = margin
    if model is None:
        return None, why, img0
    return _finish_measure(calib, view, prepared, segmenter, model, band, invalid, img0)


def _finish_measure(calib, view, prepared, segmenter, model, band, invalid, img0):
    llr = np.mean([segmenter.llr(img, model) for img, _inv in prepared], axis=0)
    mask = FloorSegmenter.wall_mask(llr)
    # Boxes are big blobs; a floor seam or a skirting strip the colour model
    # calls "box" is a thin line -- and its edges look exactly like a floor
    # contact (measured on the real kart: the fit locked onto the seam between
    # two floor boards, 120 px below the real wall base). Opening removes
    # anything thinner than the kernel and leaves the boxes' edges in place.
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_THIN_PX, _THIN_PX)))
    # ...and a seam close to the lens is thicker than that (12 px at 30 deg
    # down), but still a HORIZONTAL strip: a vertical opening takes it out
    # while every box, taller than this, stays.
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((_STRIP_PX, 1), np.uint8))
    mask = m.astype(bool)
    valid = ~band
    if invalid is not None:
        valid &= ~invalid
        mask &= ~invalid
    return RealView(view_spec(view), wall_field(mask), valid, model.accuracy), "", img0


class PoseParams:
    """x = (yaw rad, long m, lat m, common pitch nuisance rad, pan nuisance rad
    per side pan) in the frame of the heading frozen at capture -> (pinhole,
    pose) per view.

    The common pitch nuisance is a pitch error shared by every view: a
    calibration drift, the kart not sitting level. It moves the box tops and
    the floor contacts the SAME way, where a longitudinal offset moves them
    apart -- so it can be told from one, and is not left to bias it (1 deg of
    unmodelled pitch read as ~1 cm of longitudinal on the synthetic corridor).
    """

    def __init__(self, calib, views, capture_pose,
                 nuisance_sigma_deg: float = CAMERA_V2_PAN_NUISANCE_SIGMA_DEG,
                 pitch_sigma_deg: float = CAMERA_V2_PITCH_NUISANCE_SIGMA_DEG,
                 split_pan: str = CAMERA_V2_SPLIT_PAN):
        self.calib = calib
        self.views = list(views)
        self.capture = tuple(capture_pose)
        self.pins = {v.name: PinholeView.from_calibration(calib, v) for v in self.views}
        pans = []
        if nuisance_sigma_deg > 0:
            for v in self.views:
                p = v.pan_name or v.name
                if p != split_pan and v.nuisance and p not in pans:
                    pans.append(p)
        self.nuisance_pans = pans
        self.names = ["yaw", "long", "lat", "pitch"] + [f"pan_{p}" for p in pans]
        self.x0 = np.zeros(len(self.names))
        self.sigma = np.array([math.radians(CAMERA_V2_YAW_PRIOR_DEG), CAMERA_V2_LONG_PRIOR_M,
                               CAMERA_V2_LAT_PRIOR_M, math.radians(max(pitch_sigma_deg, 1e-6))]
                              + [math.radians(max(nuisance_sigma_deg, 1e-6))] * len(pans))
        self.steps = np.array([math.radians(0.2), 0.005, 0.005, math.radians(0.2)]
                              + [math.radians(0.2)] * len(pans))
        self.max_step = np.array([math.radians(2.0), 0.02, 0.02, math.radians(1.0)]
                                 + [math.radians(1.0)] * len(pans))
        self.coarse = (0, np.radians(np.arange(-CAMERA_V2_COARSE_YAW_DEG,
                                               CAMERA_V2_COARSE_YAW_DEG + 1e-9,
                                               CAMERA_V2_COARSE_YAW_STEP_DEG)))

    def pose(self, x) -> Tuple[float, float, float]:
        x0, y0, th0 = self.capture
        c, s = math.cos(th0), math.sin(th0)
        psi, a, b = x[0], x[1], x[2]
        return (x0 + c * a - s * b, y0 + s * a + c * b, th0 + psi)

    def build(self, x):
        pose = self.pose(x)
        pitch = math.degrees(x[3])
        out = {}
        for v in self.views:
            p = v.pan_name or v.name
            eta = (math.degrees(x[4 + self.nuisance_pans.index(p)])
                   if p in self.nuisance_pans else 0.0)
            if eta or pitch:
                pin = PinholeView.from_calibration(self.calib, v, extra_pan_deg=eta,
                                                   extra_pitch_deg=pitch)
            else:
                pin = self.pins[v.name]
            out[v.name] = (pin, pose)
        return out

    def etas(self, x) -> Dict[str, float]:
        return {p: float(x[4 + i]) for i, p in enumerate(self.nuisance_pans)}


class V2Iteration(NamedTuple):
    yaw: float                       # rad, true - believed heading
    long: float                      # m, along the capture heading
    lat: float                       # m, + = left (not applied)
    std: Tuple[float, float, float]
    etas: Dict[str, float]           # rad
    n_points: int
    groups: Dict[str, GroupStat]
    dropped: Dict[str, str]
    rms_px: float
    step_long: float
    step_yaw: float
    turned_deg: float
    note: str

    def describe(self) -> str:
        return (f"yaw {math.degrees(self.yaw):+.2f}+-{math.degrees(self.std[0]):.2f} deg, "
                f"long {self.long * 100:+.1f}+-{self.std[1] * 100:.1f} cm, "
                f"lat {self.lat * 100:+.1f} cm ({self.n_points} pts, rms {self.rms_px:.1f} px)"
                + (f" -> passo yaw {math.degrees(self.step_yaw):+.2f} deg, "
                   f"long {self.step_long * 100:+.1f} cm, girou {self.turned_deg:+.1f} deg"
                   if (self.step_yaw or self.step_long or self.turned_deg) else "")
                + (f" [{self.note}]" if self.note else ""))


class CameraV2Result(NamedTuple):
    outcome: str
    reason: str
    iterations: Tuple[V2Iteration, ...]
    applied_long: float
    applied_yaw: float
    turned_deg: float
    duration_s: float
    debug_paths: Tuple[str, ...]

    def summary(self) -> str:
        head = (f"{self.outcome}" + (f" ({self.reason})" if self.reason else "")
                + f", {len(self.iterations)} medida(s)")
        if self.applied_long or self.applied_yaw or self.turned_deg:
            head += (f": ancora yaw {math.degrees(self.applied_yaw):+.2f} deg / long "
                     f"{self.applied_long * 100:+.1f} cm, girou {self.turned_deg:+.1f} deg")
        if self.iterations:
            head += " | ultima: " + self.iterations[-1].describe()
        return head


class _Bundle(NamedTuple):
    view: object
    frames: List[np.ndarray]


def _moved(a, b) -> Tuple[float, float]:
    d = math.hypot(a[0] - b[0], a[1] - b[1])
    th = math.degrees(abs((a[2] - b[2] + math.pi) % (2 * math.pi) - math.pi))
    return d, th


OnStep = Callable[[tuple, tuple, float, float], None]


class CameraFixV2:
    def __init__(self, calib, head, stream, link, world: WallWorld, odom_ref,
                 purpose: str = "medicao", act: bool = True,
                 on_step: Optional[OnStep] = None, turn_speed: float = 0.35,
                 measure_only: bool = False, views=None,
                 max_iterations: int = CAMERA_V2_MAX_ITERATIONS,
                 dump_dir: Optional[str] = None, clock=time.monotonic):
        self.calib = calib
        self.head = head
        self.stream = stream
        self.link = link
        self.world = world
        self.odom_ref = odom_ref
        self.purpose = purpose
        self.act = act and not measure_only
        self.on_step = on_step
        self.turn_speed = turn_speed
        self.measure_only = measure_only
        self.max_iterations = max_iterations if self.act else 1
        self.dump_dir = dump_dir
        self._clock = clock
        self._t0 = clock()
        self.views = list(views) if views is not None else v2_views(calib)
        self.status = "assentando kart"
        self.result: Optional[CameraV2Result] = None
        self.segmenter = make_segmenter()

        self.iterations: List[V2Iteration] = []
        self.applied_long = 0.0
        self.applied_yaw = 0.0
        self.turned_deg = 0.0
        self._debug_paths: List[str] = []
        # Last iteration's data, for the bench tools.
        self.reals: Dict[str, RealView] = {}
        self.frames: Dict[str, np.ndarray] = {}
        # Undistorted frames per view, [(image, invalid)], for re-segmenting.
        self.prepared: Dict[str, list] = {}
        self.dropped: Dict[str, str] = {}
        self.fit: Optional[EdgeFit] = None
        self.params: Optional[PoseParams] = None

        self.capture_pose = None
        self.capture_rel = None
        self._odom_capture = None
        self._phase = "settle_kart"
        self._settle_until = self._t0 + CAMERA_FIX_KART_SETTLE_S
        self._index = 0
        self._frames: List[np.ndarray] = []
        self._last_count = -1
        self._deadline = 0.0
        self._ready_at = 0.0
        self._commanded = (0, 0)
        self._worker: Optional[threading.Thread] = None
        self._queue: "queue.Queue" = queue.Queue()
        self._done = threading.Event()
        self._cancel = False
        self._worker_error: Optional[str] = None
        # turning
        self._issued = False
        self._seen_active = False
        self._active_by = 0.0
        self._last_after_turn = False
        if not self.views:
            self._finish(REFUSED, "calibracao sem vistas v2 (tilts/pans)")

    # -- head ----------------------------------------------------------------
    def _command(self, pan_pct: int, tilt_pct: int) -> None:
        self._commanded = (int(pan_pct), int(tilt_pct))
        self.head.set_percents(*self._commanded)
        self._deadline = self._clock() + CAMERA_FIX_SERVO_ACK_TIMEOUT_S

    def _acked(self) -> bool:
        if self._clock() >= self._deadline:
            return True
        try:
            tel = self.link.servo_telemetry() if self.link is not None else {}
        except Exception:                                       # noqa: BLE001
            tel = {}
        if not tel:
            return True
        pan_ch = str(getattr(self.head, "PAN_CHANNEL", 5))
        tilt_ch = str(getattr(self.head, "TILT_CHANNEL", 1))
        try:
            return (int(round(float(tel.get(pan_ch)))) == self._commanded[0]
                    and int(round(float(tel.get(tilt_ch)))) == self._commanded[1])
        except (TypeError, ValueError):
            return True

    def _home(self) -> None:
        self.head.set_percents(self.calib.pan_home_pct, self.calib.tilt_pct)

    # -- poll ------------------------------------------------------------------
    def poll(self, drive_tel: dict, odom_pose, wx: float, wy: float, wth: float) -> str:
        if self._phase == "done":
            return "done"
        if self._phase == "settle_kart":
            return self._do_settle_kart(drive_tel, odom_pose, wx, wy, wth)
        if self._phase == "turning":
            return self._do_turning(drive_tel)
        if (self._phase in ("home", "aim", "wait_servo", "settle_servo", "capture")
                and self._odom_capture is not None and odom_pose is not None):
            d, th = _moved(odom_pose, self._odom_capture)
            if d > CAMERA_FIX_KART_MOVED_M or th > CAMERA_FIX_KART_MOVED_DEG:
                return self._abort(f"kart se mexeu durante a captura "
                                   f"({d * 100:.1f} cm, {th:.1f} deg)")
        handler = {"home": self._do_home, "aim": self._do_aim,
                   "wait_servo": self._do_wait_servo, "settle_servo": self._do_settle_servo,
                   "capture": self._do_capture}.get(self._phase)
        if handler is not None:
            return handler()
        return self._do_processing()

    def _do_settle_kart(self, drive_tel, odom_pose, wx, wy, wth) -> str:
        if drive_tel.get("goal_active"):
            self._settle_until = self._clock() + CAMERA_FIX_KART_SETTLE_S
            return "pending"
        if self._clock() < self._settle_until or odom_pose is None:
            return "pending"
        self.capture_pose = (float(wx), float(wy), float(wth))
        self._odom_capture = tuple(odom_pose)
        self.capture_rel = compose(inverse(self.odom_ref), self._odom_capture)
        self._index = 0
        self._done = threading.Event()
        self._queue = queue.Queue()
        self._worker_error = None
        self.reals, self.frames, self.dropped, self.fit = {}, {}, {}, None
        self.prepared = {}
        self._worker = threading.Thread(target=self._work, daemon=True, name="CameraFixV2")
        self._worker.start()
        self._command(self.calib.pan_home_pct, self.calib.tilt_pct)
        self._phase = "home"
        self.status = f"medida {len(self.iterations) + 1}: camera em casa"
        return "pending"

    def _do_home(self) -> str:
        if not self._acked():
            return "pending"
        self._phase = "aim"
        return "pending"

    def _do_aim(self) -> str:
        view = self.views[self._index]
        self._command(self.calib.pan_pct(view), self.calib.view_tilt(view)[0])
        self._phase = "wait_servo"
        self.status = f"medida {len(self.iterations) + 1}: mirando {view.name}"
        return "pending"

    def _do_wait_servo(self) -> str:
        if not self._acked():
            return "pending"
        self._ready_at = self._clock() + CAMERA_FIX_SERVO_SETTLE_S
        self._phase = "settle_servo"
        return "pending"

    def _do_settle_servo(self) -> str:
        if self._clock() < self._ready_at:
            return "pending"
        self._ready_at = self._clock() + CAMERA_FIX_FRAME_LATENCY_S
        self._deadline = self._clock() + CAMERA_FIX_VIEW_TIMEOUT_S
        _f, count, _t = self.stream.latest_stamped()
        self._last_count = count
        self._frames = []
        self._phase = "capture"
        return "pending"

    def _do_capture(self) -> str:
        view = self.views[self._index]
        frame, count, t_recv = self.stream.latest_stamped()
        if frame is not None and count != self._last_count and t_recv >= self._ready_at:
            self._frames.append(frame.copy())
            self._last_count = count
        if len(self._frames) >= CAMERA_FIX_FRAMES_PER_VIEW or self._clock() >= self._deadline:
            if self._frames:
                self._queue.put(_Bundle(view, self._frames))
            else:
                self.dropped[view.name] = "sem frames novos"
            self._index += 1
            self._frames = []
            if self._index < len(self.views):
                self._phase = "aim"
            else:
                self._home()
                self._queue.put(None)
                self._phase = "processing"
                self.status = f"medida {len(self.iterations) + 1}: ajustando"
        return "pending"

    # -- worker ----------------------------------------------------------------
    def _work(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is None or self._cancel:
                    break
                prepared = [undistort(self.calib, f) for f in item.frames]
                self.prepared[item.view.name] = prepared
                real, why, img0 = measure_prepared(self.calib, item.view, prepared,
                                                   self.capture_pose, self.world,
                                                   self.segmenter)
                self.frames[item.view.name] = img0
                if real is None:
                    self.dropped[item.view.name] = why
                else:
                    self.reals[item.view.name] = real
            if self._cancel or self.measure_only:
                return
            params = PoseParams(self.calib, [v for v in self.views if v.name in self.reals],
                                self.capture_pose)
            self.params = params
            if self.reals:
                self.fit = solve_edges(self.reals, self.world, params.build, params.x0,
                                       params.sigma, params.steps,
                                       obstacles=self.world.obstacles,
                                       coarse=params.coarse, max_step=params.max_step)
        except Exception as exc:                                # noqa: BLE001
            self._worker_error = f"falha no ajuste: {type(exc).__name__}: {exc}"
        finally:
            self._done.set()

    # -- deciding --------------------------------------------------------------
    def _do_processing(self) -> str:
        if not self._done.is_set():
            return "pending"
        if self._worker_error is not None:
            return self._finish(REFUSED, self._worker_error)
        if self.measure_only:
            return self._finish(MEASURED, "")
        fit = self.fit
        if fit is None or not fit.ok:
            why = (fit.reason if fit is not None else "nenhuma vista utilizavel")
            if self.dropped:
                why += " | " + ", ".join(f"{k}: {v}" for k, v in self.dropped.items())
            if not self.iterations:
                return self._finish(REFUSED, why)
            return self._finish(CORRECTED, f"medida {len(self.iterations) + 1} falhou: {why}")
        if self.dump_dir:
            self._dump(fit)

        psi, a, b = (float(v) for v in fit.x[:3])
        std = tuple(float(v) for v in fit.std[:3])
        etas = self.params.etas(fit.x)
        etas["pitch"] = float(fit.x[3])
        yaw_ok = std[0] <= math.radians(CAMERA_V2_MAX_STD_YAW_DEG)
        long_ok = std[1] <= CAMERA_V2_MAX_STD_LONG_M
        notes = []
        if not yaw_ok:
            notes.append("yaw nao observavel")
        if not long_ok:
            notes.append("long nao observavel")
        psi_use = psi if yaw_ok else 0.0
        a_use = a if long_ok else 0.0

        def record(step_long=0.0, step_yaw=0.0, turned=0.0, note=""):
            it = V2Iteration(psi, a, b, std, etas, fit.n_points, fit.groups, fit.dropped,
                             fit.rms_px, step_long, step_yaw, turned,
                             "; ".join([n for n in notes + [note] if n]))
            self.iterations.append(it)
            return it

        prev = self.iterations[-1] if self.iterations else None
        if (prev is not None and (prev.step_yaw or prev.turned_deg)
                and abs(math.degrees(psi_use)) > abs(math.degrees(prev.yaw))
                + CAMERA_V2_DIVERGE_DEG):
            record(note="yaw cresceu apos o passo")
            return self._finish(DIVERGED, "yaw cresceu apos o passo -- sinal "
                                          "(CAMERA_V2_YAW_SIGN) ou casamento errado?")
        if not self.act:
            record()
            return self._finish(MEASURED, "")
        if (abs(psi_use) < math.radians(CAMERA_V2_ACCEPT_YAW_DEG)
                and abs(a_use) < CAMERA_V2_ACCEPT_LONG_M):
            record()
            return self._finish(ACCEPTED, "")

        step_yaw = CAMERA_V2_YAW_SIGN * _clamp(CAMERA_V2_YAW_GAIN * psi_use,
                                               math.radians(CAMERA_V2_MAX_YAW_STEP_DEG))
        step_long = _clamp(CAMERA_V2_LONG_GAIN * a_use, CAMERA_V2_MAX_LONG_STEP_M)
        if self.on_step is not None and (step_yaw or step_long):
            self.on_step(self.capture_rel, self.capture_pose, step_long, step_yaw)
        self.applied_yaw += step_yaw
        self.applied_long += step_long
        # Back onto the grid axis, from the belief as it now stands.
        th_now = self.capture_pose[2] + step_yaw
        turn = math.degrees(_wrap(snap_to_axis(th_now) - th_now))
        last = len(self.iterations) + 1 >= self.max_iterations
        if abs(turn) < CAMERA_V2_MIN_TURN_DEG:
            turn = 0.0
        record(step_long, step_yaw, turn)
        if turn:
            self.link.turn(turn, self.turn_speed)
            self.turned_deg += turn
            self._issued = True
            self._seen_active = False
            self._active_by = self._clock() + SWEEP_START_GRACE_S
            self._phase = "turning"
            self._last_after_turn = last
            self.status = f"girando {turn:+.1f} deg para o eixo"
            return "pending"
        if last:
            return self._finish(CORRECTED, "sem mais medidas")
        return self._next_iteration()

    def _do_turning(self, drive_tel) -> str:
        if drive_tel.get("goal_active"):
            self._seen_active = True
        if not self._seen_active and self._clock() < self._active_by:
            return "pending"
        if drive_tel.get("goal_active"):
            return "pending"
        if self._last_after_turn:
            return self._finish(CORRECTED, "sem mais medidas")
        return self._next_iteration(SWEEP_SETTLE_S)

    def _next_iteration(self, settle: float = CAMERA_FIX_KART_SETTLE_S) -> str:
        self._phase = "settle_kart"
        self._settle_until = self._clock() + settle
        self._odom_capture = None
        self.status = f"medida {len(self.iterations) + 1}: assentando kart"
        return "pending"

    def _abort(self, reason: str) -> str:
        self._cancel = True
        self._queue.put(None)
        self._home()
        if self.iterations:
            return self._finish(CORRECTED, reason)
        return self._finish(REFUSED, reason)

    def _finish(self, outcome: str, reason: str) -> str:
        self._home()
        self.result = CameraV2Result(outcome, reason, tuple(self.iterations),
                                     self.applied_long, self.applied_yaw, self.turned_deg,
                                     self._clock() - self._t0, tuple(self._debug_paths))
        self.status = self.result.summary()
        self._phase = "done"
        return "done"

    def group_lines(self) -> List[str]:
        return describe_groups(self.fit.groups) if self.fit is not None else []

    # -- diagnostics -----------------------------------------------------------
    def _dump(self, fit: EdgeFit) -> None:
        """One PNG per view: the frame, the real box boundary (green), the
        simulated edges at the believed pose (red) and at the fitted one
        (cyan)."""
        os.makedirs(self.dump_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        believed = self.params.build(self.params.x0)
        for name, real in self.reals.items():
            img = self.frames.get(name)
            if img is None:
                continue
            out = img.copy()
            edge = (real.field > 0).astype(np.uint8)
            edge = cv2.morphologyEx(edge, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
            out[edge & real.valid] = (0, 200, 0)
            pin, pose = believed[name]
            before = sim_edges(real.spec, pin, pose, self.world, valid=real.valid)
            for pts, color in ((before, (0, 0, 255)), (fit.edges.get(name), (255, 255, 0))):
                if pts is None:
                    continue
                for u, v in zip(pts.u, pts.v):
                    cv2.circle(out, (int(round(u)), int(round(v))), 1, color, -1)
            cv2.putText(out, f"{name} cor {real.accuracy * 100:.0f}%", (6, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            path = os.path.join(self.dump_dir, f"{stamp}_{self.purpose}_v2_{name}"
                                f"_it{len(self.iterations) + 1}.png")
            if cv2.imwrite(path, out):
                self._debug_paths.append(path)
