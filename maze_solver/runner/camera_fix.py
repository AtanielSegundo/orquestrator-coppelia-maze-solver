"""
One camera pose fix: point the head at each view, grab fresh frames, measure the
floor-contact profile, fit the pose correction. Non-blocking, polled once per
frame like `Recenter` and `Sweep`.

    settle_kart -> preposition -> [aim -> wait_servo -> settle_servo -> capture]
                -> processing -> done | refused

Who does what
-------------
The MAIN thread only issues servo commands, watches timers and telemetry, and
picks frame references out of the MJPEG reader. Everything with a cost --
undistortion, segmentation, profile extraction, the fit -- runs on this
session's worker thread, fed one view at a time through a queue, so a view's
coarse cost table is computed while the head is still slewing to the next one.

Freshness
---------
A frame is only used if it was DECODED at least CAMERA_FIX_FRAME_LATENCY_S after
the head settled (`MjpegCamera.latest_stamped`). A new frame count alone is not
enough: the frame that arrives right after a servo move was usually taken
during it and spent the latency in transit.

Every view is approached from the same side (a preposition move below the
lowest pan first), so servo backlash lands each view on the same degree it was
calibrated at.

The capture pose
----------------
The believed pose and the odometry sample are FROZEN when the kart has settled.
The fit is a correction of that pose, and `capture_rel` is what the caller hands
to `SimMirror.reanchor` -- not whatever odometry says when the fit finishes. If
odometry moves more than CAMERA_FIX_KART_MOVED_* during the fix, the session
aborts: the frames would then describe a pose nobody recorded.

`poll()` returns "pending", "done" (result holds a fit, possibly "recusado") or
"refused" (could not run at all). The head is always sent back to the home pan
at the fix tilt before the session ends, whatever the outcome.
"""
from __future__ import annotations

import math
import os
import queue
import threading
import time
from typing import Dict, List, NamedTuple, Optional, Tuple

import cv2
import numpy as np

from camera_link.contact_profile import (FREE, HIT, NEAR, extract_contact_profile,
                                         median_profile)
from camera_link.floor_segmenter import FloorSegmenter
from runner.camera_fit import (REFUSED, CameraFit, CameraFitter, FitParams,
                               ViewMeasurement)
from runner.constants import (
    CAMERA_FIX_FRAME_LATENCY_S, CAMERA_FIX_FRAMES_PER_VIEW,
    CAMERA_FIX_KART_MOVED_DEG, CAMERA_FIX_KART_MOVED_M,
    CAMERA_FIX_KART_SETTLE_S, CAMERA_FIX_MIN_HITS_PER_VIEW,
    CAMERA_FIX_MIN_RUN_PX, CAMERA_FIX_PREPOSITION_PCT,
    CAMERA_FIX_SERVO_ACK_TIMEOUT_S, CAMERA_FIX_SERVO_SETTLE_S,
    CAMERA_FIX_VIEW_TIMEOUT_S, CAMERA_SEG_COMPONENTS, CAMERA_SEG_L_WEIGHT,
    CAMERA_SEG_LABEL_MARGIN_PX, CAMERA_SEG_MIN_ACCURACY, CAMERA_SEG_MIN_PIXELS,
)
from runner.contact_model import predict_at, vetoed_columns
from runner.reference_renderer import render_labels, upsample_labels
from runner.wall_world import WallWorld

from shadow.compose import compose
from shadow.inverse import inverse

# Label stride of the per-view prediction the colour model trains on: half
# resolution is plenty for choosing training pixels, at a quarter of the cost.
_LABEL_STRIDE = 2


class CameraFixResult(NamedTuple):
    outcome: str                     # corrigido | confirmado | recusado
    reason: str
    fit: Optional[CameraFit]
    capture_pose: Optional[Tuple[float, float, float]]
    capture_rel: Optional[Tuple[float, float, float]]
    views_used: Tuple[str, ...]
    duration_s: float
    debug_paths: Tuple[str, ...]

    def summary(self) -> str:
        if self.fit is not None and self.outcome != REFUSED:
            return self.fit.describe()
        return f"recusado: {self.reason}"


class _Bundle(NamedTuple):
    name: str
    frames: List[np.ndarray]


def _moved(a, b) -> Tuple[float, float]:
    """(metres, degrees) between two odometry poses."""
    d = math.hypot(a[0] - b[0], a[1] - b[1])
    th = math.degrees(abs((a[2] - b[2] + math.pi) % (2 * math.pi) - math.pi))
    return d, th


class CameraFix:
    def __init__(self, calib, geoms: Dict, head, stream, link, world: WallWorld,
                 odom_ref, purpose: str = "medicao",
                 views: Optional[List[str]] = None,
                 params: Optional[FitParams] = None,
                 dump_dir: Optional[str] = None,
                 clock=time.monotonic):
        self.calib = calib
        self.geoms = geoms
        self.head = head
        self.stream = stream
        self.link = link
        self.world = world
        self.odom_ref = odom_ref
        self.purpose = purpose
        self.params = params or FitParams()
        self.dump_dir = dump_dir
        self._clock = clock
        self._t0 = clock()
        self.status = "assentando kart"
        self.result: Optional[CameraFixResult] = None

        names = views if views is not None else [v.name for v in calib.views]
        # Approach order: increasing pan percent, from a preposition below the
        # lowest -- every view is reached moving the same way.
        self._views = sorted((calib.view(n) for n in names), key=calib.pan_pct)
        self._phase = "settle_kart"
        self._settle_until = self._t0 + CAMERA_FIX_KART_SETTLE_S
        self._index = 0
        self._deadline = 0.0
        self._ready_at = 0.0
        self._last_count = -1
        self._frames: List[np.ndarray] = []
        self._skipped: Dict[str, str] = {}

        self.capture_pose = None
        self.capture_rel = None
        self._odom_capture = None
        self._fitter: Optional[CameraFitter] = None
        self._queue: "queue.Queue[Optional[_Bundle]]" = queue.Queue()
        self._done = threading.Event()
        self._cancel = False
        self._thread: Optional[threading.Thread] = None
        self._worker_fit: Optional[CameraFit] = None
        self._worker_error: Optional[str] = None
        self._worker_dropped: Dict[str, str] = {}
        self._debug: Dict[str, tuple] = {}
        self._debug_paths: List[str] = []
        self.segmenter = FloorSegmenter(components=CAMERA_SEG_COMPONENTS,
                                        l_weight=CAMERA_SEG_L_WEIGHT,
                                        margin_px=CAMERA_SEG_LABEL_MARGIN_PX,
                                        min_pixels=CAMERA_SEG_MIN_PIXELS,
                                        min_accuracy=CAMERA_SEG_MIN_ACCURACY)

    # -- head ----------------------------------------------------------------
    def _command(self, pan_pct: int) -> None:
        self._commanded = (int(pan_pct), int(self.calib.tilt_pct))
        self.head.set_percents(*self._commanded)
        self._deadline = self._clock() + CAMERA_FIX_SERVO_ACK_TIMEOUT_S

    def _acked(self) -> bool:
        """The kart reports the commanded angles (server bookkeeping, see
        Server/servo.py), or it reports nothing at all (older server: nothing
        to wait for), or the ack timed out."""
        if self._clock() >= self._deadline:
            return True
        tel = {}
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

    # -- main poll -----------------------------------------------------------
    def poll(self, drive_tel: dict, odom_pose, wx: float, wy: float, wth: float) -> str:
        if self._phase == "done":
            return "done"
        if self._phase == "refused":
            return "refused"
        if self._phase == "settle_kart":
            return self._do_settle_kart(drive_tel, odom_pose, wx, wy, wth)
        if self._odom_capture is not None and odom_pose is not None \
                and self._phase != "processing":
            d, th = _moved(odom_pose, self._odom_capture)
            if d > CAMERA_FIX_KART_MOVED_M or th > CAMERA_FIX_KART_MOVED_DEG:
                return self._abort(f"kart se mexeu durante a captura "
                                   f"({d * 100:.1f} cm, {th:.1f} deg)")
        if self._phase == "preposition":
            return self._do_preposition()
        if self._phase == "aim":
            return self._do_aim()
        if self._phase == "wait_servo":
            return self._do_wait_servo()
        if self._phase == "settle_servo":
            return self._do_settle_servo()
        if self._phase == "capture":
            return self._do_capture()
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

        # Only spend servo time on views the map says can see something.
        keep = []
        for view in self._views:
            _lo, _hi, kind = predict_at(self.geoms[view.name], self.capture_pose, self.world)
            hits = int(np.count_nonzero(kind == HIT))
            if hits >= CAMERA_FIX_MIN_HITS_PER_VIEW:
                keep.append(view)
            else:
                self._skipped[view.name] = f"mapa preve {hits} contatos"
        self._views = keep
        if not keep:
            return self._finish_refused("nada visivel daqui ("
                                        + ", ".join(f"{k}: {v}" for k, v in self._skipped.items())
                                        + ")")
        self._fitter = CameraFitter(self.world, self.capture_pose, self.params)
        self._thread = threading.Thread(target=self._worker, daemon=True,
                                        name="CameraFix")
        self._thread.start()
        lowest = self.calib.pan_pct(self._views[0])
        self._command(max(0, lowest - CAMERA_FIX_PREPOSITION_PCT))
        self._phase = "preposition"
        self.status = "preposicionando camera"
        return "pending"

    def _do_preposition(self) -> str:
        if not self._acked():
            return "pending"
        self._phase = "aim"
        return "pending"

    def _do_aim(self) -> str:
        view = self._views[self._index]
        self._command(self.calib.pan_pct(view))
        self._phase = "wait_servo"
        self.status = f"mirando {view.name}"
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
        # Frames decoded from now + latency on were taken after the head
        # stopped.
        self._ready_at = self._clock() + CAMERA_FIX_FRAME_LATENCY_S
        self._deadline = self._clock() + CAMERA_FIX_VIEW_TIMEOUT_S
        _f, count, _t = self.stream.latest_stamped()
        self._last_count = count
        self._frames = []
        self._phase = "capture"
        self.status = f"capturando {self._views[self._index].name}"
        return "pending"

    def _do_capture(self) -> str:
        view = self._views[self._index]
        frame, count, t_recv = self.stream.latest_stamped()
        if frame is not None and count != self._last_count and t_recv >= self._ready_at:
            self._frames.append(frame.copy())
            self._last_count = count
        if len(self._frames) >= CAMERA_FIX_FRAMES_PER_VIEW:
            self._queue.put(_Bundle(view.name, self._frames))
            return self._next_view()
        if self._clock() >= self._deadline:
            if self._frames:
                self._queue.put(_Bundle(view.name, self._frames))
            else:
                self._skipped[view.name] = "sem frames novos"
            return self._next_view()
        return "pending"

    def _next_view(self) -> str:
        self._index += 1
        self._frames = []
        if self._index < len(self._views):
            self._phase = "aim"
            return "pending"
        self._home()
        self._queue.put(None)
        self._phase = "processing"
        self.status = "ajustando pose"
        return "pending"

    def _do_processing(self) -> str:
        if not self._done.is_set():
            return "pending"
        dur = self._clock() - self._t0
        if self._worker_error is not None:
            return self._finish(REFUSED, self._worker_error, None, dur)
        fit = self._worker_fit
        notes = {**self._skipped, **self._worker_dropped}
        reason = fit.reason
        if fit.outcome == REFUSED and notes:
            reason += " | " + ", ".join(f"{k}: {v}" for k, v in notes.items())
        return self._finish(fit.outcome, reason, fit, dur)

    def _abort(self, reason: str) -> str:
        self._cancel = True
        self._queue.put(None)
        self._home()
        return self._finish_refused(reason)

    def _finish_refused(self, reason: str) -> str:
        self.result = CameraFixResult(REFUSED, reason, None, self.capture_pose,
                                      self.capture_rel, (), self._clock() - self._t0, ())
        self.status = reason
        self._phase = "refused"
        return "refused"

    def _finish(self, outcome, reason, fit, dur) -> str:
        views = fit.views if fit is not None else ()
        self.result = CameraFixResult(outcome, reason, fit, self.capture_pose,
                                      self.capture_rel, views, dur,
                                      tuple(self._debug_paths))
        self.status = self.result.summary()
        self._phase = "done"
        return "done"

    # -- worker --------------------------------------------------------------
    def _worker(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is None or self._cancel:
                    break
                self._process_view(item)
            if self._cancel:
                return
            fit = self._fitter.solve()
            self._worker_dropped.update(self._fitter.dropped)
            self._worker_fit = fit
            if self.dump_dir:
                self._dump(fit)
        except Exception as exc:                                # noqa: BLE001
            self._worker_error = f"falha no ajuste: {type(exc).__name__}: {exc}"
        finally:
            self._done.set()

    def _prepare(self, frame: np.ndarray):
        """Frame -> (image in the ideal pinhole, invalid-pixel mask or None)."""
        W, H = self.calib.width, self.calib.height
        if frame.shape[1] != W or frame.shape[0] != H:
            if abs(frame.shape[1] / frame.shape[0] - W / H) > 0.01:
                raise ValueError(f"frame {frame.shape[1]}x{frame.shape[0]} "
                                 f"nao bate com a calibracao {W}x{H}")
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        if not self.calib.needs_remap():
            return frame, None
        maps = getattr(self.calib, "_maps", None)
        if maps is None:
            m1, m2 = cv2.initUndistortRectifyMap(
                self.calib.K(), np.asarray(self.calib.dist, dtype=float), None,
                self.calib.K_ideal(), (W, H), cv2.CV_32FC1)
            ones = np.full((H, W), 255, np.uint8)
            invalid = cv2.remap(ones, m1, m2, cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0) == 0
            maps = (m1, m2, invalid)
            self.calib._maps = maps
        m1, m2, invalid = maps
        return cv2.remap(frame, m1, m2, cv2.INTER_LINEAR), invalid

    def _process_view(self, bundle: _Bundle) -> None:
        geom = self.geoms[bundle.name]
        prepared = [self._prepare(f) for f in bundle.frames]
        img0, invalid = prepared[0]
        H, W = img0.shape[:2]
        labels = render_labels(geom.pinhole, self.capture_pose, self.world,
                               max_range=geom.max_range, stride=_LABEL_STRIDE,
                               extra_boxes=self.world.obstacles)
        labels = upsample_labels(labels, _LABEL_STRIDE, (H, W))
        exclude = np.zeros((H, W), dtype=bool)
        exclude[geom.v_near + 1:, :] = True               # kart body band
        if invalid is not None:
            exclude |= invalid
        model, why = self.segmenter.train(img0, labels, exclude)
        if model is None:
            self._worker_dropped[bundle.name] = why
            return
        profiles = []
        mask0 = None
        for img, inv in prepared:
            llr = self.segmenter.llr(img, model)
            wall = FloorSegmenter.wall_mask(llr)
            if inv is not None:
                wall &= ~inv
            if mask0 is None:
                mask0 = wall
            profiles.append(extract_contact_profile(
                wall, geom.columns, geom.v_near, geom.v_far,
                CAMERA_FIX_MIN_RUN_PX, llr=llr, invalid=inv))
        profile = median_profile(profiles)
        veto = vetoed_columns(geom, self.capture_pose, self.world)
        self._fitter.add_view(ViewMeasurement(geom, profile, veto))
        self._debug[bundle.name] = (img0, mask0, profile, model.accuracy)

    # -- diagnostics ---------------------------------------------------------
    def _dump(self, fit: CameraFit) -> None:
        """One PNG per view: the frame, walls tinted, measured contacts
        (green HIT, yellow FREE, magenta NEAR), the model at the believed pose
        (red) and at the corrected one (cyan)."""
        os.makedirs(self.dump_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        x0, y0, th0 = self.capture_pose
        a, b, psi = fit.estimate
        c, s = math.cos(th0), math.sin(th0)
        corrected = (x0 + c * a - s * b, y0 + s * a + c * b, th0 + psi)
        for name, (img, mask, prof, acc) in self._debug.items():
            geom = self.geoms[name]
            out = img.copy()
            if mask is not None:
                tint = out.copy()
                tint[mask] = (0.5 * tint[mask] + np.array([0, 0, 127])).astype(np.uint8)
                out = tint
            cv2.line(out, (0, geom.v_near), (out.shape[1] - 1, geom.v_near), (80, 80, 80), 1)
            for i, u in enumerate(geom.columns):
                k = prof.kind[i]
                if k == HIT:
                    cv2.circle(out, (int(u), int(round(prof.rows[i]))), 2, (0, 255, 0), -1)
                elif k == FREE and np.isfinite(geom.v_far[i]):
                    cv2.circle(out, (int(u), int(geom.v_far[i])), 1, (0, 255, 255), -1)
                elif k == NEAR:
                    cv2.circle(out, (int(u), geom.v_near), 1, (255, 0, 255), -1)
            eta = fit.etas.get(name, 0.0)
            for pose, color in ((self.capture_pose, (0, 0, 255)),
                                ((corrected[0], corrected[1], corrected[2] + eta), (255, 255, 0))):
                lo, _hi, kind = predict_at(geom, pose, self.world)
                pts = [(int(u), int(round(v))) for u, v, kd in zip(geom.columns, lo, kind)
                       if kd == HIT]
                for p0, p1 in zip(pts, pts[1:]):
                    if abs(p0[1] - p1[1]) < 12:
                        cv2.line(out, p0, p1, color, 1)
            cv2.putText(out, f"{name} cor {acc * 100:.0f}% | {fit.outcome}", (6, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            path = os.path.join(self.dump_dir, f"{stamp}_{self.purpose}_{name}.png")
            if cv2.imwrite(path, out):
                self._debug_paths.append(path)
