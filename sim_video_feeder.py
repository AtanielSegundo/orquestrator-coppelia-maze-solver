#python
"""
In-scene MJPEG feeder -- this file runs INSIDE CoppeliaSim, not in the orchestrator.

Why it exists
-------------
Pushing the kart's camera into a vision sensor from an external client means
shipping a full raw RGB image (640x480x3 = ~900 KB) through the ZMQ remote API
on EVERY frame. That is not a "minimal" cost: at 25 fps it is ~23 MB/s of
request/reply traffic, and each `setVisionSensorImg` call blocks the control
loop until CoppeliaSim answers.

So instead we move the whole pipeline into the scene: this script listens on the
loopback interface, the orchestrator hands it a VIDEO_URL, and from then on THIS
process pulls the MJPEG stream and writes the frames straight into the vision
sensor with an in-process API call. The control loop's per-frame cost drops to a
throttled status ping (a few dozen bytes, about once a second).

Install
-------
1. In CoppeliaSim, add a **customization script** (Add > Script > Customization
   script > Python) to any object -- the vision sensor itself is the tidiest.
   A *customization* script is required (not a child script) so the feed also
   runs while the simulation is STOPPED, which is how the mirror is normally
   used. Do NOT keep the threaded template's `sysCall_thread`: a customization
   script's thread only runs during simulation, and the callbacks below cover
   both states. (If you do want a threaded script, drop `tick()` into your loop.)

2. Paste this stub into the script -- it keeps the real code here on disk, so
   editing this file and reloading the scene is the whole edit cycle:

       #python
       import sys
       sys.path.append('C:/Users/Korvo/scripts/embeded/rasp/orquestrator')
       import sim_video_feeder
       from sim_video_feeder import (sysCall_nonSimulation, sysCall_actuation,
                                     sysCall_cleanup)

       def sysCall_init():
           sim_video_feeder.attach(require('sim'))

   `require` is injected into the SCRIPT's namespace, not into this module's,
   which is why the stub passes `sim` in rather than letting sysCall_init here
   look it up. (Pasting this entire file in works too -- then `require` is in
   scope and the sysCall_init below is the one that runs.) Forward slashes in
   the path on purpose: they work on Windows and dodge backslash escaping.

3. CoppeliaSim's Python interpreter (`defaultPython` in `usrset.txt`; when empty
   it is whatever `python` resolves to on PATH) needs `opencv-python` and
   `numpy`. If they are missing the script still loads and says so in `status`,
   it just cannot decode anything.

Protocol (one JSON object per line, one request per connection)
--------------------------------------------------------------
    -> {"cmd": "start", "url": "http://host:8080/video", "sensor": "/realCamera",
        "flip": true}
    -> {"cmd": "status"}
    -> {"cmd": "stop"}
    <- {"ok": true, "running": true, "frames": 812, "pushed": 810, "fps": 24.6,
        "res": [640, 480], "error": null, ...}

The client side of this protocol is `SimSideCamera` in camera_link.py.

NOTE: the MJPEG scanning below is deliberately a copy of `MjpegCamera`'s rather
than an import -- this file has to stand alone inside CoppeliaSim's interpreter,
which does not have the project on its path unless you use the stub above.
"""
import json
import socket
import threading
import time

try:
    import cv2
    import numpy as np
    _IMPORT_ERROR = None
except Exception as e:                                    # noqa: BLE001
    cv2 = np = None
    _IMPORT_ERROR = f"{type(e).__name__}: {e}"

# ---------------------------------------------------------------------------
# Config (must match camera_link.SimSideCamera's defaults)
# ---------------------------------------------------------------------------
HOST       = "127.0.0.1"
PORT       = 28080       # NOT 23050: that is CoppeliaSim's own WS remote API server
WATCHDOG_S = 15.0        # stop feeding if no client spoke for this long (0 = off)

sim   = None
_ctl  = None             # ControlServer
_feed = None             # MjpegFeed
_lock = threading.Lock()

# A start request parked by the control thread for the script thread to pick up.
# Every sim.* call has to happen on the script thread inside a sysCall_*, so the
# socket thread never touches the sim API -- it only leaves a note here.
_pending      = None
_pending_done = threading.Event()
_pending_err  = None

_last_contact = 0.0


# ---------------------------------------------------------------------------
# MJPEG reader: pulls the stream and pre-bakes each frame into the exact byte
# buffer the vision sensor wants, so the script thread only does the API call.
# ---------------------------------------------------------------------------
class MjpegFeed:
    def __init__(self, url, res_x, res_y, flip=True):
        self.url = url
        self.res_x = res_x
        self.res_y = res_y
        self.flip = flip
        self.running = True
        self.error = None
        self.pushed = -1                  # last count written to the sensor

        self._lock = threading.Lock()
        self._buf = None                  # ready-to-push RGB bytes
        self._count = 0
        self._stamps = []                 # recent frame times, for fps
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def latest(self):
        with self._lock:
            return self._buf, self._count

    def fps(self):
        with self._lock:
            stamps = list(self._stamps)
        if len(stamps) < 2:
            return 0.0
        span = stamps[-1] - stamps[0]
        return (len(stamps) - 1) / span if span > 0 else 0.0

    def stop(self):
        self.running = False
        self._t.join(timeout=1.5)

    def _store(self, img):
        img = cv2.resize(img, (self.res_x, self.res_y))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.flip:
            img = cv2.flip(img, 0)        # CoppeliaSim's bottom-up convention
        buf = img.tobytes()
        now = time.time()
        with self._lock:
            self._buf = buf
            self._count += 1
            self._stamps.append(now)
            if len(self._stamps) > 30:
                del self._stamps[:-30]

    def _run(self):
        # urllib is imported lazily so a missing cv2 fails with a clear message
        # instead of dying somewhere in the middle of the stream loop.
        import urllib.request

        while self.running:
            resp = None
            try:
                resp = urllib.request.urlopen(self.url, timeout=10)
                self.error = None
                print(f"[feeder] connected: {self.url}")
                raw = b""
                while self.running:
                    chunk = resp.read(8192)
                    if not chunk:
                        raise ConnectionError("stream closed by peer")
                    raw += chunk

                    # Drain every complete JPEG in the buffer, keeping only the
                    # newest, so we never fall behind the live stream.
                    while True:
                        a = raw.find(b"\xff\xd8")                          # SOI
                        b = raw.find(b"\xff\xd9", a + 2) if a != -1 else -1  # EOI
                        if a == -1 or b == -1:
                            break
                        jpg = raw[a:b + 2]
                        raw = raw[b + 2:]
                        img = cv2.imdecode(np.frombuffer(jpg, np.uint8),
                                           cv2.IMREAD_COLOR)
                        if img is not None:
                            self._store(img)

                    if len(raw) > 4_000_000:               # garbage guard
                        raw = raw[-1_000_000:]

            except Exception as e:                         # noqa: BLE001
                self.error = f"{type(e).__name__}: {e}"
                if self.running:
                    print(f"[feeder] stream error ({e}); reconnecting...")
                    time.sleep(1.0)
            finally:
                try:
                    if resp:
                        resp.close()
                except Exception:                          # noqa: BLE001
                    pass


# ---------------------------------------------------------------------------
# Control server: one JSON request per connection, on loopback only.
# ---------------------------------------------------------------------------
class ControlServer:
    def __init__(self, host, port):
        self.running = True
        self.error = None
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((host, port))
        self._srv.listen(4)
        self._srv.settimeout(0.5)                # so we can notice .running
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        print(f"[feeder] listening on {host}:{port}")

    def stop(self):
        self.running = False
        try:
            self._srv.close()
        except Exception:                        # noqa: BLE001
            pass
        self._t.join(timeout=1.5)

    def _run(self):
        while self.running:
            try:
                conn, _ = self._srv.accept()
            except (socket.timeout, OSError):
                continue
            try:
                conn.settimeout(2.0)
                raw = b""
                while b"\n" not in raw and len(raw) < 65536:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    raw += chunk
                if raw:
                    line = raw.split(b"\n", 1)[0].decode("utf-8", "replace")
                    reply = _handle(json.loads(line))
                    conn.sendall((json.dumps(reply) + "\n").encode("utf-8"))
            except Exception as e:               # noqa: BLE001
                try:
                    conn.sendall((json.dumps({"ok": False,
                                              "error": f"{type(e).__name__}: {e}"})
                                  + "\n").encode("utf-8"))
                except Exception:                # noqa: BLE001
                    pass
            finally:
                try:
                    conn.close()
                except Exception:                # noqa: BLE001
                    pass


# ---------------------------------------------------------------------------
# Command handling (control thread -- must not touch the sim API)
# ---------------------------------------------------------------------------
def _status(ok=True, error=None):
    with _lock:
        feed = _feed
    running = feed is not None and feed.running
    return {
        "ok": bool(ok),
        "running": bool(running),
        "url": feed.url if feed else None,
        "res": [feed.res_x, feed.res_y] if feed else None,
        "frames": feed._count if feed else 0,
        "pushed": (feed.pushed + 1) if feed else 0,
        "fps": round(feed.fps(), 2) if feed else 0.0,
        "error": error or (feed.error if feed else None) or _IMPORT_ERROR,
    }


def _handle(msg):
    global _pending, _pending_err, _last_contact

    _last_contact = time.time()
    cmd = msg.get("cmd")

    if cmd == "status":
        return _status()

    if cmd == "stop":
        _stop_feed()
        return _status()

    if cmd == "start":
        if _IMPORT_ERROR is not None:
            return _status(ok=False, error=f"opencv/numpy unavailable in "
                                           f"CoppeliaSim's Python ({_IMPORT_ERROR})")
        url = msg.get("url")
        if not url:
            return _status(ok=False, error="'start' needs a 'url'")

        with _lock:
            feed = _feed
        # Already streaming exactly this? Then 'start' is just a heartbeat.
        if (feed is not None and feed.running and feed.url == url
                and not feed.error):
            return _status()

        _pending_err = None
        _pending_done.clear()
        _pending = {"url": url,
                    "sensor": msg.get("sensor", "/realCamera"),
                    "flip": bool(msg.get("flip", True))}
        # The script thread resolves the sensor and starts the feed; wait briefly
        # so the caller gets the real outcome ("sensor not found") instead of a
        # hopeful "ok".
        if not _pending_done.wait(timeout=2.0):
            return _status(ok=False, error="scene script did not pick the request "
                                           "up (is the simulation/script running?)")
        return _status(ok=_pending_err is None, error=_pending_err)

    return _status(ok=False, error=f"unknown cmd {cmd!r}")


def _stop_feed():
    global _feed
    with _lock:
        feed, _feed = _feed, None
    if feed is not None:
        feed.stop()
        print("[feeder] stopped")


# ---------------------------------------------------------------------------
# Script thread: the ONLY place that talks to the sim API
# ---------------------------------------------------------------------------
def _start_pending():
    global _feed, _pending, _pending_err

    req, _pending = _pending, None
    try:
        handle = sim.getObject(req["sensor"])
        res_x, res_y = sim.getVisionSensorRes(handle)
        _stop_feed()
        feed = MjpegFeed(req["url"], res_x, res_y, req["flip"])
        feed.sensor = handle
        with _lock:
            _feed = feed
        _pending_err = None
        print(f"[feeder] {req['url']} -> {req['sensor']} ({res_x}x{res_y})")
    except Exception as e:                       # noqa: BLE001
        _pending_err = f"{type(e).__name__}: {e}"
        print(f"[feeder] start failed: {_pending_err}")
    finally:
        _pending_done.set()


def _tick():
    if _pending is not None:
        _start_pending()

    with _lock:
        feed = _feed
    if feed is None:
        return

    if WATCHDOG_S > 0 and _last_contact and (time.time() - _last_contact) > WATCHDOG_S:
        print("[feeder] no client for "
              f"{WATCHDOG_S:.0f}s; stopping the stream")
        _stop_feed()
        return

    buf, count = feed.latest()
    if buf is not None and count != feed.pushed:
        feed.pushed = count
        sim.setVisionSensorImg(feed.sensor, buf, 0)


# ---------------------------------------------------------------------------
# CoppeliaSim entry points
# ---------------------------------------------------------------------------
def attach(sim_api):
    """Hand this module the scene's `sim` object and open the control port.

    Separate from sysCall_init because `require` is injected into the SCRIPT's
    namespace, and these functions run with THIS module's namespace when the
    script pulls them in with `from sim_video_feeder import *`. The stub calls
    `attach(require('sim'))` where `require` actually exists.
    """
    global sim, _ctl
    sim = sim_api
    if _IMPORT_ERROR is not None:
        print(f"[feeder] WARNING: cannot decode video -- {_IMPORT_ERROR}")
    if _ctl is not None:                         # re-attached without a cleanup
        _ctl.stop()
        _ctl = None
    try:
        _ctl = ControlServer(HOST, PORT)
    except Exception as e:                       # noqa: BLE001
        _ctl = None
        print(f"[feeder] could not listen on {HOST}:{PORT} ({e}) -- "
              "is another copy of this script already running?")


def sysCall_init():
    # Only reached when the whole file is pasted into the scene, in which case
    # `require` IS in scope. The stub defines its own sysCall_init instead.
    attach(require('sim'))                       # noqa: F821 (CoppeliaSim injects it)


def tick():
    """Public alias for the frame pump. Safe to call from any sysCall_* and
    idempotent, so a threaded script can call it in its own loop instead."""
    _tick()


def sysCall_nonSimulation():                     # simulation stopped
    _tick()


def sysCall_actuation():                         # simulation running
    _tick()


def sysCall_cleanup():
    global _ctl
    _stop_feed()
    if _ctl is not None:
        _ctl.stop()
        _ctl = None
