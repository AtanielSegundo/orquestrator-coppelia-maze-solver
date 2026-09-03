"""
Sim-side backend: hand the URL to sim_video_feeder.py inside CoppeliaSim and
let IT do the pulling + the sensor writes. Nothing but small JSON control
messages crosses this process boundary, so the frames never pay for the trip.
"""
import json
import socket
import time

from camera_link.constants import FEEDER_HOST, FEEDER_PORT


class SimSideCamera:
    RETRY_S = 2.0            # how often to re-attempt a failed start
    POLL_S  = 1.0            # status poll rate; doubles as the feeder's heartbeat

    def __init__(self, url, sensor_path="/realCamera", flip=True,
                 host=FEEDER_HOST, port=FEEDER_PORT, timeout=2.5, start=True):
        self.url = url
        self.sensor_path = sensor_path
        self.flip = flip
        self.host = host
        self.port = port
        self.timeout = timeout

        self.running = False             # is the in-scene feed streaming?
        self.error = None
        self.fps = 0.0
        self.frames = 0                  # frames the feeder has decoded
        self._next_call = 0.0
        if start:
            self.start()

    # -- control protocol --------------------------------------------------
    def _rpc(self, payload):
        """One request, one reply, one short-lived connection. Keeping no
        persistent socket means a restarted scene script needs no reconnect
        logic here -- the next call just works."""
        with socket.create_connection((self.host, self.port), self.timeout) as s:
            s.settimeout(self.timeout)
            s.sendall((json.dumps(payload) + "\n").encode("utf-8"))
            raw = b""
            while b"\n" not in raw:
                chunk = s.recv(4096)
                if not chunk:
                    break
                raw += chunk
        if not raw:
            raise ConnectionError("no reply from the in-scene feeder")
        return json.loads(raw.split(b"\n", 1)[0].decode("utf-8"))

    def _absorb(self, reply):
        self.running = bool(reply.get("running"))
        self.error = reply.get("error")
        self.fps = float(reply.get("fps") or 0.0)
        self.frames = int(reply.get("frames") or 0)
        return bool(reply.get("ok"))

    def _fail(self, e):
        self.running = False
        self.fps = 0.0
        self.error = f"{type(e).__name__}: {e}"

    def start(self):
        """Ask the scene script to stream `url` into `sensor_path`."""
        was_running, was_error = self.running, self.error
        try:
            ok = self._absorb(self._rpc({"cmd": "start",
                                         "url": self.url,
                                         "sensor": self.sensor_path,
                                         "flip": self.flip}))
        except Exception as e:                              # noqa: BLE001
            self._fail(e)
            ok = False

        # Edge-triggered logging: pump() retries every RETRY_S, and a scene that
        # simply has not loaded the script would otherwise spam the console.
        if ok and not was_running:
            print(f"[SimSideCamera] in-scene feeder streaming {self.url} "
                  f"-> '{self.sensor_path}'")
        elif not ok and (was_running or was_error != self.error):
            print(f"[SimSideCamera] feeder unavailable ({self.error}); "
                  "retrying in the background")
        return ok

    def status(self):
        try:
            self._absorb(self._rpc({"cmd": "status"}))
        except Exception as e:                              # noqa: BLE001
            self._fail(e)
        return self.running

    # -- backend interface -------------------------------------------------
    def pump(self, sim=None, sensor=None, res_x=None, res_y=None, flip=None):
        """Throttled poll: the sensor writes happen in the scene, so there is no
        image work here. The arguments exist only so this is swappable with
        MjpegCamera.pump(); the feeder already knows the sensor.

        Returns True if the feeder decoded new frames since the last poll -- a
        liveness signal, not a per-frame one, since we deliberately do not talk
        to the scene 60 times a second.
        """
        now = time.monotonic()
        if now < self._next_call:
            return False
        before = self.frames
        if self.running:
            self._next_call = now + self.POLL_S
            self.status()
        else:
            self._next_call = now + self.RETRY_S
            self.start()                    # scene script may have come up late
        return self.running and self.frames > before

    def describe(self):
        if not self.running:
            return "sim (down)"
        return f"sim {self.fps:.0f}fps"

    def stop(self):
        try:
            self._rpc({"cmd": "stop"})
        except Exception:                                   # noqa: BLE001
            pass                    # scene gone / never up: nothing to stop
        self.running = False
