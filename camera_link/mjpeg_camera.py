"""
MJPEG reader: one background thread holding the latest decoded BGR frame.
The /video endpoint is `multipart/x-mixed-replace; boundary=frame`, but we
don't parse the multipart framing at all -- we just pull complete JPEGs out
of the byte stream by their SOI (FF D8) / EOI (FF D9) markers, which is
robust to the exact boundary formatting.
"""
import threading
import time
import urllib.request

import cv2
import numpy as np

from camera_link.push_frame_to_sensor import push_frame_to_sensor


class MjpegCamera:
    def __init__(self, url):
        self.url = url
        self._lock = threading.Lock()
        self._frame = None
        self._count = 0                 # bumped on every new frame
        self._stamp = 0.0               # time.monotonic() the frame was decoded
        self._pushed = -1               # last count written to a vision sensor
        self.running = True
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def latest(self):
        with self._lock:
            return self._frame, self._count

    def latest_stamped(self):
        """(frame, count, t_recv): the newest frame plus the monotonic time it
        finished decoding here. A consumer that must only see frames taken
        AFTER something happened (the kart stopped, a servo settled) needs the
        time, not just the count -- a new count can still be a frame that was
        captured before the event and spent the latency in transit."""
        with self._lock:
            return self._frame, self._count, self._stamp

    def pump(self, sim, sensor, res_x, res_y, flip=True):
        """Push the newest frame into `sensor`; skip if it is one we already
        pushed. Returns True when the sensor was actually updated."""
        frame, count = self.latest()
        if frame is None or count == self._pushed:
            return False
        self._pushed = count
        push_frame_to_sensor(sim, sensor, frame, res_x, res_y, flip)
        return True

    def describe(self):
        return "client" if self._frame is not None else "client (no frames)"

    def stop(self):
        self.running = False
        self._t.join(timeout=1.0)

    def _run(self):
        while self.running:
            resp = None
            try:
                resp = urllib.request.urlopen(self.url, timeout=10)
                print(f"[cam] connected: {self.url}")
                buf = b""
                while self.running:
                    chunk = resp.read(8192)
                    if not chunk:
                        raise ConnectionError("stream closed by peer")
                    buf += chunk

                    # Drain every complete JPEG currently in the buffer, keeping
                    # only the newest so we never fall behind the live stream.
                    while True:
                        a = buf.find(b"\xff\xd8")                 # SOI
                        b = buf.find(b"\xff\xd9", a + 2) if a != -1 else -1  # EOI
                        if a == -1 or b == -1:
                            break
                        jpg = buf[a:b + 2]
                        buf = buf[b + 2:]
                        img = cv2.imdecode(np.frombuffer(jpg, np.uint8),
                                           cv2.IMREAD_COLOR)
                        if img is not None:
                            with self._lock:
                                self._frame = img
                                self._count += 1
                                self._stamp = time.monotonic()

                    # Guard against unbounded growth if we somehow get garbage.
                    if len(buf) > 4_000_000:
                        buf = buf[-1_000_000:]

            except Exception as e:                                # noqa: BLE001
                # Silent once stopped: this is a daemon thread, and one still
                # blocked in urlopen when its owner exits must not reach for
                # stdout during interpreter shutdown (CPython aborts with
                # "could not acquire lock for <stdout>" if it does).
                if not self.running:
                    break
                print(f"[cam] stream error ({e}); reconnecting...")
                time.sleep(1.0)
            finally:
                try:
                    if resp:
                        resp.close()
                except Exception:                                 # noqa: BLE001
                    pass
