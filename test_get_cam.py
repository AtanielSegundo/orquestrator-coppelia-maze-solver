"""
Fake Pi Kart server for testing get_cam.py (and KartLink) without the real kart.

Serves on ONE port, like the real server:
    http://HOST:PORT/video   multipart MJPEG, synthetic frames (moving ball + HUD)
    ws://HOST:PORT/ws        telemetry out, commands in (drive / turn /
                             drive_distance / goto / reset_odometry / servo)

The WebSocket side is a tiny stdlib implementation (handshake + framing) so no
extra dependency is needed -- `websocket-client` is client-only.

Usage:
    python test_get_cam.py                 # just serve on 127.0.0.1:8080
    python test_get_cam.py --run           # serve AND launch get_cam.py against it
    python test_get_cam.py -p 9000 --fps 10 --size 320x240

Poke at things by editing FakeKart / draw_frame() below: that is the "fake point"
to experiment on (drop frames, change latency, fake an obstacle, ...).
"""
import argparse
import base64
import hashlib
import json
import math
import socket
import struct
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
BOUNDARY = "frame"


class FakeKart:
    def __init__(self):
        self._lock = threading.Lock()
        self.x = self.y = self.theta = 0.0
        self.linear = self.angular = 0.0
        self.servo = {"0": 90, "1": 90}
        self.front_distance_cm = 120.0
        self._goal = None                 # ("turn", target_theta) | ("dist", sx, sy, dist) | ("goto", x, y)
        self._last = time.monotonic()
        self.frame_count = 0              # bumped by the video side, shown in the HUD

    # -- commands (called from the WS thread) ------------------------------
    def handle(self, msg):
        t = msg.get("type")
        with self._lock:
            if t == "drive":                                   # cancels any goal, like the real server
                self._goal = None
                self.linear = float(msg.get("linear", 0.0))
                self.angular = float(msg.get("angular", 0.0))
            elif t == "reset_odometry":
                self.x = self.y = self.theta = 0.0
            elif t == "turn":
                self._goal = ("turn", self.theta + math.radians(float(msg["angle"])))
            elif t == "drive_distance":
                self._goal = ("dist", self.x, self.y, float(msg["distance"]), float(msg.get("speed", 0.2)))
            elif t == "goto":
                self._goal = ("goto", float(msg["x"]), float(msg["y"]))
            elif t == "servo":
                self.servo[str(msg["channel"])] = int(msg["angle"])
        print(f"[fake] <- {msg}")

    # -- physics -----------------------------------------------------------
    def step(self):
        now = time.monotonic()
        dt, self._last = now - self._last, now
        with self._lock:
            lin, ang = self.linear, self.angular
            g = self._goal
            if g and g[0] == "turn":
                err = _wrap(g[1] - self.theta)
                ang = max(-1.5, min(1.5, 3.0 * err))
                lin = 0.0
                if abs(err) < 0.01:
                    self._goal, ang = None, 0.0
            elif g and g[0] == "dist":
                _, sx, sy, dist, speed = g
                left = dist - math.copysign(math.hypot(self.x - sx, self.y - sy), dist)
                lin, ang = math.copysign(speed, dist), 0.0
                if abs(left) < 0.005 or left * dist < 0:
                    self._goal, lin = None, 0.0
            elif g and g[0] == "goto":
                dx, dy = g[1] - self.x, g[2] - self.y
                if math.hypot(dx, dy) < 0.01:
                    self._goal, lin, ang = None, 0.0, 0.0
                else:
                    err = _wrap(math.atan2(dy, dx) - self.theta)
                    ang = max(-1.5, min(1.5, 3.0 * err))
                    lin = 0.2 if abs(err) < 0.3 else 0.0
            self.theta = _wrap(self.theta + ang * dt)
            self.x += lin * math.cos(self.theta) * dt
            self.y += lin * math.sin(self.theta) * dt
            self._cur = (lin, ang)

    def telemetry(self):
        with self._lock:
            lin, ang = getattr(self, "_cur", (0.0, 0.0))
            half_track = 0.05
            return {
                "type": "telemetry",
                "t": time.time(),
                "drive": {
                    "pose": {"x": self.x, "y": self.y, "theta": self.theta},
                    "wheel_speed": {"left": lin - ang * half_track,
                                    "right": lin + ang * half_track},
                    "engaged": True,
                    "goal_active": self._goal is not None,
                    "guard_blocked": False,
                    "front_distance_cm": self.front_distance_cm,
                },
                "servo": dict(self.servo),
            }


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


KART = FakeKart()
CFG = argparse.Namespace(fps=20, w=640, h=480)


# ---------------------------------------------------------------------------
# Synthetic camera frame. Edit freely.
# ---------------------------------------------------------------------------
def draw_frame(n):
    w, h = CFG.w, CFG.h
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (40, 30, 20)
    # Gradient so a flipped / colour-swapped upload is obvious: blue on top-left,
    # red on bottom-right, in true BGR order.
    xs = np.linspace(0, 255, w, dtype=np.uint8)
    ys = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    img[..., 0] = xs[None, :] // 2 + 30      # B grows to the right
    img[..., 2] = ys // 2 + 30               # R grows downward
    cx = int(w / 2 + (w / 3) * math.sin(n / 25))
    cy = int(h / 2 + (h / 4) * math.cos(n / 17))
    cv2.circle(img, (cx, cy), 30, (0, 255, 255), -1)
    cv2.putText(img, "TOP-LEFT", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"frame {n}  {time.strftime('%H:%M:%S')}", (8, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    p = KART.telemetry()["drive"]["pose"]
    cv2.putText(img, f"x={p['x']:+.2f} y={p['y']:+.2f} th={math.degrees(p['theta']):+.0f}",
                (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return img


# ---------------------------------------------------------------------------
# HTTP + WebSocket handler
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path.startswith("/video"):
            return self._video()
        if self.path.startswith("/ws") and self.headers.get("Upgrade", "").lower() == "websocket":
            return self._ws()
        self.send_error(404)

    # -- /video --------------------------------------------------------------
    def _video(self):
        self.send_response(200)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        print(f"[fake] video client {self.client_address[0]} connected")
        n = 0
        try:
            while True:
                t0 = time.monotonic()
                ok, jpg = cv2.imencode(".jpg", draw_frame(n), [cv2.IMWRITE_JPEG_QUALITY, 80])
                data = jpg.tobytes()
                self.wfile.write(f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                                 f"Content-Length: {len(data)}\r\n\r\n".encode() + data + b"\r\n")
                self.wfile.flush()
                n += 1
                time.sleep(max(0.0, 1.0 / CFG.fps - (time.monotonic() - t0)))
        except (BrokenPipeError, ConnectionError, OSError):
            print("[fake] video client left")

    # -- /ws -----------------------------------------------------------------
    def _ws(self):
        key = self.headers["Sec-WebSocket-Key"]
        accept = base64.b64encode(hashlib.sha1((key + WS_GUID).encode()).digest()).decode()
        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        self.end_headers()
        print(f"[fake] ws client {self.client_address[0]} connected")

        sock = self.connection
        sock.settimeout(0.02)
        buf = bytearray()
        try:
            while True:
                KART.step()
                _ws_send(sock, 0x1, json.dumps(KART.telemetry()).encode())
                try:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    buf += chunk
                except socket.timeout:
                    pass
                while True:
                    frame = _ws_parse(buf)
                    if frame is None:
                        break
                    op, payload = frame
                    if op == 0x8:
                        return
                    if op == 0x9:
                        _ws_send(sock, 0xA, payload)
                    elif op == 0x1:
                        try:
                            KART.handle(json.loads(payload))
                        except (ValueError, KeyError) as e:
                            print(f"[fake] bad command {payload!r}: {e}")
                time.sleep(0.03)
        except (BrokenPipeError, ConnectionError, OSError):
            pass
        finally:
            print("[fake] ws client left")
            self.close_connection = True


def _ws_send(sock, opcode, payload):
    n = len(payload)
    head = bytes([0x80 | opcode])
    if n < 126:
        head += bytes([n])
    elif n < 65536:
        head += bytes([126]) + struct.pack(">H", n)
    else:
        head += bytes([127]) + struct.pack(">Q", n)
    sock.sendall(head + payload)               # server frames are never masked


def _ws_parse(buf):
    """Pop one complete client frame off `buf` -> (opcode, payload) or None."""
    if len(buf) < 2:
        return None
    op, ln = buf[0] & 0x0F, buf[1] & 0x7F
    masked, i = bool(buf[1] & 0x80), 2
    if ln == 126:
        if len(buf) < 4:
            return None
        ln, i = struct.unpack(">H", buf[2:4])[0], 4
    elif ln == 127:
        if len(buf) < 10:
            return None
        ln, i = struct.unpack(">Q", buf[2:10])[0], 10
    mask = b""
    if masked:
        if len(buf) < i + 4:
            return None
        mask, i = bytes(buf[i:i + 4]), i + 4
    if len(buf) < i + ln:
        return None
    payload = bytes(buf[i:i + ln])
    if masked:
        payload = bytes(b ^ mask[k % 4] for k, b in enumerate(payload))
    del buf[:i + ln]
    return op, payload


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Fake Pi Kart server (video + telemetry)")
    p.add_argument("-H", "--host", default="127.0.0.1")
    p.add_argument("-p", "--port", type=int, default=8080)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--size", default="640x480", help="WxH of the fake camera")
    p.add_argument("--run", action="store_true", help="also launch get_cam.py against this server")
    args = p.parse_args()

    CFG.fps = args.fps
    CFG.w, CFG.h = (int(v) for v in args.size.lower().split("x"))

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[fake] serving http://{args.host}:{args.port}/video  ws://{args.host}:{args.port}/ws")

    child = None
    if args.run:
        child = subprocess.Popen([sys.executable, "get_cam.py", "-i", args.host, "-p", str(args.port)])
    try:
        while child is None or child.poll() is None:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        if child and child.poll() is None:
            child.terminate()
        srv.shutdown()


if __name__ == "__main__":
    main()
