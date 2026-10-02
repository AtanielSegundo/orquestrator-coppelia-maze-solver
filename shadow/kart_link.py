"""
Single WebSocket link: ONE connection, one background thread that both drains
telemetry (so the server never back-pressures) AND sends the latest command
state. Two separate connections deadlock the server's broadcaster, because the
telemetry it keeps pushing to the unread command socket fills the send buffer
and stalls the single broadcast loop (-> telemetry socket times out).

`teleop` controls whether the loop keeps re-sending the last `set_drive()`
twist every iteration (a keepalive against the server's command_timeout,
needed for continuous velocity control). Discrete moves (`turn`,
`drive_distance`, `reset_odometry`, ...) must NOT be teleop-linked: the
server's `drive` handler unconditionally cancels any in-flight move/turn (see
DriveController.set_twist), so a repeating zero-twist keepalive would cancel a
`drive_distance` the instant after it starts. Pass teleop=False for any link
that only issues discrete moves (e.g. the maze-solving runner).

Deps:  pip install websocket-client
"""
import json
import socket
import threading
import time

# Optional at import time so a Coppelia-only tool that never talks to the real
# kart doesn't need it installed. Only _run() actually uses it.
try:
    import websocket
except ImportError:                        # pragma: no cover
    websocket = None


class KartLink:
    def __init__(self, url, teleop=True):
        self.url = url
        self.teleop = teleop
        self._lock = threading.Lock()
        self._pose = None                  # (x, y, theta) in the kart odom frame
        self._wheel = (0.0, 0.0)           # (left, right) m/s
        self._drive = (0.0, 0.0)           # latest-wins twist (teleop only)
        self._servo = {}                   # channel -> latest angle
        self._oneshot = []                 # queued one-shot commands (dicts)
        self._drive_telem = {}             # full `drive` telemetry sub-dict
        self._servo_telem = {}             # channel(str) -> last-commanded angle
        self.running = True
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    # -- main-thread API: teleop -------------------------------------------
    def set_drive(self, linear, angular):
        with self._lock:
            self._drive = (linear, angular)

    def set_servo(self, channel, angle):
        with self._lock:
            self._servo[int(channel)] = int(angle)

    # -- main-thread API: discrete commands ---------------------------------
    def send_command(self, msg: dict):
        """Queue one arbitrary JSON command (dict, needs a "type" key). Sent
        exactly once, in order, from the background thread."""
        with self._lock:
            self._oneshot.append(dict(msg))

    def reset_odometry(self):
        self.send_command({"type": "reset_odometry"})

    def turn(self, angle_deg, speed=1.0):
        """Relative in-place turn (deg, +ccw) -- see DriveController.turn_in_place."""
        self.send_command({"type": "turn", "angle": angle_deg, "speed": speed})

    def drive_distance(self, distance_m, speed=0.2):
        """Straight move (m, +forward) -- see DriveController.drive_distance."""
        self.send_command({"type": "drive_distance",
                           "distance": distance_m, "speed": speed})

    def goto(self, x, y, theta_deg=None):
        """Absolute go-to-pose in the kart's ODOMETRY frame (m, deg) -- see
        DriveController.goto_pose, which runs it as turn -> drive -> turn with
        every phase recomputed from the live odom pose.

        ODOMETRY frame, not the sim world frame: the odom origin is wherever
        `reset_odometry` was last called. Converting a world target into this
        frame is SimMirror.odom_target()'s job -- passing world coordinates
        straight through would drive the kart to the wrong place entirely
        (the two frames only coincide for a kart started at the world origin
        facing +X, with no anchor correction ever applied).

        `theta_deg` None leaves the final heading wherever the drive ends.
        """
        msg = {"type": "goto", "x": float(x), "y": float(y)}
        if theta_deg is not None:
            msg["theta"] = float(theta_deg)
        self.send_command(msg)

    # -- main-thread API: reading state --------------------------------------
    def latest(self):
        """(pose, wheel_speed) -- kept for callers that only care about those."""
        with self._lock:
            return self._pose, self._wheel

    def drive_telemetry(self):
        """Full `drive` telemetry sub-dict from the last message (engaged,
        goal_active, guard_blocked, front_distance_cm, move, goto, gyro, ...).
        Empty dict until the first telemetry message arrives."""
        with self._lock:
            return dict(self._drive_telem)

    def servo_telemetry(self):
        """channel(str) -> last angle the kart's own Servo actually commanded
        (see Server/servo.py -- there's no hardware readback, so this is the
        server's own bookkeeping, not a live sensor reading). Empty dict until
        the first telemetry message arrives, or against an older server that
        doesn't publish this field yet."""
        with self._lock:
            return dict(self._servo_telem)

    def stop(self):
        self.set_drive(0.0, 0.0)
        time.sleep(0.1)                    # let the loop push the stop out
        self.running = False
        self._t.join(timeout=1.0)

    # -- background thread -------------------------------------------------
    def _run(self):
        ws = None
        servo_sent = {}
        while self.running:
            try:
                if ws is None:
                    ws = websocket.create_connection(self.url, timeout=6)
                    servo_sent = {}
                    print("[kart] connected")

                ws.settimeout(0.02)
                try:
                    raw = ws.recv()
                    if raw in ("", None):
                        raise ConnectionError("closed by peer")
                    self._on_message(raw)
                except (websocket.WebSocketTimeoutException, socket.timeout):
                    pass

                # Push the latest command state.
                with self._lock:
                    lin, ang = self._drive
                    servo = dict(self._servo)
                    one, self._oneshot = self._oneshot, []
                if self.teleop:
                    ws.send(json.dumps({"type": "drive",
                                        "linear": round(lin, 3),
                                        "angular": round(ang, 3)}))
                for ch, a in servo.items():            # send only on change
                    if servo_sent.get(ch) != a:
                        ws.send(json.dumps({"type": "servo",
                                            "channel": str(ch), "angle": a}))
                        servo_sent[ch] = a
                for m in one:
                    ws.send(json.dumps(m))

            except Exception as e:                     # noqa: BLE001
                print(f"[kart] link error ({e}); reconnecting...")
                try:
                    if ws:
                        ws.close()
                except Exception:                      # noqa: BLE001
                    pass
                ws = None
                time.sleep(1.0)
        try:
            if ws:
                ws.close()
        except Exception:                              # noqa: BLE001
            pass

    def _on_message(self, raw):
        try:
            msg = json.loads(raw)
        except (ValueError, TypeError):
            return
        if msg.get("type") != "telemetry":
            return
        d = msg.get("drive", {})
        p = d.get("pose", {})
        w = d.get("wheel_speed", {})
        s = msg.get("servo")
        with self._lock:
            self._pose = (p.get("x", 0.0), p.get("y", 0.0), p.get("theta", 0.0))
            self._wheel = (w.get("left", 0.0), w.get("right", 0.0))
            self._drive_telem = d
            if s:
                self._servo_telem = s
