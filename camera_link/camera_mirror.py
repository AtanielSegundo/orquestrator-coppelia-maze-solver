"""
Camera mirror -- dirt-simple version.

The pan/tilt PERCENT (0-100, the exact units the kart's own CMD_SERVO takes --
see calibration.py's servo_cmd_fmt/set_servos_angle and the web UI's pan/tilt
sliders in Server/static/index.html) IS the single source of truth. Arrow
keys change it directly, and every change is fired at the real kart as a
one-off, fire-and-forget HTTP POST to /command -- exactly calibration.py's
set_servos_angle, just inlined here. No drive WebSocket link, no calibration
LERP tables, no telemetry round-trip: nudge() doesn't wait for anything, and
neither does the caller.

CoppeliaSim's pan/tilt joints just SHADOW that same percent through a rough
linear map (PAN_DEG_RANGE / TILT_DEG_RANGE below), purely so there is
something to look at in the sim. It is not meant to be an exact optical
match -- widen/shift/flip those two ranges by hand if the sim swings too
little, too much, or the wrong way.
"""
import math
import threading

import requests

from camera_link.mjpeg_camera import MjpegCamera
from camera_link.sim_side_camera import SimSideCamera


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _lerp(frac, lo, hi):
    return lo + (hi - lo) * frac


class CameraMirror:
    # Servo channels on the kart's PWM board.
    PAN_CHANNEL  = 5
    TILT_CHANNEL = 1

    # Rough percent -> sim degrees. Swap the two numbers in a range to flip
    # which way increasing percent turns the sim joint.
    PAN_DEG_RANGE  = (-45.0, 45.0)
    TILT_DEG_RANGE = (0.0, 90.0)

    # Measured centered-camera values (Server/servo.py boots to the same).
    PAN_HOME_PCT  = 50
    TILT_HOME_PCT = 37

    def __init__(self,
                 sim,
                 pan_joint_path: str,
                 tilt_joint_path: str,
                 video_url: str,
                 command_url: str,
                 obj_path: str   = "/realCamera",
                 is_fliped: bool = True, # CoppeliaSim's bottom-up image convention
                 video: str      = "sim",
                 ):
        self.sim = sim
        self.is_fliped = is_fliped
        self.command_url = command_url

        # A bad backend name is a config typo: fail loudly here, rather than in
        # the try below where it would masquerade as "no camera in the scene".
        if video not in ("sim", "client", "off"):
            raise ValueError(f"video backend must be 'sim', 'client' or 'off', "
                             f"got {video!r}")

        try:
            self.cam_obj_id = sim.getObject(obj_path)
            self.resX, self.resY = sim.getVisionSensorRes(self.cam_obj_id)
            self._stream = self._make_backend(video, video_url, obj_path)
            print(f"[CameraMirror] mirroring {video_url} -> '{obj_path}' "
                    f"({self.resX}x{self.resY}) via the {video} backend")

        except Exception as e:
            self.cam_obj_id = self._stream = None
            self.resX = self.resY = 0
            print(f"[CameraMirror] video disabled ({e})")

        # The sim joints are optional too: without them we still command the real
        # servos, we just have no mirror to look at.
        try:
            self.pan_joint  = sim.getObject(pan_joint_path)
            self.tilt_joint = sim.getObject(tilt_joint_path)
            for joint in (self.pan_joint, self.tilt_joint):
                sim.setJointMode(joint, sim.jointmode_kinematic, 0)
        except Exception as e:
            self.pan_joint = self.tilt_joint = None
            print(f"[CameraMirror] sim joints disabled ({e})")

        self.pan_pct  = self.PAN_HOME_PCT
        self.tilt_pct = self.TILT_HOME_PCT
        self._written = None            # last (pan_pct, tilt_pct) written to the sim
        self._sent    = None            # last (pan_pct, tilt_pct) fired at the kart

    def _make_backend(self, video, video_url, obj_path):
        """Pick who pulls the MJPEG stream. "sim" keeps ~900 KB frames out of the
        remote API entirely (see SimSideCamera); "client" is the self-contained
        path that needs nothing loaded in the scene."""
        if video == "sim":
            return SimSideCamera(video_url, obj_path, flip=self.is_fliped)
        if video == "client":
            return MjpegCamera(video_url)
        return None                      # "off"

    # -- operator input (0-100 percent, same units as CMD_SERVO) ------------
    def set_percents(self, pan_pct, tilt_pct):
        self.pan_pct  = int(_clamp(round(pan_pct),  0, 100))
        self.tilt_pct = int(_clamp(round(tilt_pct), 0, 100))
        self._fire_kart()

    def nudge(self, d_pan_pct, d_tilt_pct):
        self.set_percents(self.pan_pct + d_pan_pct, self.tilt_pct + d_tilt_pct)

    def look_forward(self):
        self.set_percents(self.PAN_HOME_PCT, self.TILT_HOME_PCT)

    # -- real kart: fire-and-forget HTTP, no drive-link dependency ----------
    def _fire_kart(self):
        state = (self.pan_pct, self.tilt_pct)
        if state == self._sent:
            return
        self._sent = state
        threading.Thread(target=self._post_servo, args=state, daemon=True).start()

    def _post_servo(self, pan_pct, tilt_pct):
        body = (f"CMD_SERVO#{self.PAN_CHANNEL}#{pan_pct}\n"
                f"CMD_SERVO#{self.TILT_CHANNEL}#{tilt_pct}")
        try:
            requests.post(self.command_url, data=body, timeout=0.5)
        except requests.RequestException:
            pass          # fire and forget -- the next nudge just tries again

    # -- sim: rough shadow, no calibration -----------------------------------
    def apply(self):
        """Write the current percent into the sim joints (rough linear map).
        The real kart already got its push directly from nudge()/set_percents()."""
        if self.pan_joint is None:
            return
        state = (self.pan_pct, self.tilt_pct)
        if state == self._written:
            return
        self._written = state
        pan_deg  = _lerp(self.pan_pct  / 100.0, *self.PAN_DEG_RANGE)
        tilt_deg = _lerp(self.tilt_pct / 100.0, *self.TILT_DEG_RANGE)
        self.sim.setJointPosition(self.pan_joint,  math.radians(pan_deg))
        self.sim.setJointPosition(self.tilt_joint, math.radians(tilt_deg))

    def push_frame(self):
        """Advance the video mirror by one control-loop iteration. Returns True
        when the sensor picked up new imagery. What this costs depends on the
        backend: a frame push for "client", a throttled ping for "sim"."""
        if self._stream is None:
            return False
        return self._stream.pump(self.sim, self.cam_obj_id,
                                 self.resX, self.resY, self.is_fliped)

    def video_status(self):
        return self._stream.describe() if self._stream is not None else "off"

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream = None
