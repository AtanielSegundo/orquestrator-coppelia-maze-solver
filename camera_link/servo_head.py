"""
Pan/tilt head of the real kart, with no simulator attached.

The same fire-and-forget CMD_SERVO POST CameraMirror sends, for the tools that
drive the head without a CoppeliaSim scene (camera_calibrate.py). It exposes the
`set_percents` / `pan_pct` / `tilt_pct` surface CameraMirror has, so the camera
pose fix accepts either one.
"""
import threading

import requests


class ServoHead:
    PAN_CHANNEL = 5
    TILT_CHANNEL = 1

    def __init__(self, command_url: str, pan_pct: int = 50, tilt_pct: int = 37):
        self.command_url = command_url
        self.pan_pct = int(pan_pct)
        self.tilt_pct = int(tilt_pct)

    def set_percents(self, pan_pct, tilt_pct):
        self.pan_pct = int(max(0, min(100, round(pan_pct))))
        self.tilt_pct = int(max(0, min(100, round(tilt_pct))))
        threading.Thread(target=self._post, args=(self.pan_pct, self.tilt_pct),
                         daemon=True).start()

    def _post(self, pan_pct, tilt_pct):
        body = (f"CMD_SERVO#{self.PAN_CHANNEL}#{pan_pct}\n"
                f"CMD_SERVO#{self.TILT_CHANNEL}#{tilt_pct}")
        try:
            requests.post(self.command_url, data=body, timeout=0.5)
        except requests.RequestException:
            pass
