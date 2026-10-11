"""
Camera link for the Lafvin PiKart -> CoppeliaSim bridge.

Two interchangeable backends put the kart's camera into a vision sensor. Both
expose `pump(sim, sensor, res_x, res_y, flip)` / `stop()` / `describe()`, so
`CameraMirror` can hold either one:

`MjpegCamera` (client side) reads the kart's `/video` MJPEG stream (web.py's
`video_handler`) in a background thread, keeping only the LATEST decoded frame
(latest-wins, auto-reconnect), and `pump()` pushes it with
`push_frame_to_sensor()`. Simple, but every frame is a ~900 KB raw RGB payload
crossing the ZMQ remote API, which shows up as latency in the control loop.

`SimSideCamera` (sim side) instead hands the URL to `sim_video_feeder.py`, a
script running INSIDE CoppeliaSim, which pulls the stream itself and writes the
sensor in-process. The control loop is then left with a throttled status ping
instead of an image upload. Needs that script loaded in the scene (see its
docstring); opencv/numpy are then required in CoppeliaSim's interpreter, not
just here.

`CameraMirror` is the pan/tilt head orchestrator built on top of either
backend: it mirrors the kart's video feed into the sim AND the sim's pan/tilt
joint angles back to the real servos (see camera_mirror.py's own docstring).

Split out of control_websocket.py so the camera plumbing stays reusable --
control_websocket.py (gamepad teleop) and maze_solver/coppelia_main.py
(autonomous maze run) both hold a `CameraMirror`, and neither has to re-derive
this control script's own drive/telemetry logic to get a camera feed. Each
piece lives in its own module (one class/function per file); this just
re-exports them so `from camera_link import CameraMirror` keeps working.

NOTE: the "external input" image feed is a CoppeliaSim **vision sensor** feature
(tick "external input" in the vision-sensor dialog), not a proximity sensor.

Deps:  pip install opencv-python numpy
"""
from camera_link.constants import FEEDER_HOST, FEEDER_PORT
from camera_link.mjpeg_camera import MjpegCamera
from camera_link.sim_side_camera import SimSideCamera
from camera_link.push_frame_to_sensor import push_frame_to_sensor
from camera_link.camera_mirror import CameraMirror
# Camera-as-a-sensor pieces of the maze runner's pose fix: calibration, the
# ideal pinhole, the per-frame wall/floor model and the floor-contact profile.
from camera_link.camera_calibration import CameraCalibration, ViewCalibration
from camera_link.pinhole_view import PinholeView
from camera_link.servo_head import ServoHead
from camera_link.floor_segmenter import FloorSegmenter
from camera_link.contact_profile import ContactProfile, extract_contact_profile

__all__ = [
    "FEEDER_HOST", "FEEDER_PORT",
    "MjpegCamera", "SimSideCamera", "push_frame_to_sensor", "CameraMirror",
    "CameraCalibration", "ViewCalibration", "PinholeView", "ServoHead",
    "FloorSegmenter", "ContactProfile", "extract_contact_profile",
]
