from typing           import List, Optional, Tuple
from dataclasses      import dataclass
from shadow.kart_link import KartLink

import json
import math
import os
import sys
import subprocess
import pyray as pr
import re
import argparse
import numpy as np
import threading
import urllib.request
import time
import cv2

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

_HERE        = os.path.dirname(os.path.abspath(__file__))   # orquestrator
_MAZE_SOLVER = os.path.join(_HERE, "maze_solver")
if _MAZE_SOLVER not in sys.path:
    sys.path.insert(0, _MAZE_SOLVER)

from runner.coppelia_evoke import _is_coppelia_runing

@dataclass(frozen=True)
class DEFAULTS:
    PIKART_LINK_IP   : str = "pikart.local"     # IPV4 or MDNS URL
    PIKART_LINK_PORT : str = 8080
    OUTPUT_FOLDER    : str = "get_cam_out"
    WANING_LOG_LEVEL : int = 0x4
    TABULAR_IDENT_INC: int = 4
    COPPELIA_OBJECT  : str = "/camera"          # alias of the sim camera to read
    # Render size of the sim view. Same 16:9 as the window, so scaling it to
    # the saved W x H is a pure resize. 1920x1080 straight from the sensor was
    # measured at ~260 ms per frame over ZMQ; this is ~30 ms.
    COPPELIA_RES     : Tuple[int, int] = (960, 540)


def get_ipv4(ipOrMdnsUrl: str) -> Optional[str]:
    MATCH_REGEX_STRING = r"\[((?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?))\]"
    
    _ipOrMdnsUrl = ipOrMdnsUrl.strip()
    if all((char.isdigit() or char == ".") for char in _ipOrMdnsUrl) and _ipOrMdnsUrl.count(".") == 3:
        return _ipOrMdnsUrl

    s = subprocess.run(["ping", _ipOrMdnsUrl, "-4", "-n", "1"], capture_output=True, text=True)
    if s.returncode != 0:
        print(f"[ERROR] Cant Resolve {_ipOrMdnsUrl} IP")
        return None
        
    _match = re.search(MATCH_REGEX_STRING, s.stdout)
    
    if _match:
        return _match.group(1)
        
    print(f"[ERROR] Could not parse IP from ping output: {s.stdout}")
    
    return None

class VideoLink:
    def __init__(self, url):
        self.url     = url
        self._lock   = threading.Lock()
        self._frame  = None
        self._count  = 0                 # bumped on every new frame
        self.running = True
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        
    def get_frame(self,res_x,res_y):
        """Latest frame as RGB resized to (res_x, res_y), or None before the first frame."""
        with self._lock:
            frame = self._frame
        if frame is None:
            return None
        img = cv2.resize(frame, (res_x, res_y))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
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
                    chunk = resp.read(8192) # 8 Kb
                    
                    if not chunk:
                        raise ConnectionError("stream closed by peer")
                    
                    buf += chunk

                    while True:
                        a = buf.find(b"\xff\xd8")                            # SOI
                        b = buf.find(b"\xff\xd9", a + 2) if a != -1 else -1  # EOI
                        
                        if a == -1 or b == -1:
                            break
                        
                        jpg = buf[a:b + 2]
                        buf = buf[b + 2:]
                        img = cv2.imdecode(np.frombuffer(jpg, np.uint8),cv2.IMREAD_COLOR)
                        
                        if img is not None:
                            with self._lock:
                                self._frame = img
                                self._count += 1

                    # Guard against unbounded growth if we somehow get garbage.
                    if len(buf) > 4_000_000:
                        buf = buf[-1_000_000:]

            except Exception as e:                                # noqa: BLE001
                print(f"[cam] stream error ({e}); reconnecting...")
                time.sleep(1.0)
                
            finally:
                
                try:
                    if resp:
                        resp.close()
                
                except Exception:                                 # noqa: BLE001
                    pass

class CoppeliaLink:
    """What the CoppeliaSim object aliased `camera` sees, pulled over the ZMQ
    remote API on a worker thread -- the sim-side twin of VideoLink, with the
    same get_frame/describe/stop surface.

    In kart_shadow.ttt `/camera` is a camera object, not a vision sensor, and
    only vision sensors can be read (sim.getVisionSensorImg). So unless the
    object is -- or has under it -- a vision sensor, one is created as its
    child with the camera's own pose, view angle and clipping planes, and is
    removed again on stop(). It is explicitly handled, so it renders even with
    the simulation stopped.

    The ZMQ client is not thread-safe, so the worker thread owns it outright:
    it is created, used and torn down there and nowhere else.
    """
    SENSOR_ALIAS = "get_cam_sensor"
    RETRY_S      = 1.0

    def __init__(self, obj_path: str, res: Tuple[int, int], fps: float = 30.0):
        self.obj_path = obj_path
        self.res      = res
        self.period   = 1.0 / fps
        self._lock    = threading.Lock()
        self._frame   = None             # RGB, top-down, sensor resolution
        self._error   = None
        self.running  = True
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def get_frame(self, res_x, res_y):
        """Latest frame as RGB resized to (res_x, res_y), or None before the first frame."""
        with self._lock:
            frame = self._frame
        if frame is None:
            return None
        return cv2.resize(frame, (res_x, res_y))

    def describe(self):
        if self._frame is not None:
            return "coppelia"
        return f"coppelia (no frames{': ' + self._error if self._error else ''})"

    def stop(self):
        # The worker removes the sensor it created on its way out, so wait
        # for it -- returning early would leave the sensor in the scene.
        self.running = False
        self._t.join(timeout=2.0)

    def _resolve_sensor(self, sim) -> Tuple[int, bool]:
        """-> (vision sensor handle, created_here)."""
        obj = sim.getObject(self.obj_path)
        if sim.getObjectType(obj) == sim.sceneobject_visionsensor:
            return obj, False

        # Leftover from a run killed before it could clean up after itself.
        for h in sim.getObjectsInTree(obj, sim.sceneobject_visionsensor, 1):
            if sim.getObjectAlias(h) == self.SENSOR_ALIAS:
                sim.removeObjects([h])

        found = sim.getObjectsInTree(obj, sim.sceneobject_visionsensor, 1)
        if found:
            return found[0], False

        if sim.getObjectType(obj) == sim.sceneobject_camera:
            angle = sim.getObjectFloatParam(obj, sim.camerafloatparam_perspective_angle)
            near  = sim.getObjectFloatParam(obj, sim.camerafloatparam_near_clipping)
            far   = sim.getObjectFloatParam(obj, sim.camerafloatparam_far_clipping)
        else:
            angle, near, far = math.radians(60.0), 0.01, 10.0

        res_x, res_y = self.res
        # options: 1 explicit handling | 2 perspective | 4 hidden in the scene
        sensor = sim.createVisionSensor(1 | 2 | 4, [res_x, res_y, 0, 0],
                                        [near, far, angle, 0.01,
                                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        sim.setObjectAlias(sensor, self.SENSOR_ALIAS)
        sim.setObjectParent(sensor, obj, False)
        # Both camera and vision sensor look down their local +Z, so the
        # identity pose relative to the camera IS the camera's view.
        sim.setObjectPose(sensor, [0, 0, 0, 0, 0, 0, 1], obj)
        return sensor, True

    def _set_error(self, msg):
        # Edge-triggered: a closed scene would otherwise print every RETRY_S.
        if msg != self._error:
            print(f"[coppelia] {msg}")
        self._error = msg

    def _run(self):
        while self.running:
            sim    = None
            sensor = None
            owned  = False
            try:
                sim = RemoteAPIClient().require("sim")
                sensor, owned = self._resolve_sensor(sim)
                print(f"[coppelia] '{self.obj_path}' -> "
                      f"{sim.getObjectAlias(sensor, 2)}"
                      f"{' (created)' if owned else ''}")
                self._error = None
                # A sensor found in the scene may be handled by the running
                # simulation instead; it then refuses explicit handling, and
                # is just read.
                handle = True

                while self.running:
                    t0 = time.monotonic()
                    if handle:
                        try:
                            sim.handleVisionSensor(sensor)
                        except Exception:                         # noqa: BLE001
                            if owned:
                                raise
                            handle = False

                    img, (res_x, res_y) = sim.getVisionSensorImg(sensor)
                    frame = np.frombuffer(img, np.uint8).reshape(res_y, res_x, 3)
                    frame = cv2.flip(frame, 0)    # sensor images are bottom-up
                    with self._lock:
                        self._frame = frame

                    time.sleep(max(0.0, self.period - (time.monotonic() - t0)))

            except Exception as e:                                # noqa: BLE001
                # Scene reloaded (handles gone), object missing, ... -- start
                # over from a fresh client and a fresh lookup.
                with self._lock:
                    self._frame = None
                self._set_error(f"{type(e).__name__}: {e}")
                time.sleep(self.RETRY_S)

            finally:
                if owned:
                    try:
                        sim.removeObjects([sensor])
                    except Exception:                             # noqa: BLE001
                        pass

def split_panels(n: int, screen_w: int, screen_h: int, aspect: float) -> List:
    """`n` side-by-side rectangles sharing the window: each gets 1/n of the
    width, and the height that width allows at `aspect` (w/h), centred."""
    slot_w = screen_w / n
    w = min(slot_w, screen_h * aspect)
    h = w / aspect
    y = (screen_h - h) / 2
    return [pr.Rectangle(i * slot_w + (slot_w - w) / 2, y, w, h) for i in range(n)]

def blank_texture(w: int, h: int):
    blank   = pr.gen_image_color(w, h, pr.BLACK)
    texture = pr.load_texture_from_image(blank)
    pr.unload_image(blank)
    return texture

def upload_frame(texture, frame):
    rgba = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
    pr.update_texture(texture, pr.ffi.cast("void *", pr.ffi.from_buffer(rgba)))

def get_parsed_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="get_cam.py",
        description="Script to get Pi Kart Camera and Telemetry"
    )
    
    p.add_argument("-i","--ip",
                   default=DEFAULTS.PIKART_LINK_IP,
                   help="ipv4 or mdns link"
                   )
    
    p.add_argument("-p","--port",
                    default=DEFAULTS.PIKART_LINK_PORT,
                    help="port"
                    )
    
    p.add_argument("-o","--output_folder",
                   default=DEFAULTS.OUTPUT_FOLDER,
                   help="folder output image + telemetry data"
                   )
    
    return p.parse_args()

def tabular_json_str_like_toml(j:dict,string=None,ident=0) -> str:
    string = "" if (string is None) else string
    for k,v in j.items():
        string += (ident * " ") + str(k) + ":\n"
        if not isinstance(v,dict):
            string += ((ident+DEFAULTS.TABULAR_IDENT_INC) * " ") + str(v) + "\n"
        else:
            string = tabular_json_str_like_toml(v,
                                                string=string,
                                                ident=ident+DEFAULTS.TABULAR_IDENT_INC)
    return string
    
if __name__ == "__main__":
    args = get_parsed_args()
    
    pikart_ipv4 = get_ipv4(args.ip)
    
    WS_URL    = f"ws://{args.ip}:{args.port}/ws"
    VIDEO_URL = f"http://{args.ip}:{args.port}/video"    
    
    if not pikart_ipv4:
        exit(-1)
    
    VideoHandler     = VideoLink(VIDEO_URL)
    # teleop=False is NOT optional here. With the default (True), KartLink's
    # background loop re-sends a `drive` twist on every iteration as a
    # keepalive -- and the server's drive handler unconditionally cancels any
    # in-flight move (DriveController.set_twist clears _move / _heading_target
    # / _goto). This script only READS telemetry, so a teleop link silently
    # cancels every drive_distance and turn the maze runner issues, dozens of
    # times a second, for as long as this window is open. That is why the kart
    # stops obeying commands when this script is running alongside it.
    # See shadow/kart_link.py's module docstring.
    TelemetryHandler = KartLink(WS_URL, teleop=False)

    # Only attach to a CoppeliaSim that is already open -- this script never
    # launches one. When it is, the sim view shares the window with the real one.
    SimHandler = None
    if _is_coppelia_runing() is not None:
        SimHandler = CoppeliaLink(DEFAULTS.COPPELIA_OBJECT, DEFAULTS.COPPELIA_RES)
    else:
        print("[INFO] CoppeliaSim not open -- real camera only")

    pr.set_config_flags(pr.ConfigFlags.FLAG_WINDOW_RESIZABLE)
    pr.set_trace_log_level(DEFAULTS.WANING_LOG_LEVEL)
    pr.set_target_fps(30)
    pr.init_window(1920, 1080, "GetCamData")

    # Frame size: what is uploaded AND what is saved, for both sources. Only
    # the on-screen panels shrink to share the window.
    W, H = 1920, 1080
    _texture     = blank_texture(W, H)
    _sim_texture = blank_texture(W, H) if SimHandler else None

    os.makedirs(args.output_folder,exist_ok=True)

    saved_counter = 0

    try:
        while not pr.window_should_close():
            frame     = VideoHandler.get_frame(W, H)
            sim_frame = SimHandler.get_frame(W, H) if SimHandler else None
            data      = TelemetryHandler.drive_telemetry()

            if frame is not None:
                upload_frame(_texture, frame)
            if sim_frame is not None:
                upload_frame(_sim_texture, sim_frame)

            pr.begin_drawing()

            pr.clear_background(pr.BLACK)

            views = [(_texture, VideoHandler)]
            if SimHandler:
                views.append((_sim_texture, SimHandler))
            panels = split_panels(len(views), pr.get_screen_width(),
                                  pr.get_screen_height(), W / H)

            for (texture, source), panel in zip(views, panels):
                pr.draw_texture_pro(
                    texture,
                    pr.Rectangle(0, 0, W, H),
                    panel,
                    pr.Vector2(0, 0), 0.0, pr.WHITE,
                )
                if len(views) > 1:
                    pr.draw_text(source.describe(), int(panel.x) + 8,
                                 int(panel.y) + 8, 20, pr.YELLOW)

            _data_str = tabular_json_str_like_toml(data)

            pr.draw_text(_data_str, pr.get_screen_width() // 4, 0, 26, pr.WHITE)

            if pr.is_key_pressed(pr.KeyboardKey.KEY_ENTER) and frame is not None:
                data_name      = os.path.join(args.output_folder,f"data_{saved_counter  + 1}.json")
                image_name     = os.path.join(args.output_folder,f"image_{saved_counter + 1}.png")
                sim_image_name = os.path.join(args.output_folder,f"sim_image_{saved_counter + 1}.png")
                if not cv2.imwrite(image_name,cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
                    print("[ERROR] CANT SAVE FRAME")
                else:
                    if sim_frame is not None and not cv2.imwrite(
                            sim_image_name, cv2.cvtColor(sim_frame, cv2.COLOR_RGB2BGR)):
                        print("[ERROR] CANT SAVE SIM FRAME")
                    with open(data_name,"w") as fp:
                        json.dump(data,fp,indent=4)

                    saved_counter += 1

            pr.end_drawing()

    except KeyboardInterrupt:
        pass
    finally:
        # Takes the vision sensor CoppeliaLink created back out of the scene.
        if SimHandler:
            SimHandler.stop()
        pr.close_window()