"""
Drive the REAL Lafvin PiKart over WebSocket from a gamepad, and mirror the
kart's odometry telemetry into CoppeliaSim.

Pipeline:
    gamepad  --(twist)-->  WS command  -->  real kart drives
    real kart  --(telemetry pose)-->  WS  -->  setObjectMatrix(kart_base)
    real kart  --(MJPEG /video)-->  decode JPEG  -->  setVisionSensorImg(camera)

The kart's odometry starts at (0,0,0); the sim's kart_base has whatever pose the
scene placed it at. We capture that INITIAL sim pose once and anchor the kart's
odometry to it, so the sim mirror starts exactly where the sim kart already is
and moves relative to that frame. That is the requested initial-pose sync.

The kart's camera feed is mirrored into a CoppeliaSim vision sensor flagged as
"external input" (see camera_link/). It is optional: if the sensor is absent
from the scene the camera mirror is skipped and drive/telemetry still run.

Deps:  pip install websocket-client pyray coppeliasim_zmqremoteapi_client \
                    opencv-python numpy
"""
import pyray as rl
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

import Coppelia.helpers as cop
from Coppelia.drive import TiredWheel
from camera_link import CameraMirror
from shadow import KartLink, SimMirror

from teleop.constants import WS_URL, VIDEO_URL, VIDEO_BACKEND, COMMAND_URL
from teleop.apply_deadzone import apply_deadzone
from teleop.joystick_to_twist import joystick_to_twist
from teleop.draw_hud import draw_hud


def main():
    client = RemoteAPIClient()
    sim = client.require("sim")

    kart_base = sim.getObject("/kart_base")
    motors = [sim.getObject(f"/motor_{i}/joint") for i in range(1, 5)]
    for j in motors:
        sim.setJointMode(j, sim.jointmode_kinematic, 0)

    camera_mirror = CameraMirror(sim, "/cam_base/joint", "/servos_link_struct/joint",
                                 video_url=VIDEO_URL, command_url=COMMAND_URL,
                                 video=VIDEO_BACKEND)

    # cop.change_floor(sim, (10.0, 10.0), thickness=0.25)
    MAZE = "saecollab/mazes/medium_eg.maze"
    cop.remove_object(sim, cop.get_maze_tag(MAZE))
    # cop.render_maze(sim, MAZE, center=(0, 0, 0), wall_color=CAIXA_COR)

    mirror = SimMirror(sim, kart_base, motors, TiredWheel(),
                       center_offset_local=cop.KART_CENTER_OFFSET_LOCAL)

    link = KartLink(WS_URL)
    link.reset_odometry()       # zero the kart's odom so it lines up with anchor
    odom_ref = None             # first telemetry pose after reset = our baseline

    CAM_SPEED = 60.0            # %/s of pan/tilt travel while held (0-100 range)

    # --- gamepad / limits ---
    rl.init_window(360, 360, "Kart WS bridge")
    rl.set_target_fps(60)
    GP = 0
    V_MAX, V_MIN, V_CEIL, V_STEP = 0.35, 0.05, 1.00, 0.05     # m/s (kart max 0.6)
    W_MAX = 4.02                                              # rad/s (kart max 4.0)
    DT = 1.0 / 60

    AXIS_TURN     = rl.GamepadAxis.GAMEPAD_AXIS_LEFT_X
    AXIS_FWD      = rl.GamepadAxis.GAMEPAD_AXIS_LEFT_Y
    AXIS_CAM_PAN  = rl.GamepadAxis.GAMEPAD_AXIS_RIGHT_X
    AXIS_CAM_TILT = rl.GamepadAxis.GAMEPAD_AXIS_RIGHT_Y
    BTN_UP        = rl.GamepadButton.GAMEPAD_BUTTON_RIGHT_TRIGGER_1
    BTN_DOWN      = rl.GamepadButton.GAMEPAD_BUTTON_LEFT_TRIGGER_1
    K             = rl.KeyboardKey                                   # WASD drive, arrows = camera, +/- = speed

    raw_fwd = raw_turn = 0.0
    linear = angular = 0.0

    try:
        while not rl.window_should_close():
            gp_lin = gp_ang = 0.0
            raw_fwd = raw_turn = 0.0

            # ---------- gamepad (optional) ----------
            if rl.is_gamepad_available(GP):
                if rl.is_gamepad_button_pressed(GP, BTN_UP):
                    V_MAX = min(V_CEIL, V_MAX + V_STEP)
                if rl.is_gamepad_button_pressed(GP, BTN_DOWN):
                    V_MAX = max(V_MIN, V_MAX - V_STEP)

                raw_fwd  = rl.get_gamepad_axis_movement(GP, AXIS_FWD)
                raw_turn = -rl.get_gamepad_axis_movement(GP, AXIS_TURN)
                gp_lin, gp_ang  = joystick_to_twist(apply_deadzone(raw_fwd),
                                                   apply_deadzone(raw_turn),
                                                   V_MAX, W_MAX)
                # right stick -> camera (stick left/up = pan left / tilt up,
                # same directions as the arrow keys below)
                cx = apply_deadzone(-rl.get_gamepad_axis_movement(GP, AXIS_CAM_PAN))
                cy = apply_deadzone(-rl.get_gamepad_axis_movement(GP, AXIS_CAM_TILT))
                camera_mirror.nudge(cx * CAM_SPEED * DT, cy * CAM_SPEED * DT)

            # ---------- keyboard: W/A/S/D drive ----------
            kb_lin = kb_ang = 0.0
            if rl.is_key_down(K.KEY_W): kb_lin += V_MAX     # forward
            if rl.is_key_down(K.KEY_S): kb_lin -= V_MAX     # reverse
            if rl.is_key_down(K.KEY_A): kb_ang += W_MAX     # turn left  (+ccw)
            if rl.is_key_down(K.KEY_D): kb_ang -= W_MAX     # turn right (-cw)
            if rl.is_key_pressed(K.KEY_EQUAL): V_MAX = min(V_CEIL, V_MAX + V_STEP)
            if rl.is_key_pressed(K.KEY_MINUS): V_MAX = max(V_MIN, V_MAX - V_STEP)

            # ---------- keyboard: arrow keys -> camera servos ----------
            cam_step = CAM_SPEED * DT
            d_pan = d_tilt = 0.0
            if rl.is_key_down(K.KEY_LEFT):  d_pan  += cam_step
            if rl.is_key_down(K.KEY_RIGHT): d_pan  -= cam_step
            if rl.is_key_down(K.KEY_UP):    d_tilt += cam_step
            if rl.is_key_down(K.KEY_DOWN):  d_tilt -= cam_step
            if rl.is_key_pressed(K.KEY_C):  camera_mirror.look_forward()
            if d_pan or d_tilt:
                camera_mirror.nudge(d_pan, d_tilt)

            # keyboard drive overrides the gamepad while a WASD key is held
            if kb_lin != 0.0 or kb_ang != 0.0:
                linear, angular = kb_lin, kb_ang
            else:
                linear, angular = gp_lin, gp_ang

            # shadow the current pan/tilt percent into the sim joints (the real
            # servos already got their fire-and-forget push from nudge() above)
            camera_mirror.apply()

            # Latest-wins; the link thread sends it (keepalive at telemetry rate,
            # so the kart never hits its 0.1 s command timeout).
            link.set_drive(linear, angular)

            # Mirror the real kart pose into the sim, anchored to the sim's
            # initial kart pose.
            pose, wheel = link.latest()
            if pose is not None:
                if odom_ref is None:
                    odom_ref = pose
                mirror.mirror(pose, odom_ref)
                mirror.spin_wheels(wheel[0], wheel[1], DT)

            # Mirror the real camera feed into the sim vision sensor (only when
            # a fresh frame arrived, to avoid re-pushing the same image).
            camera_mirror.push_frame()

            draw_hud(GP, V_MAX, linear, angular, raw_fwd, raw_turn, pose,
                    camera_mirror)

    except KeyboardInterrupt:
        pass
    finally:
        camera_mirror.stop()
        link.stop()
        rl.close_window()
        print("parado")
