from coppeliasim_zmqremoteapi_client import RemoteAPIClient

import math
import time
import pyray as rl          
import Coppelia.helpers as cop
from Coppelia.drive import DiffDrive,TiredWheel

rgb_to_norm = lambda _tuple : tuple(map(lambda e: e/ 255,_tuple))
CAIXA_COR = rgb_to_norm((178, 131, 97))

def clamp_to_arena(robot, half=0.95):
    robot.x = max(-half, min(half, robot.x))
    robot.y = max(-half, min(half, robot.y))
    robot.set_pose()

def apply_deadzone(value, dz=0.12):
    if abs(value) < dz:
        return 0.0
    return (value - math.copysign(dz, value)) / (1 - dz)

def joystick_to_wheels(axis_fwd, axis_turn, v_max=0.5, w_max=None):
    """Y -> avanco/re (mesma vel. nas duas rodas).
       X -> turn-in-place puro (rodas opostas), sem avanco."""
    if w_max is None:
        w_max = v_max
    if abs(axis_turn) >= abs(axis_fwd):
        w = axis_turn * w_max          
        return -w, +w                  
    else:
        v = -axis_fwd * v_max          
        return v, v
    
def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def update_camera(raw_x, raw_y, dt):
    ax = apply_deadzone(raw_x)
    ay = apply_deadzone(raw_y)
    cam["pan"]  = _clamp(cam["pan"]  + ax * CAM_SPEED * dt, -CAM_LIMIT, CAM_LIMIT)
    cam["tilt"] = _clamp(cam["tilt"] - ay * CAM_SPEED * dt, -CAM_LIMIT, CAM_LIMIT)  # -ay: empurrar p/ cima inclina p/ cima
    sim.setJointPosition(cam_pan,  cam["pan"])
    sim.setJointPosition(cam_tilt, cam["tilt"])

if __name__ == "__main__":
    client = RemoteAPIClient()
    sim: cop.CoppeliaSimulation = client.require('sim')
    kart_base     = sim.getObject("/kart_base")
    motor_numbers = 4
    motors_joints = [sim.getObject(f"/motor_{id}/joint")
                     for id in range(1, motor_numbers+1)]
    for joint in motors_joints:
        sim.setJointMode(joint, sim.jointmode_kinematic, 0)

    cam_pan  = sim.getObject("/cam_base/joint")            # rotação em z (pan)
    cam_tilt = sim.getObject("/servos_link_struct/joint")  # rotação em x (tilt)
    for j in (cam_pan, cam_tilt):
        sim.setJointMode(j, sim.jointmode_kinematic, 0)

    CAM_LIMIT = math.radians(90)    # +-90 graus de curso
    CAM_SPEED = math.radians(90)    # velocidade maxima do giro (graus/s)

    AXIS_CAM_PAN  = rl.GamepadAxis.GAMEPAD_AXIS_RIGHT_X
    AXIS_CAM_TILT = rl.GamepadAxis.GAMEPAD_AXIS_RIGHT_Y

    cam = {
        "pan":  sim.getJointPosition(cam_pan),
        "tilt": sim.getJointPosition(cam_tilt),
    }

    robot = DiffDrive(sim, kart_base, motors_joints, TiredWheel())

    rl.init_window(360, 360, "DiffDrive joystick")
    rl.set_target_fps(60)

    GP        = 0          
    V_MAX     = 0.5
    V_MIN     = 0.05       
    V_CEIL    = 2.0        
    V_STEP    = 0.05    
    DT        = 1.0 / 60

    AXIS_TURN = rl.GamepadAxis.GAMEPAD_AXIS_LEFT_X
    AXIS_FWD  = rl.GamepadAxis.GAMEPAD_AXIS_LEFT_Y

    BTN_UP    = rl.GamepadButton.GAMEPAD_BUTTON_RIGHT_TRIGGER_1   # R1
    BTN_DOWN  = rl.GamepadButton.GAMEPAD_BUTTON_LEFT_TRIGGER_1    # L1

    cop.change_floor(sim, (10.0, 10.0), thickness=0.25)
    MAZE = "saecollab/mazes/medium_eg.maze"
    cop.remove_object(sim,cop.get_maze_tag(MAZE))
    cop.render_maze(sim,MAZE,
                    center=(0,0,0),wall_color=CAIXA_COR)

    try:
        while not rl.window_should_close():
            v_left = v_right = 0.0
            raw_fwd = raw_turn = 0.0

            if rl.is_gamepad_available(GP):

                # R1 aumenta, L1 diminui (uma vez por toque)
                if rl.is_gamepad_button_pressed(GP, BTN_UP):
                    V_MAX = min(V_CEIL, V_MAX + V_STEP)
                if rl.is_gamepad_button_pressed(GP, BTN_DOWN):
                    V_MAX = max(V_MIN, V_MAX - V_STEP)
                raw_cam_x = -rl.get_gamepad_axis_movement(GP, AXIS_CAM_PAN)
                raw_cam_y = -rl.get_gamepad_axis_movement(GP, AXIS_CAM_TILT)

                update_camera(raw_cam_x, raw_cam_y, DT)

                raw_fwd  = rl.get_gamepad_axis_movement(GP, AXIS_FWD)
                raw_turn = -rl.get_gamepad_axis_movement(GP, AXIS_TURN)

                fwd  = apply_deadzone(raw_fwd)
                turn = apply_deadzone(raw_turn)
                v_left, v_right = joystick_to_wheels(fwd, turn, V_MAX)

                robot.drive(v_left, v_right, DT)

            # HUD simples mostrando o estado
            rl.begin_drawing()
            rl.clear_background(rl.RAYWHITE)
            if rl.is_gamepad_available(GP):
                name = rl.get_gamepad_name(GP)
                rl.draw_text(str(name), 10, 10, 14, rl.DARKGRAY)
                rl.draw_text(f"V_MAX={V_MAX:.2f} m/s", 10, 305, 16, rl.DARKBLUE)
                rl.draw_text(f"vL={v_left:+.2f}  vR={v_right:+.2f}", 10, 330, 16, rl.DARKGRAY)
                rl.draw_text(f"cam pan={math.degrees(cam['pan']):+.0f}  tilt={math.degrees(cam['tilt']):+.0f}",
                             10, 355, 16, rl.DARKGREEN)
            else:
                rl.draw_text("Sem gamepad", 10, 10, 16, rl.RED)

            # desenha o circulo do stick
            cx, cy, R = 180, 200, 80
            rl.draw_circle_lines(cx, cy, R, rl.LIGHTGRAY)
            px = int(cx + raw_turn * R)
            py = int(cy + raw_fwd  * R)
            rl.draw_circle(px, py, 10, rl.MAROON)

            rl.draw_text(f"vL={v_left:+.2f}  vR={v_right:+.2f}", 10, 330, 16, rl.DARKGRAY)
            rl.end_drawing()

    except KeyboardInterrupt:
        pass
    finally:
        rl.close_window()
        print("parado")