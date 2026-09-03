import math

import pyray as rl


def draw_hud(GP, v_max, linear, angular, raw_fwd, raw_turn, pose, cam):
    rl.begin_drawing()
    rl.clear_background(rl.RAYWHITE)
    if rl.is_gamepad_available(GP):
        rl.draw_text(str(rl.get_gamepad_name(GP)), 10, 10, 14, rl.DARKGRAY)
    else:
        rl.draw_text("Sem gamepad", 10, 10, 16, rl.RED)
    rl.draw_text("WASD move | arrows cam | C center | +/- speed", 10, 265, 12, rl.GRAY)
    rl.draw_text(f"V_MAX={v_max:.2f} m/s", 10, 285, 16, rl.DARKBLUE)
    rl.draw_text(f"cmd  v={linear:+.2f}  w={angular:+.2f}", 10, 305, 16, rl.DARKGRAY)
    if pose is not None:
        rl.draw_text(f"kart x={pose[0]:+.2f} y={pose[1]:+.2f} th={math.degrees(pose[2]):+.0f}",
                     10, 325, 16, rl.DARKGREEN)
    else:
        rl.draw_text("kart: sem telemetria", 10, 325, 16, rl.MAROON)
    rl.draw_text(f"cam servo pan={cam.pan_pct}% tilt={cam.tilt_pct}%",
                 10, 345, 14, rl.GRAY)
    rl.draw_text(f"video: {cam.video_status()}", 250, 10, 14, rl.DARKGRAY)
    cx, cy, R = 180, 180, 80
    rl.draw_circle_lines(cx, cy, R, rl.LIGHTGRAY)
    rl.draw_circle(int(cx + raw_turn * R), int(cy + raw_fwd * R), 10, rl.MAROON)
    rl.end_drawing()
