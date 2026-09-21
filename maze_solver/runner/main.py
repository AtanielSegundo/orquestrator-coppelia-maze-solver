"""
Solve the maze with the REAL Lafvin PiKart, mirrored live into both
CoppeliaSim (3D) and this Raylib window (2D top-down) -- a "shadow" run.

Flow:
  1. Build/refresh the maze in CoppeliaSim and place `kart_base` at the
     solution's start cell, facing the first leg's heading.
  2. Print the placement instructions and wait for the operator to physically
     match the real kart to that cell/heading, then type `ok`.
  3. Reset the kart's odometry (so its origin lines up with the anchor pose
     `kart_base` was just placed at) and start mirroring its live telemetry
     into both views, every frame, from then on.
  4. Drive the solution one cell at a time: turn to face the next cell (only
     if the heading actually changes), then drive_distance exactly one cell,
     waiting for each move to finish before evaluating it -- arrived, or cut
     short by the front distance/collision guard -- and only then issuing the
     next command.
  5. If the operator picks a new start/goal (mouse click, or the `start` /
     `goal` prompt commands) while live, the run is replanned from wherever
     the kart actually is -- current cell -> new start -> new goal -- taking
     effect at the next cell boundary, never interrupting a leg in flight.
  6. Meanwhile the front distance sensor keeps writing into a LIVE copy of
     the maze: anything it sees closer than OBSTACLE_DISTANCE_CM is projected
     onto the grid and that cell is blocked, so the next replan routes around
     it and both views draw it. The .maze file's own grid is kept untouched
     alongside as the reference (see MazeView's original-vs-live split), and
     `resume` restores the live one from it.

Commands (`turn`, `drive_distance`, `reset_odometry`) go out over the same
KartLink used by control_websocket.py (see shadow/), but with its teleop
keepalive OFF: a repeating zero-velocity `drive` message would cancel any
in-flight move server-side (DriveController.set_twist unconditionally clears
`_move`/`_heading_target`), so this link only ever sends the discrete
commands the maze run issues, one at a time.

Deps:  pip install websocket-client pyray coppeliasim_zmqremoteapi_client
"""
import sys
import queue
import threading

import pyray as pr
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

import src.styles as styles
import src.definitions as definitions
from src.definitions import WINDOW
from src.renders     import KartView, MazeView
from src.solver      import resolve_maze

from runner.build_parser     import build_parser
from runner.cli_thread       import cli_thread
from runner.live_maze_runner import LiveMazeRunner
from runner.coppelia_evoke   import (evoke_coppelia, load_scene_in_coppelia,
                                     terminate_coppelia)

def main():
    args = build_parser().parse_args()

    maze_path = resolve_maze(args.maze)
    if maze_path is None:
        print(f"[ERROR] maze nao encontrado: {args.maze}")
        sys.exit(-1)

    coppelia_process = evoke_coppelia()
    client = RemoteAPIClient()
    
    sim = client.require("sim")
    load_scene_in_coppelia(sim)
    
    kart_base = sim.getObject("/kart_base")
    motors = [sim.getObject(f"/motor_{i}/joint") for i in range(1, 5)]
    
    for j in motors:
        sim.setJointMode(j, sim.jointmode_kinematic, 0)

    pr.set_config_flags(pr.ConfigFlags.FLAG_WINDOW_RESIZABLE)
    pr.set_trace_log_level(definitions.LOG.LEVEL)
    pr.init_window(WINDOW.WIDTH, WINDOW.HEIGHT,
                   "Maze Solver - CoppeliaSim + Lafvin Kart")
    pr.set_target_fps(60)

    view = MazeView(maze_path)
    kart_view = KartView(styles.Kart.file_path, styles.Kart.tint)

    def rescale():
        view.recompute()
        kart_view.set_scale_to_pixels(view.cell * (args.kart_lenght / args.cell_lenght))
    rescale()

    ws_url = f"ws://{args.host}:{args.port}/ws"
    video_url = f"http://{args.host}:{args.port}/video"
    command_url = f"http://{args.host}:{args.port}/command"
    runner = LiveMazeRunner(sim, kart_base, motors, maze_path, view, kart_view,
                            args.cell_lenght, ws_url,
                            args.turn_speed, args.drive_speed,
                            video_url, command_url, args.video)

    commands: "queue.Queue" = queue.Queue()
    threading.Thread(target=cli_thread, args=(commands,), daemon=True).start()

    CAM_SPEED = 60.0    # %/s of pan/tilt travel while an arrow key is held (0-100 range)
    K = pr.KeyboardKey

    running = True
    try:
        while running and not pr.window_should_close():
            while True:
                try:
                    name, cmd_args = commands.get_nowait()
                except queue.Empty:
                    break
                if name == "quit":
                    running = False
                elif name == "ok":
                    runner.confirm_ok()
                elif name == "start":
                    cell = (int(cmd_args[0]), int(cmd_args[1]))
                    if not view.in_bounds(cell) or not runner.set_start(cell):
                        print(f"[ERROR] start invalido: {cell}")
                elif name == "goal":
                    cell = (int(cmd_args[0]), int(cmd_args[1]))
                    if not view.in_bounds(cell) or not runner.set_goal(cell):
                        print(f"[ERROR] goal invalido: {cell}")
                elif name == "resume":
                    runner.resume()
                elif name == "status":
                    print(f"[INFO] {runner.status}")

            if pr.is_window_resized():
                rescale()

            if pr.is_mouse_button_pressed(pr.MOUSE_BUTTON_LEFT):
                cell = view.mouse_cell()
                if cell and not runner.set_start(cell):
                    print(f"[ERROR] {cell} e parede ou e o goal atual")
            if pr.is_mouse_button_pressed(pr.MOUSE_BUTTON_RIGHT):
                cell = view.mouse_cell()
                if cell and not runner.set_goal(cell):
                    print(f"[ERROR] {cell} e parede ou e o start atual")

            dt = pr.get_frame_time()

            # ---------- keyboard: arrow keys -> camera pan/tilt ----------
            cam_step = CAM_SPEED * dt
            d_pan = d_tilt = 0.0
            if pr.is_key_down(K.KEY_LEFT):  d_pan  += cam_step
            if pr.is_key_down(K.KEY_RIGHT): d_pan  -= cam_step
            if pr.is_key_down(K.KEY_UP):    d_tilt += cam_step
            if pr.is_key_down(K.KEY_DOWN):  d_tilt -= cam_step
            if pr.is_key_pressed(K.KEY_C):  runner.look_forward()
            if d_pan or d_tilt:
                runner.nudge(d_pan, d_tilt)

            runner.step(dt)
            runner.render()
    except KeyboardInterrupt:
        pass
    finally:
        runner.stop()
        # Closing normaly. If we are killed hard instead, the job object
        # created in coppelia_evoke takes CoppeliaSim down with us.
        terminate_coppelia(coppelia_process)
        pr.close_window()
        print("[INFO] encerrado")
