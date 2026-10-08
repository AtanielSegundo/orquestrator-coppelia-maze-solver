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
     the maze, both ways: anything it sees closer than OBSTACLE_DISTANCE_CM is
     projected onto the grid and that cell is blocked, while the cells the beam
     passed through to get there are un-blocked again. So an obstacle that gets
     taken away is noticed, and a run with no route left re-plans itself and
     carries on -- no `resume` needed. The .maze file's own grid is kept
     untouched alongside as the reference (see MazeView's original-vs-live
     split), and `resume` restores the live one from it wholesale.

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

# The user needs to press this key at each cell reached by the planing
WAIT_TO_STEP_KEY = pr.KeyboardKey.KEY_ENTER

def main():
    args = build_parser().parse_args()
    
    maze_path = resolve_maze(args.maze)
    if maze_path is None:
        print(f"[ERROR] maze nao encontrado: {args.maze}")
        sys.exit(-1)

    _should_adopt_coppelia_process = not args.coppelia_detach   
    coppelia_process = evoke_coppelia(_should_adopt_coppelia_process)

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

    ws_url      = f"ws://{args.host}:{args.port}/ws"
    video_url   = f"http://{args.host}:{args.port}/video"
    command_url = f"http://{args.host}:{args.port}/command"
    runner = LiveMazeRunner(sim, kart_base, motors, maze_path, view, kart_view,
                            args.cell_lenght, ws_url,
                            args.turn_speed, args.drive_speed,
                            video_url, command_url, args.video,
                            wait_to_step=args.wait_to_step
                            )

    commands: "queue.Queue" = queue.Queue()
    threading.Thread(target=cli_thread, args=(commands,), daemon=True).start()

    CAM_SPEED = 60.0    # %/s of pan/tilt travel while an arrow key is held (0-100 range)
    K = pr.KeyboardKey
    #! IF PRESSED ALL KEYS PRESSED WHILE GUI IS FOCUSED AND UNTIL "\n"
    #! IS CAPTURED AND SENDED TO commands
    TRACK_GUI_KEYBOARD_SYMBOL = ":"
    TRACK_GUI_KEYBOARD_CODE   = ord(TRACK_GUI_KEYBOARD_SYMBOL)
    track_gui_keys_mode = False
    tracked_keys_str    = ""
    
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
            
            # Step gate.
            if (runner.should_wait and not track_gui_keys_mode
                    and pr.is_key_pressed(WAIT_TO_STEP_KEY)):
                runner.release_step()

            if track_gui_keys_mode:
                # control keys (key queue)
                while (_code := pr.get_key_pressed()) != 0:
                    if _code in (K.KEY_ENTER, K.KEY_KP_ENTER):
                        _split = tracked_keys_str.split()
                        if _split:
                            commands.put((_split[0].lower(), _split[1:]))
                        track_gui_keys_mode = False
                        tracked_keys_str = ""
                    
                    elif _code == K.KEY_BACKSPACE:
                        tracked_keys_str = tracked_keys_str[:-1]
                    
                    elif _code == K.KEY_ESCAPE:
                        track_gui_keys_mode = False
                        tracked_keys_str = ""
                        
                # text (char queue: respects shift/layout)
                while (_cp := pr.get_char_pressed()) != 0:
                    if track_gui_keys_mode:
                        tracked_keys_str += chr(_cp)
            else:
                # The key queue is only consumed in command mode; drain it here
                # so presses made outside it (ENTER for the step gate, arrows)
                # are not replayed as a stray submit the moment ':' is typed.
                while pr.get_key_pressed() != 0:
                    pass
                while (_cp := pr.get_char_pressed()) != 0:
                    if chr(_cp) == TRACK_GUI_KEYBOARD_SYMBOL:
                        track_gui_keys_mode = True
                
            if pr.is_key_down(K.KEY_LEFT):  
                d_pan  += cam_step
            
            if pr.is_key_down(K.KEY_RIGHT): 
                d_pan  -= cam_step
            
            if pr.is_key_down(K.KEY_UP):    
                if pr.is_key_down(K.KEY_LEFT_CONTROL):
                    runner.look_forward()
                else:
                    d_tilt += cam_step
            
            if pr.is_key_down(K.KEY_DOWN):  
                d_tilt -= cam_step
                
            if d_pan or d_tilt:
                runner.nudge(d_pan, d_tilt)

            runner.step(dt)
            runner.render(track_gui_keys_mode,tracked_keys_str)
            
    except KeyboardInterrupt:
        pass
    finally:
        runner.stop()
        # Closing normaly. If we are killed hard instead, the job object
        # created in coppelia_evoke takes CoppeliaSim down with us.
        if _should_adopt_coppelia_process:
            terminate_coppelia(coppelia_process)
        pr.close_window()
        print("[INFO] encerrado")
