import math
from typing import List, Optional, Tuple

import pyray as pr

import Coppelia.helpers as cop
from Coppelia.drive import TiredWheel
from camera_link import CameraMirror
from shadow import KartLink, SimMirror, grid_heading_to_world_theta, world_theta_to_grid_heading

import src.styles as styles
from src.renders import HUDView, KartView, MazeView
from src.solver import solve_bfs, heading_between

from runner.constants import (
    CAIXA_COR, GRID_HEADING_LABEL,
    OBSTACLE_DISTANCE_CM, SENSOR_FORWARD_OFFSET_M, SENSOR_FROZEN_TRAVEL_M,
)
from runner.leg import Leg


class LiveMazeRunner:
    """Two independent axes of state:

    `phase` -- has the operator confirmed the real kart's placement yet.
        "setup": no KartLink/mirror exist; picking start/goal only moves the
                 CoppeliaSim placement preview and updates the instructions.
        "live":  KartLink connected, odometry reset and anchored, telemetry
                 mirrored into both views every frame, path being driven.

    `state` -- only meaningful once `phase == "live"`: "running" (driving a
        leg or about to start the next one), "idle" (goal reached, waiting
        for a new one) or "blocked" (a leg came up short -- obstacle or
        stall -- waiting for the operator to clear it and `resume`).
    """

    def __init__(self, sim, kart_base, motors, maze_path: str,
                view: MazeView, kart_view: KartView,
                cell_length: float, ws_url: str,
                turn_speed: float, drive_speed: float,
                video_url: str, command_url: str, video_backend: str = "sim"):
        self.sim = sim
        self.kart_base = kart_base
        self.motors = motors
        self.maze_path = maze_path
        self.view = view
        self.kart_view = kart_view
        self.hud = HUDView()
        self.cell_length = cell_length
        self.ws_url = ws_url
        self.turn_speed = turn_speed
        self.drive_speed = drive_speed

        # Video/pan-tilt mirror. Fully independent of the kart WS link (video
        # pulls straight over HTTP, and servo nudges fire-and-forget straight
        # over HTTP too -- see CameraMirror), so it works from the very start,
        # arrow keys and all, before ever typing 'ok'.
        self.camera = CameraMirror(sim, "/cam_base/joint", "/servos_link_struct/joint",
                                   video_url=video_url, command_url=command_url,
                                   video=video_backend)

        self.origin = None                     # Coppelia world origin, cell (0,0)
        self.maze_tag = None                   # for the {tag}_start/{tag}_goal markers
        self.link: Optional[KartLink] = None
        self.mirror: Optional[SimMirror] = None
        self.odom_ref = None

        self.phase = "setup"
        self.state = "running"
        self.start_cell = view.start
        self.goal_cell = view.goal
        self.waypoint: Optional[Tuple[int, int]] = None   # pending re-route via
        self.replan_requested = False

        self.plan_path: List[Tuple[int, int]] = []
        self.plan_index = 0
        self.visited: List[Tuple[int, int]] = []
        self.leg: Optional[Leg] = None
        self.status = ""
        self._preview_heading = 0.0
        self.last_world_pose = None            # (x, y, theta), once live
        self.obstacles_found = 0               # discovered by the distance sensor
        self._dist_seen_value = None           # last front reading, and where the
        self._dist_seen_pose = None            # kart was when it first read it

        self._build_maze()

    # ---- maze (re)build ----------------------------------------------------
    def _build_maze(self):
        self.maze_tag = cop.get_maze_tag(self.maze_path)
        cop.remove_object(self.sim, self.maze_tag)
        _, origin = cop.render_maze(self.sim, self.maze_path,
                                    center=(0, 0, 0),
                                    cell_cube_size=self.cell_length,
                                    wall_color=CAIXA_COR)
        self.origin = origin
        margin = 2 * self.cell_length
        cop.change_floor(self.sim,
                         (self.view.cols * self.cell_length + margin,
                          self.view.rows * self.cell_length + margin),
                         thickness=0.05)
        # The scene was just rebuilt straight from the .maze file, and the
        # discovered-obstacle boxes went out with the old tree (they hang off
        # the maze container). Put the live maze back to the file too, or the
        # two would disagree: BFS would keep routing around obstacles that no
        # longer exist in the scene. view's own clear, not the runner's --
        # the boxes are already gone, so there is nothing to remove.
        self.view.clear_obstacles()
        self.obstacles_found = 0
        self._replan_preview()

    def load_maze(self, maze_path: str) -> bool:
        if self.phase != "setup":
            print("[ERROR] so da pra trocar de maze antes do 'ok'")
            return False
        if not self.view.set_maze(maze_path):
            return False
        self.maze_path = maze_path
        self.start_cell = self.view.start
        self.goal_cell = self.view.goal
        self.waypoint = None
        self._build_maze()
        return True

    # ---- setup phase ---------------------------------------------------
    def _replan_preview(self):
        """Setup-phase only: recompute where kart_base should sit and print
        the placement instructions for the operator."""
        path = solve_bfs(self.view, self.start_cell, self.goal_cell)
        self.plan_path = path
        if not path:
            self._preview_heading = 0.0
            self.status = f"[WARN] sem caminho entre {self.start_cell} e {self.goal_cell}"
            print(self.status)
            return

        heading_deg = heading_between(path[0], path[1]) if len(path) > 1 else 0.0
        self._preview_heading = heading_deg
        x, y, _ = cop.cell_to_world(self.origin, self.cell_length, path[0])
        # set_planar_pose centers kart_base on (x, y) using its own bounding
        # box; KART_CENTER_OFFSET_LOCAL is a hand-tunable fudge on top of that
        # (see Coppelia/helpers/kart_center_offset.py) -- keep it in sync with
        # the SimMirror(...) call in confirm_ok() below, or the initial
        # placement and the ongoing tracking will disagree.
        cop.set_planar_pose(self.sim, self.kart_base, x, y,
                            grid_heading_to_world_theta(heading_deg),
                            center_offset_local=cop.KART_CENTER_OFFSET_LOCAL)

        label = GRID_HEADING_LABEL.get(heading_deg, f"{heading_deg:.0f} deg")
        self.status = (f"Posicione o kart REAL na celula {path[0]}, apontando {label}. "
                       f"Digite 'ok' quando estiver pronto.")
        print(f"[INFO] {self.status}")

    def confirm_ok(self):
        if self.phase != "setup":
            return
        if not self.plan_path:
            print("[ERROR] sem caminho -- ajuste start/goal antes do 'ok'")
            return
        print(f"[INFO] conectando no kart ({self.ws_url}) ...")
        self.link = KartLink(self.ws_url, teleop=False)
        self.link.reset_odometry()
        self.odom_ref = None
        self.mirror = SimMirror(self.sim, self.kart_base, self.motors, TiredWheel(),
                               center_offset_local=cop.KART_CENTER_OFFSET_LOCAL)
        self.phase = "live"
        self.state = "running"
        self.plan_index = 0
        self.visited = [self.plan_path[0]]
        self.leg = None
        self.status = "aguardando telemetria..."

    # ---- start/goal, valid in both phases ----------------------------------
    def _move_marker(self, name: str, cell):
        """render_maze() only draws the {tag}_start/{tag}_goal boxes once,
        from the .maze file's baked-in start/goal -- this is what keeps them
        following the operator's own choice instead of going stale."""
        cop.move_maze_marker(self.sim, self.maze_tag, name, self.origin,
                             self.cell_length, cell)

    def set_start(self, cell) -> bool:
        if self.view.is_wall(cell) or cell == self.goal_cell:
            return False
        self._move_marker("start", cell)
        if self.phase == "setup":
            self.start_cell = cell
            self.view.set_start_goal(self.start_cell, self.goal_cell)
            self._replan_preview()
        else:
            self.waypoint = cell
            self.replan_requested = True
            self.state = "running"
        return True

    def set_goal(self, cell) -> bool:
        if self.view.is_wall(cell) or cell == self.start_cell:
            return False
        self._move_marker("goal", cell)
        self.goal_cell = cell
        if self.phase == "setup":
            self.view.set_start_goal(self.start_cell, self.goal_cell)
            self._replan_preview()
        else:
            self.replan_requested = True
            self.state = "running"
        return True

    def resume(self):
        """Operator cleared the obstruction: forget what the distance sensor
        learned -- the LIVE maze goes back to the .maze file's original -- and
        replan from wherever the kart actually stopped, toward the same goal.
        Anything still physically in the way is simply re-discovered on the
        next approach, so this is safe to use liberally."""
        if self.phase != "live":
            return
        # "blocked" is a leg that came up short. But a discovery can also close
        # the LAST route to the goal, which lands in "idle" -- and without this
        # that is a dead end: the obstacles that closed the route could only be
        # forgotten from "blocked", so nothing could ever reopen it.
        if self.state != "blocked" and not (self.state == "idle" and self.view.obstacles):
            return
        self.clear_obstacles()
        self.replan_requested = True
        self.state = "running"

    def clear_obstacles(self):
        """Live maze back to the original, in BOTH views: the discovered-cell
        boxes come out of the Coppelia scene and MazeView restores its grid
        from the pristine copy it snapshotted at load."""
        for (r, c) in self.view.obstacles:
            try:
                cop.remove_object(self.sim, f"{self.maze_tag}_obs_{r}_{c}")
            except Exception:                       # noqa: BLE001
                pass                                # already gone -- fine
        self.view.clear_obstacles()
        self.obstacles_found = 0

    # ---- camera (pan/tilt), valid in both phases ---------------------------
    def nudge(self, d_pan_deg: float, d_tilt_deg: float):
        """Move the sim pan/tilt joints by a relative amount (degrees); picked
        up next `step()` and pushed to the real servos too, so the real kart's
        camera shadows whatever the sim joints are set to."""
        self.camera.nudge(d_pan_deg, d_tilt_deg)

    def look_forward(self):
        """Re-center the camera (sim + real) to its calibrated home pose."""
        self.camera.look_forward()

    # ---- live phase ------------------------------------------------------
    def _nearest_cell(self, wx, wy) -> Tuple[int, int]:
        rf, cf = cop.world_to_cell(self.origin, self.cell_length, wx, wy)
        r = max(0, min(self.view.rows - 1, round(rf)))
        c = max(0, min(self.view.cols - 1, round(cf)))
        return (r, c)

    def _replan_live(self, current_cell):
        if self.waypoint is not None:
            via, self.waypoint = self.waypoint, None
            leg1 = solve_bfs(self.view, current_cell, via)
            leg2 = solve_bfs(self.view, via, self.goal_cell) if leg1 else []
            path = (leg1 + leg2[1:]) if (leg1 and leg2) else []
        else:
            path = solve_bfs(self.view, current_cell, self.goal_cell)

        self.replan_requested = False
        if not path:
            self.status = f"[ERROR] sem caminho a partir de {current_cell}"
            if self.view.obstacles:
                # The live maze is what closed the route, so say so and point
                # at the way out -- otherwise this reads as a dead end.
                self.status += (f" ({len(self.view.obstacles)} obstaculo(s) detectado(s));"
                                f" libere o caminho e digite 'resume'")
            self.status += ". Ou escolha outro start/goal."
            print(self.status)          # already carries its own [ERROR]
            self.plan_path = [current_cell]
            self.plan_index = 0
            self.state = "idle"
            return
        self.plan_path = path
        self.plan_index = 0
        self.state = "running"

    def step(self, dt: float):
        # Camera mirroring runs in both phases -- it doesn't need the KartLink
        # at all (servo nudges fire straight over HTTP, see CameraMirror).
        self.camera.apply()
        self.camera.push_frame()

        if self.phase != "live" or self.link is None:
            return
        pose, wheel = self.link.latest()
        if pose is None:
            return
        if self.odom_ref is None:
            self.odom_ref = pose

        wx, wy, wth = self.mirror.world_pose(pose, self.odom_ref)
        self.mirror.apply_world_pose(wx, wy, wth)
        self.mirror.spin_wheels(wheel[0], wheel[1], dt)
        self.last_world_pose = (wx, wy, wth)
        self._update_kart_view(wx, wy, wth)

        drive_tel = self.link.drive_telemetry()
        # Runs even while blocked/idle -- a stopped run is exactly when you
        # most want to learn what is actually in the way.
        self._scan_obstacle(drive_tel, wx, wy, wth)

        if self.state != "running":
            return
        self._advance(drive_tel, wx, wy, wth)

    def _scan_obstacle(self, drive_tel, wx, wy, wth):
        """Front distance sensor -> LIVE maze. Anything closer than
        OBSTACLE_DISTANCE_CM is projected onto the grid and that cell is
        blocked, so the next replan routes around it. The .maze file's own
        copy is never touched (see MazeView's original-vs-live split), so the
        original stays the reference for what was there to begin with.

        The reading is projected rather than blindly blamed on the next cell:
        it comes from the sensor, which sits SENSOR_FORWARD_OFFSET_M ahead of
        the tracked center, so adding that back and walking the measured
        distance along the current heading lands on whichever cell the
        obstacle is really in -- one cell ahead or two."""
        dist_cm = drive_tel.get("front_distance_cm")
        if dist_cm is None:
            return
        try:
            dist_cm = float(dist_cm)
        except (TypeError, ValueError):
            return
        if dist_cm <= 0.0 or dist_cm > OBSTACLE_DISTANCE_CM:
            return

        # Staleness: the kart reports `front_distance_cm` STICKILY -- the
        # server only assigns it on a VALID read and (on an un-updated server)
        # never clears it, and the RPi.GPIO sensor answers "out of range" with
        # a 255 sentinel that gets dropped. So a clear road never refreshes the
        # field and the last close reading can sit there forever, which would
        # have us stamping phantom walls across every cell we then drive
        # through. An identical reading that survives half a cell of travel is
        # physically impossible for something this close, so treat it as
        # frozen. (The server-side TTL is the real fix; this keeps the map
        # honest until that is redeployed, and costs nothing after.)
        if dist_cm == self._dist_seen_value and self._dist_seen_pose is not None:
            px, py = self._dist_seen_pose
            if math.hypot(wx - px, wy - py) > SENSOR_FROZEN_TRAVEL_M:
                return
        else:
            self._dist_seen_value = dist_cm
            self._dist_seen_pose = (wx, wy)

        reach = SENSOR_FORWARD_OFFSET_M + dist_cm / 100.0
        hx = wx + math.cos(wth) * reach
        hy = wy + math.sin(wth) * reach
        rf, cf = cop.world_to_cell(self.origin, self.cell_length, hx, hy)
        cell = (round(rf), round(cf))

        if not self.view.in_bounds(cell):
            return
        if cell == self._nearest_cell(wx, wy):
            return                  # our own footprint -- too close to blame a cell
        if cell == self.goal_cell:
            return                  # never wall off the goal; that just bricks the run
        if not self.view.mark_obstacle(cell):
            return                  # already a wall, or already discovered

        self.obstacles_found += 1
        print(f"[WARN] obstaculo detectado em {cell} ({dist_cm:.0f} cm) -- "
              f"maze ao vivo atualizado")
        try:
            cop.add_maze_obstacle(self.sim, self.maze_tag, self.origin,
                                  self.cell_length, cell)
        except Exception as e:                      # noqa: BLE001
            print(f"[WARN] nao deu pra desenhar o obstaculo no Coppelia ({e})")

        # Only worth re-routing if it actually sits on the plan. Left as a
        # REQUEST, not an immediate replan: _advance() picks it up at the next
        # cell boundary, so a leg in flight is never yanked mid-move -- and a
        # blocked run stays blocked until the operator types 'resume'.
        if cell in self.plan_path:
            self.replan_requested = True

    def _update_kart_view(self, wx, wy, wth):
        rf, cf = cop.world_to_cell(self.origin, self.cell_length, wx, wy)
        px, py = self.view.center_f(rf, cf)
        self.kart_view.set_offset(px, py)
        self.kart_view.set_rotation(world_theta_to_grid_heading(wth))

    def _advance(self, drive_tel, wx, wy, wth):
        if self.leg is None:
            if self.replan_requested:
                self._replan_live(self._nearest_cell(wx, wy))
                if self.state != "running":
                    return
            if not self.plan_path or self.plan_index + 1 >= len(self.plan_path):
                self.state = "idle"
                reached = self.plan_path[-1] if self.plan_path else "?"
                self.status = f"GOAL alcancado em {reached}"
                return
            frm = self.plan_path[self.plan_index]
            to = self.plan_path[self.plan_index + 1]
            self.leg = Leg(frm, to, wth, self.cell_length,
                           self.turn_speed, self.drive_speed)
            self.status = f"{frm} -> {to}"
            return

        outcome = self.leg.poll(self.link, drive_tel, wx, wy)
        if outcome == "pending":
            return
        if outcome == "arrived":
            self.visited.append(self.leg.to_cell)
            self.plan_index += 1
            self.leg = None
            return

        # "blocked"
        reason = "obstaculo a frente" if self.leg.guard_seen else "parou antes do previsto"
        self.status = (f"[INTERROMPIDO] {self.leg.from_cell} -> {self.leg.to_cell} "
                       f"({reason}, faltou {self.leg.shortfall:.2f} m). "
                       f"Limpe o caminho e digite 'resume'.")
        print(f"[WARN] {self.status}")
        self.state = "blocked"
        self.leg = None

    # ---- rendering ---------------------------------------------------------
    def hud_lines(self):
        video_line = (f"video: {self.camera.video_status()}   "
                      f"cam pan={self.camera.pan_pct}% tilt={self.camera.tilt_pct}%   "
                      f"(arrows=cam, C=centraliza)", 14, styles.HUDColors.text_dim)
        if self.phase == "setup":
            return (
                (self.status, 18, styles.HUDColors.text),
                video_line,
                ("L-click start   R-click goal   digite 'ok' quando o kart "
                 "REAL estiver posicionado", 16, styles.HUDColors.text_dim),
            )
        color = {"running": styles.HUDColors.text,
                "idle": styles.HUDColors.text_ok,
                "blocked": styles.HUDColors.text_fail}.get(
            self.state, styles.HUDColors.text)
        second = ""
        if self.last_world_pose is not None:
            x, y, th = self.last_world_pose
            second = f"kart em ({x:+.2f}, {y:+.2f})  heading={math.degrees(th):+.0f} deg"
        if self.obstacles_found:
            second += f"   |  {self.obstacles_found} obstaculo(s) no maze ao vivo"
        third = "L-click novo start   R-click novo goal"
        if self.state == "blocked":
            third += "   digite 'resume' para tentar de novo"
        return (
            (self.status, 18, color),
            video_line,
            (second, 16, styles.HUDColors.text_dim),
            (third, 16, styles.HUDColors.panel_line),
        )

    def render(self):
        pr.begin_drawing()
        pr.clear_background(styles.HUDColors.background)

        self.view.render()
        if self.plan_path:
            self.view.render_cells(self.plan_path, styles.HUDColors.path_cell)
        if self.visited:
            self.view.render_cells(self.visited, styles.HUDColors.trail_cell, 0.25)

        if self.phase == "setup" and self.plan_path:
            px, py = self.view.center_f(*self.plan_path[0])
            self.kart_view.set_offset(px, py)
            self.kart_view.set_rotation(self._preview_heading)
            self.kart_view.render()
        elif self.phase == "live" and self.last_world_pose is not None:
            self.kart_view.render()

        self.hud.render(self.hud_lines())
        pr.end_drawing()

    def stop(self):
        self.camera.stop()
        if self.link is not None:
            self.link.stop()
