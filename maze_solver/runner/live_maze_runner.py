import math
import time
from typing import List, Optional, Tuple

import pyray as pr

import Coppelia.helpers as cop
from Coppelia.drive import TiredWheel
from camera_link import CameraMirror
from shadow import (KartLink, SimMirror, grid_heading_to_world_theta,
                    world_theta_to_grid_heading, wrap)

import src.styles as styles
from src.renders import HUDView, KartView, MazeView
from src.solver import DIRECTIONS, solve_bfs, heading_between

from runner.constants import (
    CAIXA_COR, GRID_HEADING_LABEL, LIVE_RETRY_INTERVAL_S,
    RANGE_FIX_MAX_PREDICTED_M, SCAN_FIT_MIN_SAMPLES,
    OBSTACLE_DISTANCE_CM, SENSOR_CLEAR_CONFIRMATIONS,
    SENSOR_FORWARD_OFFSET_M, SENSOR_FROZEN_TRAVEL_M,
    OBSTACLE_HIT_DEPTH_M, OBSTACLE_ALIGN_TOLERANCE_DEG,
    OBSTACLE_STEADY_SPEED_MPS, OBSTACLE_CONFIRM_S, OBSTACLE_SUSPECT_TIMEOUT_S,
    SWEEP_EVERY_N_LEGS, LEG_MAX_CELLS, CLOSEST_OBSTACLE_AS_REFERENCE,
)
from runner.closest_obstacle import ClosestObstacleFix
from runner.leg import Leg, run_heading, straight_run_end
from runner.range_fix import RangeFix, is_trusted_reading, predict_range
from runner.recenter import Recenter
from runner.sweep import Sweep, plan_angles


class LiveMazeRunner:
    """Two independent axes of state:

    `phase` -- has the operator confirmed the real kart's placement yet.
        "setup": no KartLink/mirror exist; picking start/goal only moves the
                 CoppeliaSim placement preview and updates the instructions.
        "live":  KartLink connected, odometry reset and anchored, telemetry
                 mirrored into both views every frame, path being driven.

    `state` -- only meaningful once `phase == "live"`: "running" (driving a
        leg or about to start the next one), "idle" (goal reached, waiting
        for a new one), "blocked" (a leg came up short -- obstacle or stall)
        or "waiting" (no route to the goal through the live maze right now).

    The last two are holding patterns, NOT dead ends: the front sensor keeps
    writing the live maze while stopped, and both states re-run the planner on
    their own -- immediately when the live maze changes, and on a slow timer
    otherwise. Take a box off the floor and the kart sees the cell go clear,
    un-discovers it, finds a route and drives off without being told. `resume`
    is still there as the big hammer (forget EVERY discovery at once), but
    nothing waits for it anymore.
    """

    def __init__(self, sim, kart_base, motors, maze_path: str,
                view: MazeView, kart_view: KartView,
                cell_length: float, ws_url: str,
                turn_speed: float, drive_speed: float,
                video_url: str, command_url: str, 
                video_backend: str = "sim", wait_to_step: bool = False):
        self.sim         = sim
        self.kart_base   = kart_base
        self.motors      = motors
        self.maze_path   = maze_path
        self.view        = view
        self.kart_view   = kart_view
        self.hud         = HUDView()
        self.cell_length = cell_length
        self.ws_url      = ws_url
        self.turn_speed  = turn_speed
        self.drive_speed = drive_speed
        
        # Manual step gate: with `should_wait`, each leg is held at the cell
        # boundary until the operator calls release_step(). `_waiting_to_step`
        # is "parked at the gate right now" (a press is only honoured then, so
        # an early press cannot skip the next cell); `_step_released` is the
        # press itself, consumed by the leg it lets through.
        self.should_wait = wait_to_step
        self._waiting_to_step = False
        self._step_released = False

        # Video/pan-tilt mirror. Fully independent of the kart WS link (video
        # pulls straight over HTTP, and servo nudges fire-and-forget straight
        # over HTTP too -- see CameraMirror), so it works from the very start,
        # arrow keys and all, before ever typing 'ok'.
        self.camera = CameraMirror(sim, "/cam_base/joint", "/servos_link_struct/joint",
                                   video_url=video_url, command_url=command_url,
                                   video=video_backend)

        self.origin                      = None # Coppelia world origin, cell (0,0)
        self.maze_tag                    = None # for the {tag}_start/{tag}_goal markers
        self.link: Optional[KartLink]    = None
        self.mirror: Optional[SimMirror] = None
        self.odom_ref                    = None

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
        self._preview_heading             = 0.0
        self.last_world_pose              = None            # (x, y, theta), once live
        self.obstacles_found              = 0               # discovered by the distance sensor
        self._dist_seen_value             = None            # last front reading, and where the
        self._dist_seen_pose              = None            # kart was when it first read it
        self._last_slip                   = {}              # kart's slip sub-dict, for the HUD
        self.recenter: Optional[Recenter] = None            # physical recentring in flight
        self.sweep: Optional[Sweep]       = None            # angular sweep in flight
        # CLOSEST_OBSTACLE_AS_REFERENCE's replacement for the sweep, and the
        # side it turned to last (+1 = +90, -1 = -90, 0 = none yet) so the
        # next one turns the other way.
        self.side_ref: Optional[ClosestObstacleFix] = None
        self._last_side_turn = 0
        
        self._cells_since_sweep = 0     # cells, not legs -- see SWEEP_EVERY_N_LEGS
        self.sweeps_done = 0
        self.last_sweep_summary = ""

        # Obstacle confirmation (see _scan_obstacle). `_ahead_verdict` is the
        # running ((cell, blocked), since) a steady kart is accumulating about
        # the cell in front of it. `_suspect` is a cell a moving leg was
        # stopped short of, held until that verdict (or the timeout) settles
        # it; `_dismissed` holds suspects that did not confirm, so the leg
        # driven right after the hold does not stop for them again.
        self._ahead_verdict = None
        self._suspect: Optional[Tuple[int, int]] = None
        self._suspect_verdict: Optional[str] = None     # "blocked" | "clear"
        self._suspect_deadline: Optional[float] = None
        self._dismissed = set()

        # Live-maze bookkeeping for the self-retry loop. `_live_revision` ticks
        # on every discovery and every un-discovery; a holding state compares it
        # against the revision its last failed plan was computed from, so a real
        # change retries instantly while an unchanged maze only retries on the
        # timer.
        self._live_revision = 0
        self._planned_revision = -1
        self._retry_at = 0.0
        self._clear_looks = {}

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
        # 1-D longitudinal pose correction, rebuilt with the maze because it
        # raycasts against this maze's walls and origin.
        self.range_fix = RangeFix(self.view, origin, self.cell_length)
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
        self._forget_discoveries()
        self._replan_preview()

    def load_maze(self, maze_path: str) -> bool:
        if self.phase != "setup":
            print("[ERROR] so da pra trocar de maze antes do 'ok'")
            return False
        if not self.view.set_maze(maze_path):
            return False
        self.maze_path  = maze_path
        self.start_cell = self.view.start
        self.goal_cell  = self.view.goal
        self.waypoint   = None
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
        """Forget EVERY discovery at once and replan from wherever the kart
        actually stopped. The runner already un-discovers cells by itself as
        the sensor sees them go clear, so this is no longer needed to get
        moving again -- it is the manual override for when the operator knows
        the whole live maze is stale (things moved somewhere the kart is not
        looking) and would rather re-learn it from scratch than wait. Anything
        still physically in the way is re-discovered on the next approach."""
        if self.phase != "live":
            return
        self.clear_obstacles()
        self.replan_requested = True
        self.state = "running"

    def release_step(self) -> bool:
        """Let the leg parked at the step gate go. Only counts while actually
        parked there -- returns False (and does nothing) otherwise."""
        if not self._waiting_to_step:
            return False
        self._step_released   = True
        self._waiting_to_step = False
        return True

    def clear_obstacles(self):
        """Live maze back to the original, in BOTH views: the discovered-cell
        boxes come out of the Coppelia scene and MazeView restores its grid
        from the pristine copy it snapshotted at load."""
        for (r, c) in self.view.obstacles:
            self._remove_obstacle_box((r, c))
        self.view.clear_obstacles()
        self.obstacles_found = 0
        self._forget_discoveries()

    def _forget_discoveries(self):
        """Drop the per-cell evidence the live maze was built from and mark it
        as changed, so any holding state re-plans against what is there now."""
        self._clear_looks = {}
        self._live_revision += 1

    def _remove_obstacle_box(self, cell):
        for_name = f"{self.maze_tag}_obs_{cell[0]}_{cell[1]}"
        try:
            cop.remove_object(self.sim, for_name)
        except Exception:                           # noqa: BLE001
            pass                                    # already gone -- fine

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
        self._planned_revision = self._live_revision
        
        if not path:
            # Not a dead end: _reconsider_live() keeps re-running this, so the
            # moment the live maze opens up -- something taken off the floor,
            # a cell un-discovered -- the run continues on its own.
            if self.view.obstacles:
                self.status = (f"[ESPERANDO] sem caminho a partir de {current_cell} "
                               f"({len(self.view.obstacles)} obstaculo(s) detectado(s)). "
                               f"Libere o caminho -- eu sigo sozinho.")
            else:
                self.status = (f"[ESPERANDO] sem caminho a partir de {current_cell}. "
                               f"Escolha outro start/goal.")
                
            if self.state != "waiting":
                print(self.status)      # once per entry, not once per retry
                
            self.plan_path = [current_cell]
            self.plan_index = 0
            self.state = "waiting"
            
            return
        
        if self.state in ("blocked", "waiting"):
            print(f"[INFO] caminho reaberto a partir de {current_cell} -- seguindo")
        
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
        # Cached for the HUD, and consulted by both the obstacle scan and the
        # range fix. Empty against a kart whose server predates the detector.
        self._last_slip = drive_tel.get("slip") or {}
        # Nothing commanded AND the wheels actually stopped: the only state in
        # which a reading is allowed to block a cell (see _scan_obstacle).
        steady = (not drive_tel.get("goal_active")
                  and max(abs(wheel[0]), abs(wheel[1])) < OBSTACLE_STEADY_SPEED_MPS)
        # Runs even while blocked/waiting -- a stopped run is exactly when you
        # most want to learn what is actually in the way, and (now) when it
        # stops being in the way.
        self._scan_obstacle(drive_tel, wx, wy, wth, steady)

        if self.state in ("blocked", "waiting"):
            self._reconsider_live(wx, wy)
        if self.state != "running":
            return
        
        self._advance(drive_tel, wx, wy, wth)

    def _reconsider_live(self, wx, wy):
        now = time.monotonic()
        if self._live_revision == self._planned_revision and now < self._retry_at:
            return
        self._retry_at = now + LIVE_RETRY_INTERVAL_S
        self._replan_live(self._nearest_cell(wx, wy))

    def _obstacle_scaned_is_valid(self,wx,wy,wth,cell) -> bool:
        """Gate a projected hit before it is allowed to block a cell.

        Every check here distrusts the PROJECTION, not the reading: the pose
        drifts, and a hit that lands somewhere the sensor could not have seen
        from means the pose was wrong -- not that the maze changed.

        Order matters. mark_obstacle() mutates the live maze, so it has to be
        the last thing that runs; anything checked after it would already have
        blocked the cell it was meant to reject."""
        # Cells lay in the maze
        if not self.view.in_bounds(cell):
            return False

        origin = self._nearest_cell(wx, wy)
        # Cell is not the one the kart is in, nor the current kart goal
        if cell == origin or cell == self.goal_cell:
            return False

        # Origin Cell is an blocked wall
        if self.view.is_wall(origin):
            return False

        # A ray from the Origin center to the Cell center must not cross an
        # already Blocked/Wall Cell
        if not self._has_line_of_sight(origin, cell):
            return False

        # Cell already a wall, or already discovered
        return self.view.mark_obstacle(cell)

    def _has_line_of_sight(self, src, dst) -> bool:
        """Walk the straight line between two cell centers and report whether
        it reaches `dst` without crossing a cell already known solid. The two
        endpoints are skipped -- they are what the line connects, not things
        standing in the way of it."""
        dr = dst[0] - src[0]
        dc = dst[1] - src[1]
        # ~20 samples per cell crossed: fine enough to catch any cell the line
        # genuinely passes through, coarse enough to ignore corner grazes that
        # clip a wall by a hair on a diagonal.
        steps = max(1, int(math.hypot(dr, dc) * 20))
        for i in range(1, steps):
            t = i / steps
            mid = (round(src[0] + dr * t), round(src[1] + dc * t))
            if mid == src or mid == dst:
                continue
            if self.view.is_wall(mid):
                return False
        return True

    def _scan_obstacle(self, drive_tel, wx, wy, wth, steady):
        """Front distance sensor <-> LIVE maze, in BOTH directions.

        Runs every frame, moving or not. A reading says three things:

        * Everything the beam passed THROUGH is empty -- it would have stopped
          earlier otherwise -- so any cell in that span previously discovered
          blocked gets un-discovered (`_carve_free`). Live, at any time: that
          is what lets a run resume by itself once a box is taken off the
          floor, without anybody typing 'resume'.
        * While a leg is driving, a hit on a cell that leg is about to enter
          makes it a SUSPECT, and the kart is stopped short of it, in the cell
          before (`_check_suspect`). Nothing is marked yet.
        * A STEADY kart, facing a grid axis, gets to judge the ONE cell right
          in front of it: blocked if the hit lands there, clear if the beam
          went past it (`_judge_ahead`). That is the only place a cell is ever
          marked.

        Why not mark wherever the beam lands, as this used to: a hit two cells
        out, from a moving kart, is projected through a pose that is still
        integrating the move, along a heading that may be mid-turn -- one
        frame of that is enough to block a corridor that is open. A steady
        kart one cell away has neither problem, and a wrong verdict there
        would need the obstacle to be somewhere the beam physically cannot
        miss.

        The hit is projected rather than blindly blamed on the next cell: it
        comes from the sensor, which sits SENSOR_FORWARD_OFFSET_M ahead of the
        tracked center, so adding that back and walking the measured distance
        along the current heading lands on whichever cell the obstacle is
        really in. The cell it is BLAMED on is taken OBSTACLE_HIT_DEPTH_M
        further on (the reading is the near face; see constants.py).

        The .maze file's own copy is never touched in either direction (see
        MazeView's original-vs-live split), so the original stays the reference
        for what was there to begin with -- and a wall that came from the file
        is never something a reading gets to erase.

        Suspended while the kart reports slipping. The reading itself is fine
        -- the ultrasonic does not care what the wheels are doing -- but the
        POSE it gets projected through was integrated from those slipping
        wheels, so the hit lands in the wrong cell."""
        if self._last_slip.get("slipping"):
            self._ahead_verdict = None
            return
        dist_cm = drive_tel.get("front_distance_cm")
        try:
            dist_cm = float(dist_cm)
        except (TypeError, ValueError):
            dist_cm = None
        if dist_cm is None or dist_cm <= 0.0:
            # No measurement is not "clear": the verdict has to be built from
            # readings, so a gap in them starts it over.
            self._ahead_verdict = None
            return

        # Frozen readings gate everything: a stuck sensor must not be allowed
        # to erase obstacles any more than it is allowed to invent them.
        if dist_cm == self._dist_seen_value and self._dist_seen_pose is not None:
            px, py = self._dist_seen_pose
            if math.hypot(wx - px, wy - py) > SENSOR_FROZEN_TRAVEL_M:
                self._ahead_verdict = None
                return
        else:
            self._dist_seen_value = dist_cm
            self._dist_seen_pose = (wx, wy)

        if dist_cm > OBSTACLE_DISTANCE_CM:
            # Nothing within range, so the whole span out to the range limit is
            # empty. Past that the reading is not a measurement of anything in
            # particular, just "far", so it carves nothing.
            self._carve_free(wx, wy, wth,
                             SENSOR_FORWARD_OFFSET_M + OBSTACLE_DISTANCE_CM / 100.0,
                             hit_cell=None)
            self._judge_ahead(None, steady, wx, wy, wth, dist_cm)
            return

        reach = SENSOR_FORWARD_OFFSET_M + dist_cm / 100.0
        # Carve against the face itself, not the deeper blamed cell: a face
        # projected just short of a boundary may be the next cell's obstacle,
        # and carving that cell free on it would erase what is really there.
        self._carve_free(wx, wy, wth, reach,
                         hit_cell=self._project(wx, wy, wth, reach))

        cell = self._project(wx, wy, wth, reach + OBSTACLE_HIT_DEPTH_M)
        self._check_suspect(cell, wx, wy, wth)
        self._judge_ahead(cell, steady, wx, wy, wth, dist_cm)

    def _project(self, wx, wy, wth, reach) -> Tuple[int, int]:
        """The cell `reach` metres ahead of the kart's center, along wth."""
        rf, cf = cop.world_to_cell(self.origin, self.cell_length,
                                   wx + math.cos(wth) * reach,
                                   wy + math.sin(wth) * reach)
        return (round(rf), round(cf))

    def _cell_ahead(self, wx, wy, wth) -> Optional[Tuple[int, int]]:
        """The neighbour of the kart's cell that it is facing, or None when
        the heading is not within OBSTACLE_ALIGN_TOLERANCE_DEG of a grid axis
        (then it is not facing any one cell)."""
        heading = world_theta_to_grid_heading(wth)
        axis = (round(heading / 90.0) * 90.0) % 360.0
        if abs((heading - axis + 180.0) % 360.0 - 180.0) > OBSTACLE_ALIGN_TOLERANCE_DEG:
            return None
        r, c = self._nearest_cell(wx, wy)
        for dr, dc, grid_heading in DIRECTIONS:
            if grid_heading == axis:
                return (r + dr, c + dc)
        return None

    def _judge_ahead(self, cell, steady, wx, wy, wth, dist_cm):
        """The only place a reading BLOCKS a cell: the kart is steady in
        (r, c), facing a grid axis, and the hit lands in the very next cell
        along it. `cell` is the blamed cell of this reading, or None when
        nothing was in range.

        The verdict -- that cell blocked, or that cell clear because the beam
        went past it -- has to hold for OBSTACLE_CONFIRM_S of continuous
        steadiness before anything acts on it. Blocked marks it; either way
        it settles a pending suspect for that cell (see _hold_for_suspect)."""
        ahead = self._cell_ahead(wx, wy, wth) if steady else None
        if ahead is None or not self.view.in_bounds(ahead):
            self._ahead_verdict = None
            return
        if cell == ahead:
            blocked = True
        elif cell is None or cell != self._nearest_cell(wx, wy):
            blocked = False
        else:
            # Hit inside the kart's own cell: too close to say anything about
            # the next one.
            self._ahead_verdict = None
            return

        now = time.monotonic()
        verdict = (ahead, blocked)
        if self._ahead_verdict is None or self._ahead_verdict[0] != verdict:
            self._ahead_verdict = (verdict, now)
            return
        if now - self._ahead_verdict[1] < OBSTACLE_CONFIRM_S:
            return

        if blocked:
            self._discover(ahead, wx, wy, wth, dist_cm)
        if ahead == self._suspect and self._suspect_verdict is None:
            self._suspect_verdict = "blocked" if blocked else "clear"

    def _discover(self, cell, wx, wy, wth, dist_cm):
        """Block `cell` in the live maze and the Coppelia scene, once."""
        if not self._obstacle_scaned_is_valid(wx, wy, wth, cell):
            return

        self._live_revision += 1
        self.obstacles_found = len(self.view.obstacles)
        self._clear_looks.pop(cell, None)
        print(f"[WARN] obstaculo detectado em {cell} ({dist_cm:.0f} cm) -- "
              f"maze ao vivo atualizado")
        try:
            cop.add_maze_obstacle(self.sim, self.maze_tag, self.origin,
                                  self.cell_length, cell)
        except Exception as e:                      # noqa: BLE001
            print(f"[WARN] nao deu pra desenhar o obstaculo no Coppelia ({e})")

        if cell in self.plan_path:
            self.replan_requested = True

    # ---- suspects: stop short, then let the steady check decide ------------
    def _check_suspect(self, cell, wx, wy, wth):
        """A hit on a cell the running leg is about to enter: stop the kart
        in the cell before it, so the steady check can look at it from there.

        Only once the kart is on the run's heading. While it is still turning
        onto it the beam sweeps across cells at an angle, and a hit there says
        nothing about the corridor the kart is going to drive."""
        leg = self.leg
        if leg is None or self._suspect is not None or cell in self._dismissed:
            return
        if abs(math.degrees(wrap(wth - leg.target_theta))) > OBSTACLE_ALIGN_TOLERANCE_DEG:
            return
        first = self.plan_index + 1
        route = self.plan_path[first:first + leg.n_cells]
        if cell not in route:
            return
        origin = self._nearest_cell(wx, wy)
        # Same gates as marking (_obstacle_scaned_is_valid), minus the mark.
        if (cell == origin or cell == self.goal_cell or self.view.is_wall(cell)
                or not self._has_line_of_sight(origin, cell)):
            return
        self._stop_short_of(cell, first + route.index(cell) - 1, wx, wy)

    def _stop_short_of(self, suspect, approach_index, wx, wy):
        """End the running leg on plan_path[approach_index], the cell right
        before `suspect`.

        Still behind that cell: the leg is cut short in flight and the kart
        drives on to its centre (Leg.retarget) -- one smooth stop, facing the
        suspect, already where the steady check needs it. Already inside it:
        stop right there, since aiming a goto at a centre the kart has passed
        would turn it around to reach it."""
        approach = self.plan_path[approach_index]
        self._suspect = suspect
        self._suspect_verdict = None
        self._suspect_deadline = None

        if (approach_index > self.plan_index
                and self._nearest_cell(wx, wy) != approach):
            target_world, odom_target = self._leg_target(approach)
            self.leg.retarget(self.link, approach, wx, wy,
                              target_world=target_world, odom_target=odom_target)
            self.status = f"{self.leg.from_cell} -> {approach} (suspeita em {suspect})"
            print(f"[WARN] possivel obstaculo em {suspect} -- parando antes, em {approach}")
            return

        self.link.halt()
        self._advance_plan(approach_index - self.plan_index)
        self.leg = None
        print(f"[WARN] possivel obstaculo em {suspect} -- parando em {approach}")

    def _hold_for_suspect(self) -> bool:
        """Keep the kart parked until the steady check settles the suspect.
        True once it is settled (or timed out) and the run may go on."""
        now = time.monotonic()
        if self._suspect_deadline is None:
            self._suspect_deadline = now + OBSTACLE_SUSPECT_TIMEOUT_S
        verdict = self._suspect_verdict
        if verdict is None and now < self._suspect_deadline:
            self.status = f"parado: confirmando possivel obstaculo em {self._suspect}"
            return False

        cell, self._suspect = self._suspect, None
        if verdict == "blocked" and self.view.is_wall(cell):
            # _discover already logged it and asked for the replan.
            return True
        # Not confirmed: drive on, and do not stop for it again on the way.
        self._dismissed.add(cell)
        why = "livre" if verdict == "clear" else "sem leitura conclusiva"
        print(f"[INFO] {cell}: {why} -- seguindo")
        return True

    def _advance_plan(self, cells):
        """The kart is now `cells` further along plan_path."""
        first = self.plan_index + 1
        self.visited.extend(self.plan_path[first:first + cells])
        self.plan_index += cells
        self._cells_since_sweep += cells

    def _carve_free(self, wx, wy, wth, length, hit_cell):
        """Un-discover the cells this reading just proved empty.

        Walks outward from the kart along its heading, stopping at `hit_cell`
        (where the beam STOPPED -- the opposite of empty) and at the first cell
        the .maze file itself walls off, since nothing beyond that was measured.

        A cell must read clear SENSOR_CLEAR_CONFIRMATIONS frames in a row before
        it is actually dropped. Pose drift makes the projected ray wander by a
        cell now and then, and one stray frame must not unblock something real
        -- while an obstacle that is genuinely gone reads clear on every frame
        and clears the bar in a fraction of a second."""
        crossed = set()
        for cell in self._cells_along(wx, wy, wth, length):
            if cell == hit_cell or self.view.is_original_wall(cell):
                break
            crossed.add(cell)
            if cell not in self.view.obstacles:
                continue
            looks = self._clear_looks.get(cell, 0) + 1
            self._clear_looks[cell] = looks
            if looks >= SENSOR_CLEAR_CONFIRMATIONS:
                self._unmark_obstacle(cell)

        # The streak has to be consecutive to mean anything, so a cell the beam
        # is no longer crossing starts over instead of banking progress from
        # the last time the kart happened to be pointed at it.
        for cell in list(self._clear_looks):
            if cell not in crossed:
                del self._clear_looks[cell]

    def _cells_along(self, wx, wy, wth, length):
        """The cells a beam of `length` crosses, in order, kart outward."""
        cx, cy = math.cos(wth), math.sin(wth)
        steps = max(1, int(length / (self.cell_length * 0.25)))
        out = []
        for i in range(steps + 1):
            t = length * i / steps
            rf, cf = cop.world_to_cell(self.origin, self.cell_length,
                                       wx + cx * t, wy + cy * t)
            cell = (round(rf), round(cf))
            if not out or out[-1] != cell:
                out.append(cell)
        return out

    def _unmark_obstacle(self, cell):
        """Drop one discovery from the live maze and the Coppelia scene."""
        if not self.view.clear_obstacle(cell):
            return
        self._clear_looks.pop(cell, None)
        self._live_revision += 1
        self.obstacles_found = len(self.view.obstacles)
        self._remove_obstacle_box(cell)
        print(f"[INFO] {cell} livre de novo -- maze ao vivo atualizado")

    def _start_recenter(self, wx, wy, wth):
        """Begin a physical recentring session at a cell boundary.

        The pose is passed in because the session FREEZES its target range from
        it -- a target recomputed from the moving belief would retreat exactly
        as fast as the kart advanced. See runner/recenter.py.
        """
        self.recenter = Recenter(self.range_fix, self.link, wx, wy, wth)

    def _poll_recenter(self, drive_tel, wx, wy, wth):
        """Drive the session forward one frame.

        On success the kart now sits where the belief already said it was, so
        nothing further is needed -- the two were reconciled by moving the kart
        rather than by editing the pose.

        On "giveup" the kart could not be moved there, so the estimated anchor
        correction is applied instead. That fallback matters: a kart that
        physically cannot close the gap (wheels against a wall, a surface it
        cannot grip) would otherwise keep both a wrong pose AND a wrong
        position. Moving the belief is always available, even when moving the
        kart is not.
        """
        outcome = self.recenter.poll(drive_tel)
        if outcome == "pending":
            self.status = f"recentrando: {self.recenter.status}"
            return

        summary = self.recenter.summary()
        if outcome == "giveup":
            print(f"[WARN] recentragem: {summary} -- corrigindo so a crenca")
            self._apply_range_fix(drive_tel, wx, wy, wth)
        elif self.recenter.moved():
            print(f"[INFO] recentragem: {summary}")

        self.recenter = None
        # Longitudinal axis is now as good as the sensor can make it, so hand
        # over to the sweep for the lateral one. Chained here rather than in
        # `_advance` so it only runs at a boundary that actually completed its
        # recentring, never mid-leg.
        self._start_sweep(wx, wy, wth)

    # ---- angular sweep (2-DOF: the lateral axis nothing else sees) --------
    def _start_sweep(self, wx, wy, wth):
        """Begin an angular sweep at a cell boundary, if one is due AND the map
        says there is something to measure against.

        Deliberately AFTER recentring rather than instead of it: recentring is
        closed-loop on the sensor and fixes the longitudinal axis properly, so
        the sweep arrives with only the axis it is actually good at left to
        measure. Running it first would have it fit a longitudinal error that
        was about to be driven out anyway.

        Two gates, and the second one matters more than it looks:

        `SWEEP_EVERY_N_LEGS` is a floor on the interval -- in cells driven,
        since a leg can now span several (see _advance_plan) -- not a target. Driving
        it purely on a leg count was measured to be fragile on tcc.maze -- WHICH
        cells the count lands on decides the outcome, so a cadence of 2 came out
        WORSE than 3 (12.8 cm vs 4.8 cm worst lateral error) purely by landing
        on cells with no geometry. Aliasing against the maze layout is not a
        property to tune against.

        So the second gate asks the map, from the BELIEVED pose, whether a sweep
        here would collect enough samples to fit at all -- a raycast-only probe,
        no rotations spent. Skipping the hopeless boundaries and sweeping every
        other one reaches the same accuracy as sweeping unconditionally while
        attempting it 13 times instead of 16 on this path.

        With CLOSEST_OBSTACLE_AS_REFERENCE on, the same interval gate leads to
        the closest-side-wall correction instead (_start_side_ref), and the
        sweep never runs.
        """
        if self._cells_since_sweep < SWEEP_EVERY_N_LEGS:
            return False
        if CLOSEST_OBSTACLE_AS_REFERENCE:
            return self._start_side_ref(wx, wy, wth)
        if not self._sweep_worth_trying(wx, wy, wth):
            # Leave the counter standing: the next boundary with geometry should
            # sweep immediately rather than wait out another full interval.
            return False
        self._cells_since_sweep = 0
        self.sweep = Sweep(self.view, self.origin, self.cell_length,
                           self.link, wx, wy, wth)
        return True

    def _sweep_worth_trying(self, wx, wy, wth) -> bool:
        """Would a sweep here collect enough samples to fit? Map only.

        Predicted against the BELIEVED pose and the original walls, which is the
        only information available before spending the rotations. It can be
        wrong in both directions -- the kart is not exactly where it is believed
        to be -- but it is right about the thing that matters: a boundary in open
        space where the map foresees nothing in range will not become fittable
        just because the kart turns around in it.
        """
        angles = plan_angles(self.view, self.origin, self.cell_length,
                             wx, wy, wth)
        usable = 0
        for offset in angles:
            predicted, _hit = predict_range(
                self.view, self.origin, self.cell_length,
                wx, wy, wth + math.radians(offset),
                RANGE_FIX_MAX_PREDICTED_M)
            if predicted is None:
                continue
            if is_trusted_reading(predicted * 100.0):
                usable += 1
        return usable >= SCAN_FIT_MIN_SAMPLES

    def _poll_sweep(self, drive_tel, wx, wy, wth):
        """Drive the sweep forward one frame, and apply what it found.

        What gets applied, and what does not:

        * LATERAL moves the anchor. It is the component no other corrector in
          the stack can observe (a forward ray cannot see sideways -- see
          recenter.py). The sweep has usually also driven the KART back onto
          the centre line by now (sweep.py, "Lateral recentring"), but that
          move was tracked by odometry like any other, so the belief still
          carries the fitted error and still needs this shift either way.
        * LONGITUDINAL is applied only when `scan_fit` reports it as
          observable. In an open corridor it is not, and the number the search
          returns there is arbitrary -- measured at -16 cm against a true
          +2 cm on tcc.maze. `Recenter` has already closed that axis against
          the sensor anyway, so skipping it costs nothing.
        """
        outcome = self.sweep.poll(drive_tel, wx, wy, wth)
        if outcome == "pending":
            self.status = f"varredura: {self.sweep.status}"
            return

        summary = self.sweep.summary()
        self.last_sweep_summary = summary
        lateral = self.sweep.applied_lateral
        longitudinal = self.sweep.applied_longitudinal
        if lateral is None:
            print(f"[INFO] varredura: {summary}")
            self.sweep = None
            return

        # Both components are expressed against the heading the sweep froze, so
        # they rotate into the world frame through that heading, not the current
        # one (the kart has turned since, and turned back).
        th = self.sweep.pth
        c, s = math.cos(th), math.sin(th)
        along = longitudinal or 0.0
        # lateral is +left of the heading; the left normal of (c, s) is (-s, c).
        dx = c * along - s * lateral
        dy = s * along + c * lateral
        self.mirror.shift_anchor(dx, dy)
        self.sweeps_done += 1
        gated = "" if longitudinal is not None else " (long nao observavel)"
        print(f"[INFO] varredura #{self.sweeps_done}: {summary}{gated}")

        # Re-place both views now, as the range fix does, so the correction is
        # visible at the boundary instead of appearing as a jump next frame.
        cx, cy = wx + dx, wy + dy
        self.mirror.apply_world_pose(cx, cy, wth)
        self.last_world_pose = (cx, cy, wth)
        self._update_kart_view(cx, cy, wth)
        self.sweep = None

    # ---- closest obstacle as lateral reference (replaces the sweep) -------
    def _start_side_ref(self, wx, wy, wth) -> bool:
        """Begin a lateral correction against the closest side wall, if either
        side has one the sensor can read (see runner/closest_obstacle.py).

        No usable side: nothing to do here, and the interval counter is left
        standing so the very next boundary tries again. The counter only
        restarts when a correction actually took a reading (_poll_side_ref)."""
        fix = ClosestObstacleFix(self.view, self.origin, self.cell_length,
                                 self.link, wx, wy, wth,
                                 last_turn=self._last_side_turn,
                                 turn_speed=self.turn_speed)
        if fix.refused is not None:
            self.last_sweep_summary = fix.summary()
            print(f"[INFO] referencia lateral: {fix.summary()} -- tento na proxima")
            return False
        self._last_side_turn = fix.turn
        self.side_ref = fix
        return True

    def _poll_side_ref(self, drive_tel, wx, wy, wth):
        """Drive the correction forward one frame. Once the kart is back on the
        leg heading, apply the anchor shift it measured -- AFTER the physical
        move, which odometry tracked and so did not fix the belief by itself."""
        outcome = self.side_ref.poll(drive_tel, wx, wy, wth)
        if outcome == "pending":
            self.status = f"referencia lateral: {self.side_ref.status}"
            return

        fix, self.side_ref = self.side_ref, None
        summary = fix.summary()
        self.last_sweep_summary = summary
        if not fix.succeeded:
            print(f"[INFO] referencia lateral: {summary} -- tento na proxima")
            return
        self._cells_since_sweep = 0
        self.sweeps_done += 1
        print(f"[INFO] referencia lateral #{self.sweeps_done}: {summary}")
        if fix.anchor_shift is None:
            return

        dx, dy = fix.anchor_shift
        self.mirror.shift_anchor(dx, dy)
        # Re-place both views now, as the sweep and the range fix do.
        cx, cy = wx + dx, wy + dy
        self.mirror.apply_world_pose(cx, cy, wth)
        self.last_world_pose = (cx, cy, wth)
        self._update_kart_view(cx, cy, wth)

    def _apply_range_fix(self, drive_tel, wx, wy, wth):
        """Pull the believed pose along its heading to match what the front
        sensor sees, when the map predicts a wall the beam should be hitting.

        This is the only correction to TRANSLATION anywhere in the stack.
        Heading is already ground truth (the kart closes it on the MPU6050
        gyro), but `d_center` comes from the encoders alone, so wheel slip
        accumulates in x/y without bound -- nothing else observes it.

        Only the longitudinal component is touched. One range reading observes
        one degree of freedom, and pointed down a corridor that DOF is the
        distance to the wall ahead; it says nothing about lateral offset within
        the corridor. So the correction is applied strictly along the heading.
        Recovering the lateral DOF needs a multi-heading profile (rotate,
        sample, repeat), which is a separate and much larger feature.

        Skipped while `slip.slipping` is set: the reading would be compared
        against a pose integrated from the very wheels that are slipping, so
        the residual would mix the error being measured with the error being
        corrected.
        """
        if self.mirror is None:
            return
        slip = drive_tel.get("slip") or {}
        correction = self.range_fix.evaluate(
            wx, wy, wth, drive_tel.get("front_distance_cm"),
            slipping=bool(slip.get("slipping")))
        if correction is None:
            return

        # `correction` is already signed to be ADDED along the heading (moving
        # forward shortens the predicted range, so it comes out as -residual --
        # see range_fix's module docstring).
        dx = math.cos(wth) * correction
        dy = math.sin(wth) * correction
        self.mirror.shift_anchor(dx, dy)
        print(f"[INFO] range-fix: {correction:+.3f} m "
              f"(previsto {self.range_fix.last_predicted_m:.2f} m, "
              f"medido {self.range_fix.last_measured_m:.2f} m)")

        # Re-place both views immediately, so the jump happens visibly at the
        # cell boundary instead of appearing as an unexplained discontinuity on
        # the next telemetry frame.
        cx, cy = wx + dx, wy + dy
        self.mirror.apply_world_pose(cx, cy, wth)
        self.last_world_pose = (cx, cy, wth)
        self._update_kart_view(cx, cy, wth)

    def _update_kart_view(self, wx, wy, wth):
        rf, cf = cop.world_to_cell(self.origin, self.cell_length, wx, wy)
        px, py = self.view.center_f(rf, cf)
        self.kart_view.set_offset(px, py)
        self.kart_view.set_rotation(world_theta_to_grid_heading(wth))

    def _leg_target(self, to_cell):
        """The absolute destination for a leg ending on `to_cell`, in the two
        frames `Leg` needs it in: (world_xy, odom_xyth_deg).

        The world pair is the cell's exact center (`cell_to_world` of an integer
        cell IS the center -- see its docstring and `world_to_cell`, its exact
        inverse). The odom triple is that same point expressed in the kart's
        odometry frame, which is the only frame the kart's `goto` understands.

        Recomputed per leg rather than cached with the plan, because the
        transform depends on `mirror.initial`, and `range_fix` moves that
        anchor mid-run (`shift_anchor`). A target converted once, before a
        correction landed, would aim at where the cell center USED to be
        believed to be -- quietly re-introducing exactly the error the
        correction just removed.

        Returns (None, None) when the conversion is not available yet (no
        telemetry reference), which makes `Leg` fall back to the relative
        turn-then-drive hop rather than aim at an unknown point.
        """
        tx, ty, _ = cop.cell_to_world(self.origin, self.cell_length, to_cell)
        if self.mirror is None or self.odom_ref is None:
            return None, None
        # The heading is commanded too: goto's third phase settles it gyro-
        # closed when a gyro is present, which beats this side's open-loop
        # turn chunks at the same job. `to_cell` may be several cells down a
        # straight run, so it is the run's heading, not a neighbour's.
        heading = run_heading(self.plan_path[self.plan_index], to_cell)
        wth_target = grid_heading_to_world_theta(heading)
        ox, oy, oth = self.mirror.odom_target(self.odom_ref, tx, ty, wth_target)
        return (tx, ty), (ox, oy, oth)

    def _advance(self, drive_tel, wx, wy, wth):
        # A recentring session owns the kart while it runs: it is issuing its
        # own drive_distance commands, so the next leg must not be started on
        # top of them. Checked before anything else for that reason.
        if self.recenter is not None:
            self._poll_recenter(drive_tel, wx, wy, wth)
            return
        # Same exclusivity as recentring, and for the same reason: a sweep is
        # issuing its own turns, so no leg may start on top of them.
        if self.sweep is not None:
            self._poll_sweep(drive_tel, wx, wy, wth)
            return
        if self.side_ref is not None:
            self._poll_side_ref(drive_tel, wx, wy, wth)
            return

        # Stopped short of a suspect: parked until the steady check settles
        # it, and only then the usual arrival work (recentre -> sweep). Before
        # the replan on purpose -- a confirmed obstacle is what asks for it.
        if self.leg is None and self._suspect is not None:
            if self._hold_for_suspect():
                self._start_recenter(wx, wy, wth)
            return

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
            # A straight stretch is one leg: one goto to the far centre, no
            # stop on the cells in between (see LEG_MAX_CELLS).
            to = self.plan_path[straight_run_end(self.plan_path, self.plan_index,
                                                 LEG_MAX_CELLS)]

            # Gate AFTER recentring/sweep (they returned above) and the replan,
            # so the operator releases the leg that will really be driven, from
            # the corrected pose. The goal check above is before this on
            # purpose: nothing to confirm once the run is over.
            if self.should_wait and not self._step_released:
                self._waiting_to_step = True
                self.status = f"[PASSO] em {frm} -- aperte ENTER para ir a {to}"
                return
            self._waiting_to_step = False
            self._step_released = False

            target_world, odom_target = self._leg_target(to)
            self.leg = Leg(frm, to, wth, self.cell_length,
                           self.turn_speed, self.drive_speed,
                           target_world=target_world,
                           odom_target=odom_target)

            self.status = f"{frm} -> {to}"

            return

        outcome = self.leg.poll(self.link, drive_tel, wx, wy)
        if outcome == "pending":
            return
        
        if outcome == "arrived":
            self._advance_plan(self.leg.n_cells)
            self.leg = None
            # The kart has driven past whatever it decided to ignore.
            self._dismissed.clear()
            # Cut short in front of a suspect: the hold at the top of this
            # method takes over, and recentres once the suspect is settled.
            if self._suspect is not None:
                return
            # Recentre HERE, between legs: the kart is stationary, the odometry
            # has settled, and the next leg is only built once this finishes --
            # so it starts from the corrected position.
            self._start_recenter(wx, wy, wth)
            return

        # "blocked" -- a holding state, not a dead end: _reconsider_live() will
        # keep re-planning, so this recovers by itself either way. A recorded
        # obstacle gets routed around on the next try; one the sensor never saw
        # (a stall, a slipped wheel) just gets retried until it goes.
        reason = "obstaculo a frente" if self.leg.guard_seen else "parou antes do previsto"
        self.status = (f"[INTERROMPIDO] {self.leg.from_cell} -> {self.leg.to_cell} "
                       f"({reason}, faltou {self.leg.shortfall:.2f} m). "
                       f"Replanejando sozinho...")
        print(f"[WARN] {self.status}")
        self.state = "blocked"
        guard_seen = self.leg.guard_seen
        self.leg = None
        # A leg that came up short is the case where the pose is MOST suspect --
        # the kart stopped somewhere nobody planned -- and it used to be the one
        # case that got no correction at all, after which `_reconsider_live`
        # replanned from that uncorrected pose.
        #
        # Only when the front guard did NOT trip, though. A guard stop means
        # something unmapped is in the beam, so the reading describes THAT, not
        # the mapped wall the correction raycasts against; range_fix's own
        # MAX_CORRECTION ceiling would mostly reject it, and "mostly" is not a
        # guarantee worth leaning on.
        if not guard_seen:
            self._apply_range_fix(drive_tel, wx, wy, wth)

    # ---- rendering ---------------------------------------------------------
    def hud_lines(self):
        video_line = (f"video: {self.camera.video_status()}   "
                      f"cam pan={self.camera.pan_pct}% tilt={self.camera.tilt_pct}%   "
                      f"(arrows=cam, C=centraliza)", 14, styles.HUDColors.text_dim)
        if self.phase == "setup":
            return [
                (self.status, 18, styles.HUDColors.text),
                video_line,
                ("L-click start   R-click goal   digite 'ok' quando o kart "
                 "REAL estiver posicionado", 16, styles.HUDColors.text_dim),
            ]
        color = {"running": styles.HUDColors.text,
                "idle": styles.HUDColors.text_ok,
                "waiting": styles.HUDColors.text_fail,
                "blocked": styles.HUDColors.text_fail}.get(
            self.state, styles.HUDColors.text)
        second = ""
        if self.last_world_pose is not None:
            x, y, th = self.last_world_pose
            second = f"kart em ({x:+.2f}, {y:+.2f})  heading={math.degrees(th):+.0f} deg"
        if self.obstacles_found:
            second += f"   |  {self.obstacles_found} obstaculo(s) no maze ao vivo"
        third = "L-click novo start   R-click novo goal"
        if self._waiting_to_step:
            third += "   |  ENTER = proximo passo"
        if self.state in ("blocked", "waiting"):
            third += "   (replanejando sozinho; 'resume' esquece tudo de uma vez)"

        # The range fix moves the kart's believed position on its own. Showing
        # what it did -- and, just as importantly, why it declined -- keeps it
        # from being an invisible process nudging the pose behind the operator's
        # back. Slip is on the same line because it is the main reason the fix
        # holds off.
        if self.recenter is not None:
            fix_line = f"recentrando: {self.recenter.status}"
        elif self.sweep is not None:
            fix_line = f"varredura: {self.sweep.status}"
        elif self.side_ref is not None:
            fix_line = f"referencia lateral: {self.side_ref.status}"
        elif self.last_sweep_summary:
            # The sweep is the only thing correcting the lateral axis, and it
            # runs every few legs rather than continuously, so its last result
            # stays on screen until the next one replaces it.
            fix_line = f"varredura ({self.sweeps_done}x): {self.last_sweep_summary}"
        else:
            fix_line = self.range_fix.status_line()
        if self._last_slip.get("slipping"):
            fix_line += (f"   |  DERRAPANDO "
                         f"(res {self._last_slip.get('residual_dps', 0):+.0f} deg/s)")
        elif self._last_slip.get("events"):
            fix_line += f"   |  {self._last_slip['events']} derrapagem(ns)"

        return [
            (self.status, 18, color),
            video_line,
            (second, 16, styles.HUDColors.text_dim),
            (fix_line, 14, styles.HUDColors.text_dim),
            (third, 16, styles.HUDColors.panel_line),
        ]

    def render(self,track_gui_keys=False,track_guys_str=None):
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

        _hud_lines = self.hud_lines()
        # _hud_lines.append((" "*10,10,pr.WHITE))
        _fmt_str = "Command: "
        if track_gui_keys:
            _fmt_str += track_guys_str
            
        _fmt_current_cmd = (_fmt_str,22,pr.WHITE)
        _hud_lines.append(_fmt_current_cmd)
        self.hud.render(_hud_lines)
        
        pr.end_drawing()

    def stop(self):
        self.camera.stop()
        if self.link is not None:
            self.link.stop()
