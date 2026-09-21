"""
One edge of the path: turn to face `to_cell` (if needed) then drive exactly
one cell length, tracked against live telemetry.
"""
import math
import time

from shadow import KartLink, wrap, grid_heading_to_world_theta

from src.solver import heading_between


class Leg:
    """`poll()` is called every frame with the latest drive-telemetry dict and
    the kart's current world pose. It issues the WS command itself the first
    time it runs a phase and thereafter only watches telemetry for that
    phase's completion -- it never blocks the render loop.

    Success is judged by geometry, not by catching a flag on the exact tick it
    pulses: `guard_blocked` only enriches the failure message (front obstacle
    vs. a plain stall) while the decisive test is how far the kart actually
    travelled versus the intended cell length. A single missed telemetry
    sample can't defeat that.

    A `goal_active` reading of False is only trusted as "the move finished"
    AFTER we've actually seen it go True first. Right after `link.turn()` /
    `link.drive_distance()` is issued, the very next frame's telemetry can
    still be a STALE sample from before the server even processed the
    command -- the DriveController only picks it up on its own ~20 Hz tick,
    on top of the WS round trip -- so a naive "not active -> done" check would
    judge the move using a pose that hasn't moved from its start point yet
    (this is exactly what produced a bogus "travelled 0 m" interruption while
    the real kart went on to complete the move server-side undisturbed).
    `_active_by` is a short real-time fallback for the pathological case where
    a move is so brief telemetry never catches the True pulse at all.

    A turn wider than `turn_max_angle_step` is not issued as one command:
    it is sliced into successive chunks of at most that many degrees (the
    last one being the remainder, i.e. `delta % step`), each issued only
    once the previous chunk's `goal_active` has gone back down. Slewing in
    bounded steps keeps the kart's real rotation closer to the commanded one
    than a single wide sweep does, and lets the front sensor sample along
    the way instead of only at the two extremes.
    """

    TURN_TOLERANCE_DEG = 3.0     # skip an in-place turn this small
    ARRIVE_TOLERANCE_M = 0.03    # "close enough" to a full cell length
    START_GRACE_S = 0.25         # max time to wait for goal_active's True edge

    def __init__(self, from_cell, to_cell, current_world_theta, cell_length,
                turn_speed=0.4,
                drive_speed=0.2,
                turn_max_angle_step= 90.0    #deg
        ):
        self.from_cell = from_cell
        self.to_cell = to_cell
        self.cell_length = cell_length
        self.turn_speed = turn_speed
        self.drive_speed = drive_speed
        self.turn_max_angle_step = turn_max_angle_step

        grid_heading = heading_between(from_cell, to_cell)
        self.target_theta = grid_heading_to_world_theta(grid_heading)
        self._turn_delta = math.degrees(wrap(self.target_theta - current_world_theta))
        self._turn_left = self._turn_delta   # signed degrees still to be issued
        self.phase = "drive" if abs(self._turn_delta) <= self.TURN_TOLERANCE_DEG else "turn"

        self._issued = False
        self._seen_active = False
        self._active_by = None
        self._turn_chunk = 0.0
        self._drive_origin = None
        self.guard_seen = False
        self.shortfall = None

    def _wait_for_start(self, drive_tel: dict) -> bool:
        """True while still waiting to confirm the server actually started the
        move just issued -- i.e. `goal_active`'s True edge, or the grace period
        elapsing (telemetry too slow/coarse to have caught it)."""
        if drive_tel.get("goal_active"):
            self._seen_active = True
        if self._seen_active:
            return False
        return time.monotonic() < self._active_by

    def _next_turn_chunk(self) -> float:
        """Signed degrees for the next in-place turn command: a whole
        `turn_max_angle_step` while that much is left, then the remainder
        (`abs(self._turn_left) % turn_max_angle_step`) on the final chunk.
        A non-positive step means "no limit" -- turn it all at once."""
        left = self._turn_left
        step = self.turn_max_angle_step
        if step is None or step <= 0:
            return left
        return math.copysign(min(step, abs(left)), left)

    def poll(self, link: KartLink, drive_tel: dict, wx: float, wy: float) -> str:
        """Returns "pending", "arrived" or "blocked"."""
        if self.phase == "turn":
            if not self._issued:
                self._turn_chunk = self._next_turn_chunk()
                link.turn(self._turn_chunk, self.turn_speed)
                self._issued = True
                self._seen_active = False
                self._active_by = time.monotonic() + self.START_GRACE_S
                return "pending"
            if self._wait_for_start(drive_tel):
                return "pending"
            if drive_tel.get("goal_active"):
                return "pending"
            # Chunk finished. The server's front guard only vetoes NET-FORWARD
            # duty (see DriveController._front_blocked), never a pure in-place
            # turn, so there is no "blocked" outcome for this phase -- any
            # residual heading error self-corrects on the next leg's fresh
            # delta computation.
            self._turn_left -= self._turn_chunk
            self._issued = False
            if abs(self._turn_left) > self.TURN_TOLERANCE_DEG:
                return "pending"   # re-issues the next chunk on the next frame
            self.phase = "drive"
            return "pending"

        # phase == "drive"
        if not self._issued:
            self._drive_origin = (wx, wy)
            link.drive_distance(self.cell_length, self.drive_speed)
            self._issued = True
            self._seen_active = False
            self._active_by = time.monotonic() + self.START_GRACE_S
            self.guard_seen = False
            return "pending"
        if drive_tel.get("guard_blocked"):
            self.guard_seen = True
        if self._wait_for_start(drive_tel):
            return "pending"
        if drive_tel.get("goal_active"):
            return "pending"

        ox, oy = self._drive_origin
        traveled = math.hypot(wx - ox, wy - oy)
        self.shortfall = self.cell_length - traveled
        return "blocked" if self.shortfall > self.ARRIVE_TOLERANCE_M else "arrived"
