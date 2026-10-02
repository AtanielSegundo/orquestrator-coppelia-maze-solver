"""
One edge of the path: put the kart on the CENTER of `to_cell`, tracked against
live telemetry.

Absolute target, not a relative hop
-----------------------------------
This module used to drive each edge as "turn by the heading delta, then
`drive_distance(cell_length)`". That is a purely RELATIVE move, and it has no
mechanism for removing the error it starts with. A kart sitting at
(c1_x + dx, c1_y + dy) instead of the center of cell 1 commands one cell length
along its heading and lands at (c2_x + dx, c2_y + dy): the offset is carried
through untouched, every leg adds its own fresh error on top, and nothing in
the loop ever subtracts any of it.

The two error components were not even treated alike. `Recenter` runs between
legs and closes the LONGITUDINAL component against the front sensor, but a
single forward-facing ray cannot observe lateral offset at all -- so the
crosswise component was corrected by nothing whatsoever and just accumulated
until the kart clipped a wall.

So a leg now names its destination absolutely: the world-frame center of
`to_cell`, converted into the kart's odometry frame, handed to the kart's own
`goto` (DriveController.goto_pose). The server runs it as face-the-point ->
drive-straight -> turn-to-final-heading with every phase recomputed from the
live odom pose, so the distance it drives is the distance to the TARGET rather
than a nominal cell length: enter the leg 3 cm short and 2 cm to the left and
both the turn and the distance absorb it. Error stops compounding per leg and
instead gets re-closed on every leg, in both axes.

What the target is, and what it is not
--------------------------------------
The target is the center of `to_cell` as the MAP defines it, composed through
the current anchor, so it is only ever as good as the belief about where the
kart started -- `goto` closes its loop on odometry, which still holds no
absolute reference of its own. This does not make odometry drift go away. It
makes the drift bounded by whatever corrects the BELIEF (range_fix shifting the
anchor) or the kart (`Recenter`), instead of compounding on top of it once per
leg. That is the whole claim, and it is worth being precise about it: the kart
is now wrong by the pose error, not by the pose error times the number of legs
driven.

The final heading is commanded too, as the leg's grid heading. `goto`'s third
phase is a gyro-closed turn when a gyro is present, which makes it a better
place to settle the heading than this module's own open chunking was.

What the absolute target does NOT replace: bounded turn steps
-------------------------------------------------------------
`goto` runs its turns through `DriveController._turn_to`, which issues the
whole delta as a single command. So handing a wide heading change straight to
`goto` would silently drop both things `turn_max_angle_step` is for -- a sweep
that overshoots more the wider it gets, and a front sensor that only samples
the two extremes of the turn. The chunking therefore survives the rewrite and
runs BEFORE the goto, in both modes; see `Leg._needs_pre_turn`.

Fallback
--------
A kart whose server predates `goto` would silently never move, so `relative=True`
restores the turn-then-drive path verbatim. It is also what runs when a caller
cannot supply the target in both frames (see `__init__`), and what the unit
tests drive to keep the legacy semantics pinned.
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
    vs. a plain stall) while the decisive test is where the kart actually ended
    up. A single missed telemetry sample can't defeat that.

    A `goal_active` reading of False is only trusted as "the move finished"
    AFTER we've actually seen it go True first. Right after the command is
    issued, the very next frame's telemetry can still be a STALE sample from
    before the server even processed it -- the DriveController only picks it up
    on its own ~20 Hz tick, on top of the WS round trip -- so a naive "not
    active -> done" check would judge the move using a pose that hasn't moved
    from its start point yet (this is exactly what produced a bogus
    "travelled 0 m" interruption while the real kart went on to complete the
    move server-side undisturbed). `_active_by` is a short real-time fallback
    for the pathological case where a move is so brief telemetry never catches
    the True pulse at all.

    A turn wider than `turn_max_angle_step` is not issued as one command, in
    EITHER mode: it is sliced into successive chunks of at most that many
    degrees, each issued only once the previous chunk's `goal_active` has gone
    back down. Slewing in bounded steps keeps the kart's real rotation closer
    to the commanded one than a single wide sweep does, and lets the front
    sensor sample along the way instead of only at the two extremes. The
    difference between the modes is only where the remainder goes: `relative`
    chunks all the way down to TURN_TOLERANCE_DEG and then drives, while
    absolute stops at one step and hands the rest to `goto`'s own gyro-closed
    turn1 (see `_needs_pre_turn`).

    After any pre-turn, the absolute mode's translation is ONE `goto` and the
    server sequences turn -> drive -> turn itself, so there is a single phase
    to wait on -- and `goal_active` stays True across all three of them (see
    DriveController.move_active, which ORs the in-flight goto in), so the same
    edge logic reads that sequence as one move.
    """

    TURN_TOLERANCE_DEG = 3.0     # skip an in-place turn this small
    ARRIVE_TOLERANCE_M = 0.03    # "close enough" to the target
    START_GRACE_S = 0.25         # max time to wait for goal_active's True edge

    def __init__(self, from_cell, to_cell, current_world_theta, cell_length,
                turn_speed=0.4,
                drive_speed=0.2,
                turn_max_angle_step= 90.0,   #deg
                target_world=None,           # (wx, wy): center of to_cell
                odom_target=None,            # (ox, oy, otheta_deg): same point
                relative=False,
        ):
        self.from_cell = from_cell
        self.to_cell = to_cell
        self.cell_length = cell_length
        self.turn_speed = turn_speed
        self.drive_speed = drive_speed
        self.turn_max_angle_step = turn_max_angle_step
        self.target_world = target_world
        self.odom_target = odom_target
        # Absolute mode needs the destination in BOTH frames: odom to command
        # the kart with, world to judge arrival against the pose the runner
        # tracks. Missing either, fall back rather than aim at nothing.
        self.relative = (relative or target_world is None
                         or odom_target is None)

        grid_heading = heading_between(from_cell, to_cell)
        self.target_theta = grid_heading_to_world_theta(grid_heading)
        self._turn_delta = math.degrees(wrap(self.target_theta - current_world_theta))
        self._turn_left = self._turn_delta   # signed degrees still to be issued
        if self.relative:
            self.phase = ("drive"
                          if abs(self._turn_delta) <= self.TURN_TOLERANCE_DEG
                          else "turn")
        else:
            # Pre-turn in bounded chunks when the heading change is wider than
            # one step, THEN goto. `goto`'s own turn phases go through
            # DriveController._turn_to, which issues the whole delta as a
            # single command -- so leaving a wide sweep to the server would
            # drop both things the chunking is for (see _needs_pre_turn).
            self.phase = "turn" if self._needs_pre_turn() else "goto"

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

    def _needs_pre_turn(self) -> bool:
        """Whether the absolute path should slew the heading itself before
        handing the leg to `goto`.

        Only when the pending change exceeds one step, because within a step
        there is nothing to slice and `goto`'s turn1 (gyro-closed, recomputed
        from the live pose) is the better way to make it.

        Beyond a step, two things are lost by letting the server sweep it as
        one command, and both were the reason the chunking exists:

        1. A single wide sweep overshoots more than the same rotation taken in
           bounded steps -- a skid turn's stiction and inertia are not what the
           heading PID's model assumes, and the error grows with the angle.
        2. The front sensor only samples where the kart actually dwells, so one
           180 deg sweep yields readings at the two extremes and nothing in
           between, starving `_scan_obstacle` of exactly the mid-turn looks
           that discover an obstacle off the current axis.

        After the pre-turn the remaining heading error is inside a step, so
        `goto`'s turn1 still gets the last word on it -- the chunking gets the
        kart close, the gyro-closed phase finishes the job.
        """
        step = self.turn_max_angle_step
        if step is None or step <= 0:
            return False                      # no limit asked for
        return abs(self._turn_left) > step

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
        if self.phase == "goto":
            return self._poll_goto(link, drive_tel, wx, wy)
        if self.phase == "turn":
            return self._poll_turn(link, drive_tel)
        return self._poll_drive(link, drive_tel, wx, wy)

    # -- absolute: one goto, sequenced server-side --------------------------
    def _poll_goto(self, link, drive_tel, wx, wy) -> str:
        if not self._issued:
            self._drive_origin = (wx, wy)
            ox, oy, oth = self.odom_target
            link.goto(ox, oy, oth)
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

        # Judged as distance REMAINING to the target, not as distance
        # travelled. That is the substantive difference from the relative path:
        # "it drove about a cell length" is a question whose right answer is
        # still compatible with being in the wrong place.
        tx, ty = self.target_world
        self.shortfall = math.hypot(tx - wx, ty - wy)
        return "blocked" if self.shortfall > self.ARRIVE_TOLERANCE_M else "arrived"

    # -- chunked in-place turn (both paths) --------------------------------
    def _poll_turn(self, link, drive_tel) -> str:
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
        if self.relative:
            if abs(self._turn_left) > self.TURN_TOLERANCE_DEG:
                return "pending"   # re-issues the next chunk on the next frame
            self.phase = "drive"
            return "pending"
        # Absolute: stop chunking once what is left fits in one step and let
        # `goto`'s turn1 finish it. Driving the remainder down to
        # TURN_TOLERANCE_DEG here instead would spend extra commands doing
        # worse work -- turn1 closes on the gyro against the live pose, while
        # these chunks are open-loop deltas off a heading read one frame ago.
        if self._needs_pre_turn():
            return "pending"
        self.phase = "goto"
        return "pending"

    def _poll_drive(self, link, drive_tel, wx, wy) -> str:
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
