"""
Drive the kart onto the centre of its cell after a camera fix has corrected the
belief. Polled once per frame, like `Recenter`.

Why this needs its own move at all
----------------------------------
The kart cannot translate sideways. A lateral offset is removed by turning
toward the centre, driving, and turning back -- which is exactly what the
server's `goto` does (turn1 -> drive -> turn2). Sent straight from where the
kart stands, though, a mostly-lateral offset makes turn1 and turn2 ~90 deg each,
in place, off-centre, with a 0.28 m kart in a 0.30 m cell: the very turns that
put 1-5 deg of heading error into the run in the first place.

So the default ("backoff") first reverses CAMERA_FIX_BACKOFF_M along the current
heading -- a straight move, heading held on the gyro -- and only then sends the
goto. From 12 cm back, a 3 cm offset is a ~15 deg turn1. The cell behind must
be open for that (it is, almost always: the kart just came from it); if it is
not, the plain goto is used.

The goto target is converted to the odometry frame AT THE MOMENT IT IS SENT,
from the live anchor -- the same rule as Leg: the anchor was just moved by the
camera fix, and a target converted before that would aim at the old centre.

Outcomes: "pending", "done" (within CAMERA_FIX_MOVE_ARRIVE_M of the centre by
the belief), "blocked" (the goto ended further than that -- front guard, stall).
"""
from __future__ import annotations

import math
import time
from typing import Optional, Tuple

from runner.constants import (
    CAMERA_FIX_BACKOFF_M, CAMERA_FIX_MOVE_ARRIVE_M, CAMERA_FIX_MOVE_MODE,
    RECENTER_SETTLE_S, RECENTER_SPEED, RECENTER_START_GRACE_S,
)

import Coppelia.helpers as cop


def cell_behind(cell: Tuple[int, int], theta: float) -> Tuple[int, int]:
    """The neighbour behind `cell` for a kart facing world heading `theta`.
    Columns grow with +X, rows with -Y (Coppelia/helpers/cell_to_world.py)."""
    dc = int(round(math.cos(theta)))
    dr = -int(round(math.sin(theta)))
    return cell[0] - dr, cell[1] - dc


class CenterMove:
    def __init__(self, link, mirror, odom_ref, view, origin, cell_length: float,
                 cell: Tuple[int, int], axis_theta: float,
                 mode: str = CAMERA_FIX_MOVE_MODE,
                 backoff_m: float = CAMERA_FIX_BACKOFF_M,
                 clock=time.monotonic):
        self.link = link
        self.mirror = mirror
        self.odom_ref = odom_ref
        self.cell = cell
        self.axis_theta = axis_theta
        self._clock = clock
        cx, cy, _ = cop.cell_to_world(origin, cell_length, cell)
        self.centre = (cx, cy)
        behind = cell_behind(cell, axis_theta)
        can_back = view.in_bounds(behind) and not view.is_wall(behind)
        self.mode = "backoff" if (mode == "backoff" and can_back) else "goto"
        self._phase = "recuando" if self.mode == "backoff" else "indo"
        self._issued = False
        self._seen_active = False
        self._active_by = 0.0
        self._settle_until = 0.0
        self.final_error_m: Optional[float] = None
        self.status = self._phase
        self.backoff_m = backoff_m

    def _wait_for_start(self, drive_tel) -> bool:
        if drive_tel.get("goal_active"):
            self._seen_active = True
        if self._seen_active:
            return False
        return self._clock() < self._active_by

    def _issue(self) -> None:
        self._issued = True
        self._seen_active = False
        self._active_by = self._clock() + RECENTER_START_GRACE_S

    def poll(self, drive_tel: dict, wx: float, wy: float) -> str:
        if self._phase == "recuando":
            if not self._issued:
                self.link.drive_distance(-self.backoff_m, RECENTER_SPEED)
                self._issue()
                self.status = f"recuando {self.backoff_m * 100:.0f} cm"
                return "pending"
            if self._wait_for_start(drive_tel) or drive_tel.get("goal_active"):
                return "pending"
            self._phase = "indo"
            self._issued = False
            return "pending"
        if self._phase == "indo":
            if not self._issued:
                ox, oy, oth = self.mirror.odom_target(self.odom_ref, self.centre[0],
                                                      self.centre[1], self.axis_theta)
                self.link.goto(ox, oy, oth)
                self._issue()
                self.status = "indo ao centro da celula"
                return "pending"
            if self._wait_for_start(drive_tel) or drive_tel.get("goal_active"):
                return "pending"
            self._phase = "assentando"
            self._settle_until = self._clock() + RECENTER_SETTLE_S
            self.status = "assentando"
            return "pending"
        if self._phase == "assentando":
            if self._clock() < self._settle_until:
                return "pending"
            self.final_error_m = math.hypot(self.centre[0] - wx, self.centre[1] - wy)
            self._phase = "fim"
            if self.final_error_m > CAMERA_FIX_MOVE_ARRIVE_M:
                self.status = f"parou a {self.final_error_m * 100:.1f} cm do centro"
                return "blocked"
            self.status = f"centralizado ({self.final_error_m * 100:.1f} cm)"
            return "done"
        return "done" if (self.final_error_m or 0.0) <= CAMERA_FIX_MOVE_ARRIVE_M else "blocked"
