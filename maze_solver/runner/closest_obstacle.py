"""
Lateral correction against the closest side obstacle -- the alternative to the
angular sweep, selected by CLOSEST_OBSTACLE_AS_REFERENCE.

The idea
--------
Recenter closes the longitudinal axis on the wall AHEAD. The lateral axis is
the same problem turned 90 degrees: face a wall that sits on the orthogonal
axis, and the forward sensor measures the lateral offset directly -- no fit,
no profile, one reading.

    kart at (r, c) heading along the leg
      -> pick a side whose closest cell on the orthogonal axis is a wall
      -> turn +-90 to face it, settle, read
      -> error = measured - expected        (expected: map raycast from the
                                             believed pose, like Recenter)
      -> |alpha * error| > CLOSEST_OBSTACLE_MINIMAL_DELTA_CM ?
             drive alpha * error            (forward = toward the wall)
             settle, read again
      -> anchor shift from the LAST reading
      -> turn back to the leg heading

The anchor correction, and why it comes from a second reading
-------------------------------------------------------------
Odometry tracks the move, so driving the kart moves the BELIEF by the same
amount: the gap between where the kart is and where it is believed to be does
not change by moving (see recenter.py's "the belief moves with the kart"). The
move puts the KART back where the belief was; the belief still has to be put
back on the kart, which is a shift of the anchor.

The residual to shift by is re-measured after the move rather than assumed:

    residual = measured_after - expected_after       (expected from the
                                                      believed pose NOW)
    anchor  += -residual along the beam

so a move that slipped or came up short is accounted for by what the sensor
sees, not by what was commanded. If that second reading is unusable, the first
residual is used: with a perfectly tracked move it is exactly the same number.

Which side
----------
Only walls from the .maze file count as a reference, never obstacles
discovered at runtime -- same trust boundary as range_fix (a discovered
obstacle is placed through the pose, so localising against it would confirm
whatever error placed it, and its real size is unknown anyway). A discovered
obstacle or the map edge before any wall means no reference on that side.

The wall must also be one the sensor can actually read from here: its expected
range has to sit inside the sensor's valid band (blind zone .. trust ceiling).
Turning to face a wall the sensor cannot measure would cost two rotations for
nothing. NOTE the geometry this rules out: with 0.30 m cells and the sensor
SENSOR_FORWARD_OFFSET_M ahead of the centre, a wall in the ADJACENT cell is
expected at 0.15 - 0.13 = 2 cm -- inside the 3 cm blind zone. A wall with one
open cell in between (32 cm) is readable.

When both sides qualify, the side NOT taken last time is chosen, so successive
corrections alternate +90/-90 and a systematic turn bias does not keep
stacking up in the same direction. When only one side qualifies it is used
even if it repeats the last one: a correction beats none.

`poll()` returns "pending" or "done"; `succeeded` says whether a usable
reading was actually taken (what the caller restarts its interval on), and
`anchor_shift` holds the (dx, dy) for the caller to apply, or None.
"""
from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple

import Coppelia.helpers as cop
from shadow import world_theta_to_grid_heading
from src.solver import DIRECTIONS

from runner.constants import (
    CLOSEST_OBSTACLE_ALPHA, CLOSEST_OBSTACLE_MINIMAL_DELTA_CM,
    RANGE_FIX_ALIGN_TOLERANCE_DEG, RANGE_FIX_MAX_PREDICTED_M,
    RANGE_FIX_MAX_TRUSTED_M, RANGE_FIX_MIN_CORRECTION_M,
    RECENTER_MAX_STEP_M, RECENTER_SPEED,
    SCAN_FIT_MAX_LATERAL_M, SENSOR_MIN_RANGE_CM,
    SWEEP_MAX_LOOKS_PER_ANGLE, SWEEP_SAMPLES_PER_ANGLE, SWEEP_SETTLE_S,
    SWEEP_START_GRACE_S,
)
from runner.range_fix import is_trusted_reading, predict_range

HALF_PI = math.pi / 2.0


def _wrap(t: float) -> float:
    """Radians into (-pi, pi]."""
    return (t + math.pi) % (2 * math.pi) - math.pi


def snap_to_axis(wth: float) -> float:
    """The grid axis (world theta, a multiple of 90 deg) closest to wth."""
    return _wrap(round(wth / HALF_PI) * HALF_PI)


def _step_for(theta: float) -> Tuple[int, int]:
    """(dr, dc) of one cell along a grid-axis world heading."""
    heading = world_theta_to_grid_heading(theta)
    axis = (round(heading / 90.0) * 90.0) % 360.0
    for dr, dc, grid_heading in DIRECTIONS:
        if grid_heading == axis:
            return dr, dc
    raise ValueError(f"not a grid axis: {math.degrees(theta):.1f} deg")


def side_reference(view, origin, cell_length, wx, wy, theta):
    """The closest .maze wall straight along `theta` from the kart's cell, and
    the range the sensor should read to it from the believed pose.

    Returns (cell, expected_m, None) when it is usable, or
    (cell_or_None, expected_or_None, reason) when it is not."""
    rf, cf = cop.world_to_cell(origin, cell_length, wx, wy)
    here = (round(rf), round(cf))
    dr, dc = _step_for(theta)
    cell = here
    while True:
        cell = (cell[0] + dr, cell[1] + dc)
        if not view.in_bounds(cell):
            return None, None, "borda do mapa"
        if view.is_original_wall(cell):
            break
        if view.is_wall(cell):
            # Discovered at runtime: unknown size, and placed through the very
            # pose this would correct -- never a reference (see docstring).
            return cell, None, f"obstaculo descoberto em {cell}"

    expected, hit = predict_range(view, origin, cell_length, wx, wy, theta,
                                  RANGE_FIX_MAX_PREDICTED_M)
    if expected is None or hit != cell:
        return cell, None, f"raio nao chega em {cell}"
    if expected * 100.0 < SENSOR_MIN_RANGE_CM:
        return cell, expected, (f"{cell} a {expected * 100:.1f} cm, dentro da "
                                f"zona cega ({SENSOR_MIN_RANGE_CM:.0f} cm)")
    if expected > RANGE_FIX_MAX_TRUSTED_M:
        return cell, expected, (f"{cell} a {expected * 100:.0f} cm, alem do "
                                f"limite de confianca")
    return cell, expected, None


def pick_side(view, origin, cell_length, wx, wy, wth, last_turn: int):
    """Which way to turn: +1 (+90, ccw) or -1 (-90, cw), with the reference
    cell; or (None, None, reasons) when neither side has a usable one.

    Prefers the opposite of `last_turn` (0 = no history, try +90 first)."""
    axis = snap_to_axis(wth)
    usable = {}
    reasons: List[str] = []
    for sign in (+1, -1):
        cell, _expected, why = side_reference(view, origin, cell_length,
                                              wx, wy, axis + sign * HALF_PI)
        label = "+90" if sign > 0 else "-90"
        if why is None:
            usable[sign] = cell
        else:
            reasons.append(f"{label}: {why}")
    order = (-last_turn, last_turn) if last_turn else (+1, -1)
    for sign in order:
        if sign in usable:
            return sign, usable[sign], reasons
    return None, None, reasons


class ClosestObstacleFix:
    """One lateral correction, polled once per frame (non-blocking, like
    Recenter and Sweep). See the module docstring.

    Phases:  turn_out -> settle -> sample -> [move -> settle -> sample]
             -> turn_back -> settle -> done
    """

    def __init__(self, view, origin, cell_length, link, wx, wy, wth,
                 last_turn: int = 0, turn_speed: float = 0.35,
                 clock=time.monotonic):
        self._clock = clock
        self.view = view
        self.origin = origin
        self.cell_length = cell_length
        self.link = link
        self.turn_speed = turn_speed

        # The heading to come back to: the grid axis the leg runs on, not the
        # raw heading -- returning to the axis also takes out whatever turn
        # error the kart arrived with.
        self.axis = snap_to_axis(wth)
        self.turn, self.ref_cell, reasons = pick_side(
            view, origin, cell_length, wx, wy, wth, last_turn)
        self.refused: Optional[str] = (
            None if self.turn is not None
            else "sem referencia lateral (" + "; ".join(reasons) + ")")

        self.succeeded = False
        self.anchor_shift: Optional[Tuple[float, float]] = None
        self.first_error_m: Optional[float] = None
        self.residual_m: Optional[float] = None
        self.moved_m: Optional[float] = None
        self.status = self.refused or "girando para a referencia"
        self._result = ""

        self._phase = "turn_out"
        self._issued = False
        self._seen_active = False
        self._active_by = 0.0
        self._settle_until = 0.0
        self._after_settle = "sample"
        self._readings: List[float] = []
        self._looks = 0

    # -- helpers -----------------------------------------------------------
    def _wait_for_start(self, drive_tel) -> bool:
        """Same stale-telemetry guard as Leg/Recenter/Sweep."""
        if drive_tel.get("goal_active"):
            self._seen_active = True
        if self._seen_active:
            return False
        return self._clock() < self._active_by

    def _armed(self) -> None:
        self._issued = True
        self._seen_active = False
        self._active_by = self._clock() + SWEEP_START_GRACE_S

    def _move_finished(self, drive_tel) -> bool:
        if self._wait_for_start(drive_tel) or drive_tel.get("goal_active"):
            return False
        self._issued = False
        return True

    def _settle_then(self, phase: str) -> None:
        self._phase = "settle"
        self._after_settle = phase
        self._settle_until = self._clock() + SWEEP_SETTLE_S

    def _begin_return(self) -> None:
        self._phase = "turn_back"
        self._issued = False

    # -- main poll ---------------------------------------------------------
    def poll(self, drive_tel: dict, wx: float, wy: float, wth: float) -> str:
        if self.refused is not None:
            return "done"
        if self._phase == "turn_out":
            return self._do_turn(drive_tel, wth, self.axis + self.turn * HALF_PI,
                                 then="sample")
        if self._phase == "settle":
            if self._clock() < self._settle_until:
                return "pending"
            if self._after_settle == "done":
                self.status = self._result
                self._phase = "done"
                return "done"
            self._phase = self._after_settle
            self._readings, self._looks = [], 0
            return "pending"
        if self._phase == "sample":
            return self._do_sample(drive_tel, wx, wy, wth)
        if self._phase == "move":
            if self._move_finished(drive_tel):
                self._settle_then("resample")
            return "pending"
        if self._phase == "resample":
            return self._do_sample(drive_tel, wx, wy, wth)
        if self._phase == "turn_back":
            return self._do_turn(drive_tel, wth, self.axis, then="done")
        return "done"

    def _do_turn(self, drive_tel, wth, target, then) -> str:
        if not self._issued:
            self.link.turn(math.degrees(_wrap(target - wth)), self.turn_speed)
            self._armed()
            self.status = ("voltando ao rumo da perna" if then == "done" else
                           f"girando {self.turn * 90:+d} deg para {self.ref_cell}")
            return "pending"
        if self._move_finished(drive_tel):
            self._settle_then(then)
        return "pending"

    def _do_sample(self, drive_tel, wx, wy, wth) -> str:
        """Average a few readings (same bounded scheme as the sweep), then
        judge them against the map from the pose believed right now."""
        if (drive_tel.get("slip") or {}).get("slipping"):
            return "pending"
        self._looks += 1
        dist_cm = drive_tel.get("front_distance_cm")
        if is_trusted_reading(dist_cm):
            self._readings.append(float(dist_cm) / 100.0)
        if (len(self._readings) < SWEEP_SAMPLES_PER_ANGLE
                and self._looks < SWEEP_MAX_LOOKS_PER_ANGLE):
            return "pending"

        error = self._error(wx, wy, wth)
        if self._phase == "sample":
            self._judge_first(error, wth)
        else:
            self._judge_after_move(error, wth)
        return "pending"

    def _error(self, wx, wy, wth) -> Optional[float]:
        """measured - expected, metres along the beam (+ = further from the
        wall than believed), or None when it cannot be judged."""
        if not self._readings:
            return None
        off = abs(math.degrees(_wrap(wth - (self.axis + self.turn * HALF_PI))))
        if off > RANGE_FIX_ALIGN_TOLERANCE_DEG:
            return None
        expected, hit = predict_range(self.view, self.origin, self.cell_length,
                                      wx, wy, wth, RANGE_FIX_MAX_PREDICTED_M)
        if expected is None or hit != self.ref_cell:
            return None
        measured = sum(self._readings) / len(self._readings)
        return measured - expected

    def _judge_first(self, error, wth) -> None:
        if error is None:
            self._result = f"sem leitura utilizavel de {self.ref_cell}"
            self._begin_return()
            return
        if abs(error) > SCAN_FIT_MAX_LATERAL_M:
            # Most of a cell off is a wrong-wall hypothesis, not drift: the
            # beam caught something the map does not have.
            self._result = (f"erro {error * 100:+.1f} cm grande demais em "
                            f"{self.ref_cell} (parede errada?)")
            self._begin_return()
            return

        self.succeeded = True
        self.first_error_m = error
        move = CLOSEST_OBSTACLE_ALPHA * error
        if abs(move) * 100.0 <= CLOSEST_OBSTACLE_MINIMAL_DELTA_CM:
            self._finish(error, wth, note="sem movimento")
            return
        move = max(-RECENTER_MAX_STEP_M, min(RECENTER_MAX_STEP_M, move))
        self.moved_m = move
        self.link.drive_distance(move, RECENTER_SPEED)
        self._armed()
        self._phase = "move"
        self.status = (f"{'avancando' if move > 0 else 'recuando'} "
                       f"{abs(move) * 100:.1f} cm (erro {error * 100:+.1f} cm)")

    def _judge_after_move(self, error, wth) -> None:
        if error is None or abs(error) > SCAN_FIT_MAX_LATERAL_M:
            # Second look unusable: a tracked move leaves the belief error
            # where it was, so the first residual is the right one to shift by.
            self._finish(self.first_error_m, wth, note="releitura falhou")
            return
        self._finish(error, wth, note=f"moveu {self.moved_m * 100:+.1f} cm")

    def _finish(self, residual, wth, note) -> None:
        """Turn the residual into the anchor correction, then head back."""
        self.residual_m = residual
        if abs(residual) >= RANGE_FIX_MIN_CORRECTION_M:
            # residual > 0: the kart is further from the wall than believed,
            # i.e. behind the belief along the beam -- pull the belief back.
            self.anchor_shift = (-residual * math.cos(wth),
                                 -residual * math.sin(wth))
            anchor = f"ancora {-residual * 100:+.1f} cm"
        else:
            anchor = "ancora mantida"
        self._result = (f"{self.turn * 90:+d} deg -> {self.ref_cell}: erro "
                        f"{self.first_error_m * 100:+.1f} cm, {note}, {anchor}")
        self._begin_return()

    # -- reporting ---------------------------------------------------------
    def summary(self) -> str:
        return self.refused or self._result or self.status
