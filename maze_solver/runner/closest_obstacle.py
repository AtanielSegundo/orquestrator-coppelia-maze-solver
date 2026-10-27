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
      -> [CLOSEST_OBSTACLE_ALWAYS_CHECKS_THE_TWO: the same on the other side]
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

With both sides visited, the shift found on the first side is already folded
into the pose the second side is judged from (the runner only applies the
total once the kart is back on its heading), so the two do not double-count.

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
nothing. Whether the ADJACENT cell's wall qualifies depends on the geometry:
it is expected at cell/2 - SENSOR_FORWARD_OFFSET_M, which is 5 cm at 0.30 m
cells and a 0.10 m offset but only 2 cm (blind zone) at 0.13 m.

A BETTER reference is a readable wall with exactly one empty cell between it
and the kart's cell (`SideRef.better`): far enough to sit well clear of the
blind zone, close enough for the cone to stay on that wall. NEXT_BEST_CLOSEST_
AS_REFERENCE uses it to postpone a correction to a cell ahead on the path that
has one (`find_next_best`; the runner does the scheduling).

When both sides qualify, the side NOT taken last time is chosen, so successive
corrections alternate +90/-90 and a systematic turn bias does not keep
stacking up in the same direction. When only one side qualifies it is used
even if it repeats the last one: a correction beats none. `force_side`
overrides the alternation (the side a postponed correction was planned for).

`poll()` returns "pending" or "done"; `succeeded` says whether a usable
reading was actually taken (what the caller restarts its interval on), and
`anchor_shift` holds the (dx, dy) for the caller to apply, or None.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import Coppelia.helpers as cop
from shadow import grid_heading_to_world_theta, world_theta_to_grid_heading
from src.solver import DIRECTIONS, heading_between

from runner.constants import (
    CLOSEST_OBSTACLE_ALPHA, CLOSEST_OBSTACLE_ALWAYS_CHECKS_THE_TWO,
    CLOSEST_OBSTACLE_MINIMAL_DELTA_CM,
    RANGE_FIX_ALIGN_TOLERANCE_DEG, RANGE_FIX_MAX_PREDICTED_M,
    RANGE_FIX_MAX_TRUSTED_M, RANGE_FIX_MIN_CORRECTION_M,
    RECENTER_MAX_STEP_M, RECENTER_SPEED,
    SCAN_FIT_MAX_LATERAL_M, SENSOR_MIN_RANGE_CM,
    SWEEP_MAX_LOOKS_PER_ANGLE, SWEEP_SAMPLES_PER_ANGLE, SWEEP_SETTLE_S,
    SWEEP_START_GRACE_S,
)
from runner.range_fix import is_trusted_reading, predict_range

HALF_PI = math.pi / 2.0
# Widest single turn command. Going from one side to the other is 180 deg;
# it is issued as two of these, the same bound Leg's turn chunking uses.
MAX_TURN_STEP_DEG = 90.0


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


def _order(last_turn: int) -> Tuple[int, int]:
    """Sides in order of preference: the opposite of the last turn first."""
    return (-last_turn, last_turn) if last_turn else (+1, -1)


@dataclass(frozen=True)
class SideRef:
    """What one side (+1 = +90 ccw, -1 = -90 cw) offers as a reference."""
    sign: int
    cell: Optional[Tuple[int, int]] = None      # the wall, if one was found
    expected: Optional[float] = None            # sensor range to it, metres
    gap: Optional[int] = None                   # empty cells in between
    reason: Optional[str] = None                # why not usable; None = usable

    @property
    def usable(self) -> bool:
        return self.reason is None

    @property
    def better(self) -> bool:
        """Readable, with exactly one empty cell between it and the kart."""
        return self.usable and self.gap == 1

    @property
    def label(self) -> str:
        return "+90" if self.sign > 0 else "-90"


def side_reference(view, origin, cell_length, wx, wy, axis, sign) -> SideRef:
    """The closest .maze wall on side `sign` of the grid axis `axis`, seen
    from the believed pose (wx, wy), and whether the sensor can read it."""
    theta = axis + sign * HALF_PI
    rf, cf = cop.world_to_cell(origin, cell_length, wx, wy)
    cell = (round(rf), round(cf))
    dr, dc = _step_for(theta)
    gap = 0
    while True:
        cell = (cell[0] + dr, cell[1] + dc)
        if not view.in_bounds(cell):
            return SideRef(sign, reason="borda do mapa")
        if view.is_original_wall(cell):
            break
        if view.is_wall(cell):
            # Discovered at runtime: unknown size, and placed through the very
            # pose this would correct -- never a reference (see docstring).
            return SideRef(sign, cell, gap=gap,
                           reason=f"obstaculo descoberto em {cell}")
        gap += 1

    expected, hit = predict_range(view, origin, cell_length, wx, wy, theta,
                                  RANGE_FIX_MAX_PREDICTED_M)
    if expected is None or hit != cell:
        return SideRef(sign, cell, gap=gap, reason=f"raio nao chega em {cell}")
    if expected * 100.0 < SENSOR_MIN_RANGE_CM:
        return SideRef(sign, cell, expected, gap,
                       f"{cell} a {expected * 100:.1f} cm, dentro da zona "
                       f"cega ({SENSOR_MIN_RANGE_CM:.0f} cm)")
    if expected > RANGE_FIX_MAX_TRUSTED_M:
        return SideRef(sign, cell, expected, gap,
                       f"{cell} a {expected * 100:.0f} cm, alem do limite "
                       f"de confianca")
    return SideRef(sign, cell, expected, gap)


def side_refs(view, origin, cell_length, wx, wy, wth) -> Dict[int, SideRef]:
    """Both sides of the grid axis closest to wth."""
    axis = snap_to_axis(wth)
    return {s: side_reference(view, origin, cell_length, wx, wy, axis, s)
            for s in (+1, -1)}


def pick_side(view, origin, cell_length, wx, wy, wth, last_turn: int):
    """Which way to turn: +1 (+90, ccw) or -1 (-90, cw), with the reference
    cell; or (None, None, reasons) when neither side has a usable one.

    Prefers the opposite of `last_turn` (0 = no history, try +90 first)."""
    refs = side_refs(view, origin, cell_length, wx, wy, wth)
    for sign in _order(last_turn):
        if refs[sign].usable:
            return sign, refs[sign].cell, []
    return None, None, [f"{r.label}: {r.reason}" for r in refs.values()]


def better_side(refs: Dict[int, SideRef], last_turn: int) -> Optional[int]:
    """The side offering a BETTER reference, alternation breaking a tie."""
    for sign in _order(last_turn):
        if refs[sign].better:
            return sign
    return None


def find_next_best(view, origin, cell_length, path, index: int,
                   max_cells: int, last_turn: int = 0):
    """First cell AHEAD on `path` (at most `max_cells` past path[index]) where
    the kart will have a better reference on one side.

    Judged from each cell's centre and the heading the kart ARRIVES there
    with -- the previous step's -- since that is the pose it stops in before
    the next leg turns. Map only, no rotations spent. Returns
    (path index, side) or None."""
    for i in range(index + 1, min(len(path), index + 1 + max_cells)):
        heading = heading_between(path[i - 1], path[i])
        if heading is None:
            continue
        x, y, _ = cop.cell_to_world(origin, cell_length, path[i])
        refs = side_refs(view, origin, cell_length, x, y,
                         grid_heading_to_world_theta(heading))
        side = better_side(refs, last_turn)
        if side is not None:
            return i, side
    return None


class ClosestObstacleFix:
    """One lateral correction, polled once per frame (non-blocking, like
    Recenter and Sweep). See the module docstring.

    Per side:  turn_out -> settle -> sample -> [move -> settle -> resample]
    Then:      turn_back -> settle -> done

    `both` defaults to CLOSEST_OBSTACLE_ALWAYS_CHECKS_THE_TWO: visit every
    side that has a usable reference, instead of only the preferred one. A
    side with none is never turned to.
    """

    def __init__(self, view, origin, cell_length, link, wx, wy, wth,
                 last_turn: int = 0, turn_speed: float = 0.35,
                 force_side: Optional[int] = None, both: Optional[bool] = None,
                 clock=time.monotonic):
        self._clock = clock
        self.view = view
        self.origin = origin
        self.cell_length = cell_length
        self.link = link
        self.turn_speed = turn_speed
        if both is None:
            both = CLOSEST_OBSTACLE_ALWAYS_CHECKS_THE_TWO

        # The heading to come back to: the grid axis the leg runs on, not the
        # raw heading -- returning to the axis also takes out whatever turn
        # error the kart arrived with.
        self.axis = snap_to_axis(wth)
        self.refs = side_refs(view, origin, cell_length, wx, wy, wth)
        usable = [s for s in _order(last_turn) if self.refs[s].usable]
        if force_side in usable:
            usable.remove(force_side)
            usable.insert(0, force_side)
        self.sides: List[int] = usable if both else usable[:1]
        self.refused: Optional[str] = None
        if not self.sides:
            self.refused = ("sem referencia lateral (" + "; ".join(
                f"{r.label}: {r.reason}" for r in self.refs.values()) + ")")
        self.last_side = self.sides[-1] if self.sides else 0

        self.succeeded = False
        self.anchor_shift: Optional[Tuple[float, float]] = None
        self.first_error_m: Optional[float] = None
        self.moved_m: Optional[float] = None
        self.status = self.refused or "girando para a referencia"
        self._parts: List[str] = []
        # Anchor correction found so far, not yet applied by the runner: later
        # sides are judged from the pose WITH it (see the module docstring).
        self._shift = (0.0, 0.0)

        self._i = 0
        self._phase = "turn_out"
        self._issued = False
        self._more_turn = False
        self._seen_active = False
        self._active_by = 0.0
        self._settle_until = 0.0
        self._after_settle = "sample"
        self._readings: List[float] = []
        self._looks = 0

    # -- the side being worked on ------------------------------------------
    @property
    def turn(self) -> int:
        return self.sides[self._i] if self.sides else 0

    @property
    def ref_cell(self):
        return self.refs[self.turn].cell if self.sides else None

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

    def _next_side(self) -> None:
        """This side is finished: on to the next one, or back to the leg."""
        self._i += 1
        self._issued = False
        self.first_error_m = None
        self.moved_m = None
        self._phase = "turn_out" if self._i < len(self.sides) else "turn_back"

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
                self._phase = "done"
                self.status = self.summary()
                return "done"
            self._phase = self._after_settle
            self._readings, self._looks = [], 0
            return "pending"
        if self._phase in ("sample", "resample"):
            return self._do_sample(drive_tel, wx, wy, wth)
        if self._phase == "move":
            if self._move_finished(drive_tel):
                self._settle_then("resample")
            return "pending"
        if self._phase == "turn_back":
            return self._do_turn(drive_tel, wth, self.axis, then="done")
        return "done"

    def _do_turn(self, drive_tel, wth, target, then) -> str:
        if not self._issued:
            delta = math.degrees(_wrap(target - wth))
            # Side to side is 180 deg: issued as two bounded steps, the second
            # recomputed from where the first actually ended.
            self._more_turn = abs(delta) > MAX_TURN_STEP_DEG + 1.0
            if self._more_turn:
                delta = math.copysign(MAX_TURN_STEP_DEG, delta)
            self.link.turn(delta, self.turn_speed)
            self._armed()
            self.status = ("voltando ao rumo da perna" if then == "done" else
                           f"girando {self.turn * 90:+d} deg para {self.ref_cell}")
            return "pending"
        if self._move_finished(drive_tel):
            if not self._more_turn:
                self._settle_then(then)
            # else: _issued is False again, so the next frame issues the rest
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
        wall than believed), or None when it cannot be judged. Expected is
        predicted from the believed pose plus the shift already found."""
        if not self._readings:
            return None
        off = abs(math.degrees(_wrap(wth - (self.axis + self.turn * HALF_PI))))
        if off > RANGE_FIX_ALIGN_TOLERANCE_DEG:
            return None
        expected, hit = predict_range(self.view, self.origin, self.cell_length,
                                      wx + self._shift[0], wy + self._shift[1],
                                      wth, RANGE_FIX_MAX_PREDICTED_M)
        if expected is None or hit != self.ref_cell:
            return None
        measured = sum(self._readings) / len(self._readings)
        return measured - expected

    def _judge_first(self, error, wth) -> None:
        side = f"{self.turn * 90:+d} deg"
        if error is None:
            self._parts.append(f"{side}: sem leitura utilizavel de {self.ref_cell}")
            self._next_side()
            return
        if abs(error) > SCAN_FIT_MAX_LATERAL_M:
            # Most of a cell off is a wrong-wall hypothesis, not drift: the
            # beam caught something the map does not have.
            self._parts.append(f"{side}: erro {error * 100:+.1f} cm grande "
                               f"demais em {self.ref_cell} (parede errada?)")
            self._next_side()
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
        """Fold this side's residual into the anchor correction, move on."""
        if abs(residual) >= RANGE_FIX_MIN_CORRECTION_M:
            # residual > 0: the kart is further from the wall than believed,
            # i.e. behind the belief along the beam -- pull the belief back.
            sx = self._shift[0] - residual * math.cos(wth)
            sy = self._shift[1] - residual * math.sin(wth)
            self._shift = (sx, sy)
            self.anchor_shift = self._shift
            anchor = f"ancora {-residual * 100:+.1f} cm"
        else:
            anchor = "ancora mantida"
        self._parts.append(f"{self.turn * 90:+d} deg -> {self.ref_cell}: erro "
                           f"{self.first_error_m * 100:+.1f} cm, {note}, {anchor}")
        self._next_side()

    # -- reporting ---------------------------------------------------------
    def summary(self) -> str:
        return self.refused or " | ".join(self._parts) or self.status
