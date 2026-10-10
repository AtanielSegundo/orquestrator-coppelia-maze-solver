"""
One angular sweep: rotate in steps, sample the front sensor at each heading,
fit a 2-DOF pose correction to the profile.

Why a sweep at all
------------------
`range_fix` and `Recenter` both measure along a single forward ray, so between
them they close ONE degree of freedom -- the longitudinal one. Lateral offset is
invisible to a forward ray (recenter.py's docstring says so outright), and
lateral is the component that actually puts the kart into a wall: a corridor is
one cell wide, so a few centimetres sideways is most of the clearance there is.

Rotating the kart turns that blind axis into a measured one. Readings at several
headings see the maze from several directions, so the profile of range-versus-
angle shifts when the kart is off to one side in a way no single reading can
show. The fit lives in `scan_fit`; this module is the part that has to talk to a
real kart without blocking the render loop.

Shape of the thing
------------------
Non-blocking, polled once per frame, exactly like `Recenter` -- issue a
rotation, watch telemetry for it to finish, settle, average a few readings,
rotate again. States:

    plan -> turning -> settling -> sampling -> (next angle) ...
         -> fitting -> [moving -> settling] -> returning -> done

The fit runs while the kart still faces the LAST angle, because that is the
one heading in the whole session from which the lateral error can be driven
out physically -- see "Lateral recentring" below.

`poll()` returns "pending", "done" or "refused". The caller owns what happens to
the correction; this module only produces it.

Two things worth knowing before changing it
-------------------------------------------
**The heading recorded with each sample is the MEASURED one**, read back from
telemetry, never the angle that was commanded. A 15 degree step that came out as
12 must enter the fit as 12, or the rotation error is laundered into the position
estimate -- see ScanSample's own docstring. This is also why the sweep does not
simply trust its own plan: it reads where the kart actually ended up.

**The fit runs on a worker thread.** Measured at ~60-90 ms on tcc.maze (a few
hundred candidate positions times a raycast per sample), which is 4-6 frames at
60 FPS. Everything else in this runner is built to never block the render loop,
and a position fit is not a good reason to be the exception. The thread is
started when the last sample is in and `poll()` returns "pending" until it
finishes -- the kart is stationary by then, so there is nothing to race with.

The kart is left pointing where it started. The leg that follows was planned
from the heading the kart had before it; returning to that heading keeps the
sweep transparent to the caller.

Lateral recentring
------------------
The kart cannot translate sideways, which is why a lateral error has so far
only ever been corrected in the BELIEF (the caller's `shift_anchor`): the real
kart stays off-centre, and the next leg's goto has to cut diagonally to reach
the next centre.

But at the end of the sweep the kart is NOT facing along the leg -- it is up to
SWEEP_HALF_SPAN_DEG off it, nearly side-on to the corridor. A straight move
from there is mostly lateral. So before turning back, with the leg heading
`pth`, the current heading `th` and `e` the fitted pose's lateral offset from
the cell centre (along the left normal of `pth`), the kart drives

    d = -e / sin(th - pth)

which zeroes the lateral offset. It also shifts the kart by d*cos(th - pth)
along the leg, which is left to the next goto: that one aims at an absolute
centre, so a longitudinal offset is just a slightly longer or shorter leg.

The move is open loop and best effort, and it does not need to be exact.
Odometry records whatever the wheels actually did, so the anchor correction
the caller applies afterwards stays valid however far the kart really got --
a short move only leaves some of the offset for the next sweep to find.
"""
from __future__ import annotations

import math
import threading
import time
from typing import List, Optional

import Coppelia.helpers as cop

from runner.constants import (
    RANGE_FIX_MAX_PREDICTED_M, SWEEP_HALF_SPAN_DEG, SWEEP_SAMPLES_PER_ANGLE,
    SWEEP_MAX_LOOKS_PER_ANGLE, SWEEP_SETTLE_S, SWEEP_SIDE_PROBE_M,
    SWEEP_START_GRACE_S, SWEEP_STEP_DEG,
    SWEEP_RECENTER_LATERAL, SWEEP_RECENTER_MIN_ANGLE_DEG,
    RECENTER_MAX_STEP_M, RECENTER_MIN_STEP_M, RECENTER_SPEED,
    RECENTER_THRESHOLD_M,
    SCAN_FIT_MAX_LATERAL_M, SCAN_FIT_MIN_IMPROVEMENT, SCAN_FIT_MIN_SAMPLES,
)
from runner.range_fix import is_trusted_reading, predict_range
from runner.scan_fit import ScanSample, fit_scan


def plan_angles(view, origin, cell_length, wx, wy, wth,
                half_span=SWEEP_HALF_SPAN_DEG, step=SWEEP_STEP_DEG) -> List[float]:
    """Offsets from the current heading to sample, in the order to visit them.

    Sides the map says are open are skipped. Not an optimisation for its own
    sake: a sample whose raycast predicts no wall is dropped by the fit anyway
    (`scan_fit._residuals` skips it), so sweeping into open space spends
    rotations and settling time to collect nothing. On a maze with pillars this
    typically halves the sweep.

    Headings that face straight off the map are skipped too (see
    `_leaves_maze`): on a border cell that trims the outward end of the span,
    and in a corner it can leave too few angles for the sweep to run at all --
    which is the right answer there, not a loss, since those headings could
    never have produced a sample.

    Ordered to walk monotonically from one end to the other rather than
    out-and-back from centre, so consecutive rotations are one step apart and
    the kart never re-traverses ground it already sampled -- ending on the end
    furthest from the heading, where the lateral recentring move happens (see
    Sweep). A kart FACING off the map loses the middle of the span instead of
    an end; it then turns across that gap in one rotation, without stopping in
    it.
    """
    def side_has_wall(sign: float) -> bool:
        # Probe at the mid-span angle of that side: a wall the sweep could
        # actually catch, rather than straight out at 90 degrees where the
        # reading would be inside the sensor's blind zone anyway.
        probe = math.radians(sign * half_span / 2.0)
        predicted, _hit = predict_range(view, origin, cell_length,
                                        wx, wy, wth + probe,
                                        SWEEP_SIDE_PROBE_M)
        return predicted is not None

    left = side_has_wall(+1.0)
    right = side_has_wall(-1.0)
    if not left and not right:
        # Nothing on either flank. Still sweep: the forward samples alone can
        # carry a fit, and refusing here would skip the cells where a pillar
        # sits dead ahead.
        left = right = True

    lo = -half_span if right else 0.0
    hi = +half_span if left else 0.0
    n = int(round((hi - lo) / step))
    span = [lo + i * step for i in range(n + 1)]
    angles = [a for a in span
              if not _leaves_maze(view, origin, cell_length, wx, wy,
                                  wth + math.radians(a))]
    # End on the WIDER end: the lateral recentring move is driven from the
    # last angle, and a move there only reaches the lateral axis by
    # sin(angle) -- so a one-sided sweep must finish at its +-75, not at 0.
    if angles and abs(angles[0]) > abs(angles[-1]):
        angles.reverse()
    return angles


def _leaves_maze(view, origin, cell_length, wx, wy, theta) -> bool:
    """Does a beam along `theta` cross straight out of the grid -- i.e. is the
    first cell it enters, past the kart's own, off the map?

    Such a heading can never contribute a sample: `predict_range` returns None
    the moment its ray leaves the map, and the fit drops every sample without a
    prediction. Turning to it is pure rotation and settling time.

    Judged from the tracked CENTER rather than the sensor face, because the
    question is which neighbour the kart is facing, not where the beam starts.
    """
    rf, cf = cop.world_to_cell(origin, cell_length, wx, wy)
    here = (round(rf), round(cf))
    cx, cy = math.cos(theta), math.sin(theta)
    step = cell_length * 0.05
    # 1.5 cells: past the far corner of the kart's own cell from anywhere in it.
    for i in range(1, 31):
        t = i * step
        rf, cf = cop.world_to_cell(origin, cell_length, wx + cx * t, wy + cy * t)
        cell = (round(rf), round(cf))
        if cell != here:
            return not view.in_bounds(cell)
    return False


class Sweep:
    """One sweep session, polled once per frame.

    `poll(drive_tel, wx, wy, wth)` returns:

        "pending"   still working -- call again next frame
        "done"      finished, back on the original heading; `fit` holds the
                    correction (or None if refused)
        "refused"   could not run at all (nothing to measure against)

    The pose arguments are the live believed pose. The pose the fit is computed
    AGAINST is the one captured at construction, because that is the pose every
    sample's geometry refers to and the one being corrected.

    `lateral_move_m` is the recentring move actually commanded (None if none
    was), for the caller's log; the anchor correction to apply is still
    `applied_lateral` / `applied_longitudinal`, move or no move.
    """

    def __init__(self, view, origin, cell_length, link, wx, wy, wth,
                 clock=time.monotonic):
        # `clock` is injectable so tests can drive the settle/grace timers
        # deterministically instead of sleeping. Production always uses the
        # real monotonic clock.
        self._clock = clock
        self.view = view
        self.origin = origin
        self.cell_length = cell_length
        self.link = link
        # Frozen at construction: the sweep measures the error of THIS pose.
        self.px, self.py, self.pth = wx, wy, wth

        self.angles = plan_angles(view, origin, cell_length, wx, wy, wth)
        self.samples: List[ScanSample] = []
        self.fit = None
        self.status = "planejando"
        self.applied_lateral: Optional[float] = None
        self.applied_longitudinal: Optional[float] = None
        self.lateral_move_m: Optional[float] = None

        self._index = 0
        self._phase = "turning"
        self._issued_from: Optional[float] = None
        self._seen_active = False
        self._active_by = 0.0
        self._settle_until = 0.0
        self._readings: List[float] = []
        self._looks = 0
        self._thread: Optional[threading.Thread] = None
        self._fit_done = threading.Event()
        self._fit_error: Optional[str] = None
        self._returning = False
        self._result = ""                # final status, shown once back on heading
        self._refused: Optional[str] = None

        if len(self.angles) < SCAN_FIT_MIN_SAMPLES:
            # Not enough headings worth visiting to ever satisfy the fit, so
            # spending the rotations would be pure cost.
            self._refused = (f"so {len(self.angles)} angulo(s) uteis, "
                             f"minimo {SCAN_FIT_MIN_SAMPLES}")

    # -- helpers -----------------------------------------------------------
    def _target_theta(self, index: int) -> float:
        return self.pth + math.radians(self.angles[index])

    def _wait_for_start(self, drive_tel) -> bool:
        """Same stale-telemetry guard as Leg/Recenter: `goal_active` False is
        only believed once its True edge has been seen, or the grace expires."""
        if drive_tel.get("goal_active"):
            self._seen_active = True
        if self._seen_active:
            return False
        return self._clock() < self._active_by

    def _issue_turn(self, wth: float, target: float) -> None:
        delta = math.degrees(_wrap(target - wth))
        self.link.turn(delta, 0.35)
        self._seen_active = False
        self._active_by = self._clock() + SWEEP_START_GRACE_S

    # -- main poll ---------------------------------------------------------
    def poll(self, drive_tel: dict, wx: float, wy: float, wth: float) -> str:
        if self._refused is not None:
            self.status = self._refused
            return "refused"
        if self._phase == "turning":
            return self._do_turning(drive_tel, wth)
        if self._phase == "settling":
            return self._do_settling()
        if self._phase == "sampling":
            return self._do_sampling(drive_tel, wth)
        if self._phase == "fitting":
            return self._do_fitting(wx, wy, wth)
        if self._phase == "moving":
            return self._do_moving(drive_tel)
        return "done"

    def _do_turning(self, drive_tel, wth) -> str:
        if self._issued_from is None:
            target = (self.pth if self._returning
                      else self._target_theta(self._index))
            self._issue_turn(wth, target)
            self._issued_from = wth
            label = ("voltando ao rumo inicial" if self._returning else
                     f"girando para {self.angles[self._index]:+.0f} deg")
            self.status = f"{label} ({len(self.samples)} amostras)"
            return "pending"
        if self._wait_for_start(drive_tel):
            return "pending"
        if drive_tel.get("goal_active"):
            return "pending"
        self._issued_from = None
        self._phase = "settling"
        self._settle_until = self._clock() + SWEEP_SETTLE_S
        return "pending"

    def _do_settling(self) -> str:
        if self._clock() < self._settle_until:
            return "pending"
        if self._returning:
            # Back on the original heading: fitted, recentred, finished.
            self._phase = "done"
            self.status = self._result
            return "done"
        self._readings = []
        self._looks = 0
        self._phase = "sampling"
        return "pending"

    def _do_sampling(self, drive_tel, wth) -> str:
        """Average a few readings at this heading before moving on.

        The sensor quantises to whole centimetres while the fit is resolving
        millimetres, so one reading throws away resolution that averaging a
        handful gets back. Invalid or distant readings are skipped rather than
        counted: a heading that yields nothing usable contributes no sample,
        which the fit handles (it just has one fewer).
        """
        slip = drive_tel.get("slip") or {}
        if slip.get("slipping"):
            # Mid-slip the pose is moving under us, so a reading taken now does
            # not belong to the heading it would be recorded against.
            return "pending"

        self._looks += 1
        dist_cm = drive_tel.get("front_distance_cm")
        if is_trusted_reading(dist_cm):
            self._readings.append(float(dist_cm) / 100.0)

        if len(self._readings) < SWEEP_SAMPLES_PER_ANGLE:
            # Bounded, not "until we have enough": telemetry repeats the same
            # cached reading between sensor updates, and a heading pointed at
            # open space never yields one at all. Either way the sweep must move
            # on rather than stall the run on one angle.
            if self._looks < SWEEP_MAX_LOOKS_PER_ANGLE:
                return "pending"
            if not self._readings:
                self._advance_angle()      # nothing here; no sample from it
                return "pending"
            # Fewer looks than asked for, but real ones -- use what we have.

        measured = sum(self._readings) / len(self._readings)
        # The heading recorded is the one telemetry reports NOW, not the one
        # that was commanded -- see the module docstring.
        self.samples.append(ScanSample(theta=wth, measured=measured))
        self._advance_angle()
        return "pending"

    def _advance_angle(self) -> None:
        self._index += 1
        self._issued_from = None
        # Past the last angle: fit right here, still facing it -- see the
        # module docstring's "Lateral recentring" for why not after returning.
        self._phase = "turning" if self._index < len(self.angles) else "fitting"

    def _begin_return(self) -> None:
        self._returning = True
        self._phase = "turning"
        self._issued_from = None

    # -- the fit, off the render thread ------------------------------------
    def _do_fitting(self, wx, wy, wth) -> str:
        if self._thread is None:
            if len(self.samples) < SCAN_FIT_MIN_SAMPLES:
                self._result = (f"{len(self.samples)} amostras validas, "
                                f"minimo {SCAN_FIT_MIN_SAMPLES}")
                self._begin_return()
                return "pending"
            self.status = f"ajustando {len(self.samples)} amostras..."
            self._thread = threading.Thread(target=self._fit_worker,
                                            daemon=True, name="ScanFit")
            self._thread.start()
            return "pending"
        if not self._fit_done.is_set():
            return "pending"          # a few frames at most; kart is stopped
        self._judge_fit()
        if self._start_lateral_move(wx, wy, wth):
            return "pending"
        self._begin_return()
        return "pending"

    def _fit_worker(self) -> None:
        try:
            # yield_every: release the GIL every few search rows so the
            # render loop keeps its frame rate while this runs. Without it a
            # CPU-bound worker thread stalls the main thread for ~16 ms -- a
            # whole frame -- which defeats the point of being off-thread.
            self.fit = fit_scan(self.view, self.origin, self.cell_length,
                                self.px, self.py, self.pth, self.samples,
                                yield_every=2)
        except Exception as exc:                                  # noqa: BLE001
            # Not `_refused`: that would end the session right here, leaving
            # the kart facing the last sweep angle. It still has to turn back.
            self.fit = None
            self._fit_error = f"ajuste falhou: {exc}"
        finally:
            self._fit_done.set()

    def _judge_fit(self) -> None:
        """Decide what of the fit may be believed, and record it."""
        fit = self.fit
        if self._fit_error is not None:
            self._result = self._fit_error
            return
        if fit is None:
            self._result = "sem ajuste (geometria insuficiente)"
            return

        # A fit that explains the readings no better than the uncorrected pose
        # did is the search finding noise, not evidence of an error.
        if fit.rms_after > fit.rms_before * SCAN_FIT_MIN_IMPROVEMENT:
            self._result = (f"ajuste rejeitado: rms {fit.rms_before*100:.1f} -> "
                            f"{fit.rms_after*100:.1f} cm (sem ganho)")
            self.fit = None
            return

        if abs(fit.lateral) > SCAN_FIT_MAX_LATERAL_M:
            # Most of a cell sideways is a wrong-cell hypothesis, not drift.
            self._result = (f"lateral {fit.lateral*100:+.1f} cm grande demais "
                            f"(parede errada?)")
            self.fit = None
            return

        self.applied_lateral = fit.lateral
        self.applied_longitudinal = fit.longitudinal
        lo = ("n/d" if fit.longitudinal is None
              else f"{fit.longitudinal*100:+.1f} cm")
        self._result = (f"lateral {fit.lateral*100:+.1f} cm, long {lo} "
                        f"({fit.used} amostras, rms {fit.rms_before*100:.1f} -> "
                        f"{fit.rms_after*100:.1f} cm)")

    # -- lateral recentring, from the last angle ---------------------------
    def _start_lateral_move(self, wx, wy, wth) -> bool:
        """Issue the move that puts the kart on the cell's centre line, if one
        is due. Returns True when a move was issued. See the module docstring.
        """
        if not SWEEP_RECENTER_LATERAL or self.applied_lateral is None:
            return False

        off = _wrap(wth - self.pth)
        reach = math.sin(off)                # lateral metres per metre driven
        if abs(reach) < math.sin(math.radians(SWEEP_RECENTER_MIN_ANGLE_DEG)):
            self._result += (f" | sem recentragem lateral (ultimo angulo "
                             f"{math.degrees(off):+.0f} deg)")
            return False

        # Where the fit says the kart is NOW. The correction is to the anchor,
        # so it holds for the live belief just as for the frozen one -- the
        # in-place rotations since then moved both together.
        c, s = math.cos(self.pth), math.sin(self.pth)
        along = self.applied_longitudinal or 0.0
        fx = wx + c * along - s * self.applied_lateral
        fy = wy + s * along + c * self.applied_lateral
        rf, cf = cop.world_to_cell(self.origin, self.cell_length, fx, fy)
        ccx, ccy, _ = cop.cell_to_world(self.origin, self.cell_length,
                                        (round(rf), round(cf)))
        # Offset from the centre line, along the left normal (-s, c) of pth.
        error = (fx - ccx) * -s + (fy - ccy) * c
        if abs(error) < RECENTER_THRESHOLD_M:
            self._result += f" | lateral ja centrada ({error*100:+.1f} cm)"
            return False

        move = -error / reach
        move = max(-RECENTER_MAX_STEP_M, min(RECENTER_MAX_STEP_M, move))
        if abs(move) < RECENTER_MIN_STEP_M:
            self._result += (f" | recentragem lateral {move*100:+.1f} cm menor "
                             f"que o passo minimo")
            return False

        self.lateral_move_m = move
        self.link.drive_distance(move, RECENTER_SPEED)
        self._seen_active = False
        self._active_by = self._clock() + SWEEP_START_GRACE_S
        self._phase = "moving"
        verb = "avancando" if move > 0 else "recuando"
        self.status = (f"recentrando lateral: {verb} {abs(move)*100:.1f} cm "
                       f"a {math.degrees(off):+.0f} deg")
        self._result += (f" | recentrou lateral {error*100:+.1f} cm "
                         f"({verb} {abs(move)*100:.1f} cm a "
                         f"{math.degrees(off):+.0f} deg)")
        return True

    def _do_moving(self, drive_tel) -> str:
        if self._wait_for_start(drive_tel):
            return "pending"
        if drive_tel.get("goal_active"):
            return "pending"
        # No settle of its own: the return turn that follows settles anyway,
        # and nothing is measured in between.
        self._begin_return()
        return "pending"

    # -- reporting ---------------------------------------------------------
    def summary(self) -> str:
        return self.status


def _wrap(t: float) -> float:
    """Radians into (-pi, pi]."""
    return (t + math.pi) % (2 * math.pi) - math.pi
