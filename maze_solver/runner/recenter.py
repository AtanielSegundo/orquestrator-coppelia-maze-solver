"""
Physical recentring: drive the kart onto the position the map predicts, instead
of only correcting the belief.

Relation to range_fix
---------------------
`range_fix` measures the longitudinal error and shifts the ANCHOR, so the kart
stays put and the belief moves onto it. This module does the opposite: it moves
the KART onto the belief, commanding short moves until the front sensor reads
the range frozen at the start of the session.

The target: the range predicted from the believed pose, FROZEN
---------------------------------------------------------------
`target_range` is computed once, from the pose believed when the session opens,
and never recomputed. Driving until the measurement matches therefore puts the
kart where odometry thought it already was -- which is the frame the whole run
is planned in.

Only the longitudinal coordinate. A single forward-facing ray observes one
degree of freedom; lateral offset is invisible to it and is left alone
(recovering it needs a multi-heading profile, a separate feature).

Why the target must be FROZEN
------------------------------
The tempting variant recomputes the predicted range from the *current* believed
pose each pass and drives the kart to close that gap. It cannot work, and the
reason is not obvious:

**the belief moves with the kart.** The server's `drive_distance` closes its
position loop on the encoders, which keep counting while the wheels slip, so a
commanded move of `m` makes the kart travel `m*(1-slip)` but makes odometry --
and therefore the belief -- advance by the full `m`. The gap between them
evolves as

    e' = e + m*(1-slip) - m = e - m*slip

With no slip the gap is *invariant*: the kart chases a belief that retreats
exactly as fast. With `m = -e` it actively diverges, to `e*(1+slip)`.

Freezing the target breaks that loop: a number captured once cannot retreat as
the kart advances. Each pass then drives on

    error = measured - target_range

Moving forward shortens the measured range, so the move to command is `+error`,
and the error shrinks by a factor of `slip` per pass -- convergent for any
slip < 1, with no faith required in the accuracy of any individual move.

A successful session leaves the kart physically where the belief already said it
was, so no anchor correction is needed afterwards. That is the point: rather
than reconciling the two by editing the belief, it reconciles them by moving the
kart, and the belief turns out to have been right.

Giving up
---------
After `RECENTER_MAX_ATTEMPTS` the kart stops trying and the caller falls back to
correcting the anchor (the belief-only path). A kart that cannot close the gap
physically -- wheels against a wall, a surface it cannot grip, a sensor that
went away -- must not stall the run over 2 cm.
"""
from __future__ import annotations

import time
from typing import Optional

from runner.constants import (
    RANGE_FIX_ALIGN_TOLERANCE_DEG, RANGE_FIX_MAX_CORRECTION_M,
    RANGE_FIX_MAX_PREDICTED_M, RANGE_FIX_MAX_TRUSTED_M,
    RANGE_FIX_MIN_PREDICTED_M,
    RECENTER_MAX_ATTEMPTS, RECENTER_MAX_STEP_M, RECENTER_MIN_STEP_M,
    RECENTER_SETTLE_S, RECENTER_SPEED, RECENTER_START_GRACE_S,
    RECENTER_THRESHOLD_M,
)
from runner.range_fix import (
    heading_alignment_error_deg, is_trusted_reading, predict_range,
)


class Recenter:
    """One recentring session, polled once per frame.

    Non-blocking, like `Leg`: it issues a move, watches telemetry for that
    move's completion, settles, then re-measures. `poll()` returns:

        "pending"   still working -- call again next frame
        "done"      inside the threshold, or nothing to correct against
        "giveup"    out of attempts; the caller should fall back to the anchor
    """

    def __init__(self, range_fix, link, wx: float, wy: float, wth: float):
        self.range_fix = range_fix
        self.link = link
        self.wth = wth
        # Frozen at construction, from the pose believed RIGHT NOW. Never
        # recomputed -- see the module docstring on why a target that tracks
        # the belief is a target that runs away.
        self.target_range: Optional[float] = None
        self.attempts = 0
        self.first_error_m = None
        self.last_error_m = None
        self.status = "medindo"
        self._phase = "measure"      # measure -> moving -> settle -> measure
        self._seen_active = False
        self._active_by = 0.0
        self._settle_until = 0.0
        self._commanded_m = 0.0
        self._refuse = self._freeze_target(wx, wy, wth)

    # -- setup -------------------------------------------------------------
    def _freeze_target(self, wx, wy, wth) -> Optional[str]:
        """Compute and store the target range. Returns a refusal reason, or
        None when the session may proceed."""
        align = heading_alignment_error_deg(wth)
        if align > RANGE_FIX_ALIGN_TOLERANCE_DEG:
            # Off-axis the ray grazes wall corners, so the predicted range
            # becomes hypersensitive to heading error and the residual would
            # measure yaw rather than position.
            return f"fora de eixo ({align:.0f} deg)"

        predicted, _hit = predict_range(self.range_fix.view,
                                        self.range_fix.origin,
                                        self.range_fix.cell_length,
                                        wx, wy, wth,
                                        RANGE_FIX_MAX_PREDICTED_M)
        if predicted is None:
            return "mapa nao preve parede no alcance"
        if not (RANGE_FIX_MIN_PREDICTED_M <= predicted
                <= RANGE_FIX_MAX_PREDICTED_M):
            return f"parede prevista a {predicted:.2f} m (fora da faixa)"
        # A target beyond the trust ceiling would have the loop chasing a
        # reading it is not allowed to act on -- refuse the session outright
        # rather than let every pass reject the measurement.
        if predicted > RANGE_FIX_MAX_TRUSTED_M:
            return (f"parede prevista a {predicted:.2f} m, alem do limite de "
                    f"confianca ({RANGE_FIX_MAX_TRUSTED_M:.2f} m)")
        self.target_range = predicted
        return None

    # -- measurement -------------------------------------------------------
    def _error(self, drive_tel) -> Optional[float]:
        """Signed metres to travel along the heading, or None if unjudgeable.

        Positive = the kart is further from the wall than the target, so it
        must move forward. Compared against the FROZEN target, never against a
        freshly predicted range.
        """
        dist_cm = drive_tel.get("front_distance_cm")
        # Same two gates as range_fix: a real measurement, and one close enough
        # to trust. Driving the kart on a reading is at least as consequential
        # as moving the belief on it, so the bar is not lower here.
        if not is_trusted_reading(dist_cm):
            self.status = "leitura invalida ou distante demais"
            return None
        slip = drive_tel.get("slip") or {}
        if slip.get("slipping"):
            # A reading taken while the wheels scrub is a fine DISTANCE, but
            # acting on it means driving on evidence gathered mid-slip. Same
            # rule range_fix applies before touching the belief.
            self.status = "derrapagem detectada"
            return None
        return float(dist_cm) / 100.0 - self.target_range

    # -- main poll ---------------------------------------------------------
    def poll(self, drive_tel: dict) -> str:
        """One frame. Takes no pose: the target was frozen at construction and
        the error comes from the sensor, so the believed pose is not an input
        at all -- which is the whole point (see the module docstring)."""
        if self._refuse is not None:
            self.status = self._refuse
            return "done"
        if self._phase == "moving":
            return self._do_moving(drive_tel)
        if self._phase == "settle":
            return self._do_settle()
        return self._do_measure(drive_tel)

    def _do_measure(self, drive_tel) -> str:
        error = self._error(drive_tel)
        if error is None:
            return "done"        # status already set by _error

        self.last_error_m = error
        if self.first_error_m is None:
            self.first_error_m = error
            # Sanity, once: an initial error this large is not drift, it is the
            # beam hitting a different wall than the map predicts. Driving on it
            # would be worse than leaving the kart alone.
            if abs(error) > RANGE_FIX_MAX_CORRECTION_M:
                self.status = (f"erro inicial {error * 100:+.1f} cm grande "
                               f"demais (parede errada?)")
                return "done"

        if abs(error) < RECENTER_THRESHOLD_M:
            self.status = f"centrado ({error * 100:+.1f} cm)"
            return "done"

        if self.attempts >= RECENTER_MAX_ATTEMPTS:
            self.status = (f"desistiu apos {self.attempts} tentativa(s), "
                           f"faltou {error * 100:+.1f} cm")
            return "giveup"

        # Clamp: the floor avoids issuing a move too small for the kart to
        # execute (it would never complete and would burn an attempt), the
        # ceiling bounds how far one bad measurement can drive it.
        step = max(-RECENTER_MAX_STEP_M, min(RECENTER_MAX_STEP_M, error))
        if abs(step) < RECENTER_MIN_STEP_M:
            self.status = f"resto {error * 100:+.1f} cm menor que o passo minimo"
            return "done"

        self.attempts += 1
        self._commanded_m = step
        self.link.drive_distance(step, RECENTER_SPEED)
        self._phase = "moving"
        self._seen_active = False
        self._active_by = time.monotonic() + RECENTER_START_GRACE_S
        self.status = (f"{'avancando' if step > 0 else 'recuando'} "
                       f"{abs(step) * 100:.1f} cm "
                       f"(tentativa {self.attempts}/{RECENTER_MAX_ATTEMPTS})")
        return "pending"

    def _do_moving(self, drive_tel) -> str:
        # Same stale-telemetry guard as Leg: the frame right after a command can
        # still carry a sample from before the server processed it, so
        # `goal_active` False is only believed once its True edge has been seen
        # (or the grace period expires, for a move too brief to catch).
        if drive_tel.get("goal_active"):
            self._seen_active = True
            return "pending"
        if not self._seen_active and time.monotonic() < self._active_by:
            return "pending"

        # Let the chassis stop rocking before re-measuring: an ultrasonic
        # reading taken mid-wobble is what would make this oscillate rather
        # than converge.
        self._phase = "settle"
        self._settle_until = time.monotonic() + RECENTER_SETTLE_S
        self.status = "assentando"
        return "pending"

    def _do_settle(self) -> str:
        if time.monotonic() < self._settle_until:
            return "pending"
        self._phase = "measure"
        return "pending"

    # -- reporting ---------------------------------------------------------
    def moved(self) -> bool:
        return self.attempts > 0

    def summary(self) -> str:
        if self.first_error_m is None:
            return self.status
        return (f"{self.status} | de {self.first_error_m * 100:+.1f} cm "
                f"para {(self.last_error_m or 0.0) * 100:+.1f} cm "
                f"em {self.attempts} movimento(s)")
