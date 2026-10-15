from runner.rgb_to_norm import rgb_to_norm

# ---------------------------------------------------------------------------
# Kart Connection
# ---------------------------------------------------------------------------
KART_HOST = "192.168.0.238"
KART_PORT = 8080

CAIXA_COR = rgb_to_norm((178, 131, 97))

GRID_HEADING_LABEL = {
    0.0:   "-> direita/leste (0 deg)",
    90.0:  "^ cima/norte (90 deg)",
    180.0: "<- esquerda/oeste (180 deg)",
    270.0: "v baixo/sul (270 deg)",
}

HELP = """
COMANDOS DO PROMPT
  ok                  confirma o kart REAL posicionado; comeca a rodar
  start <lin> <col>   antes do 'ok': novo start | ao vivo: novo waypoint
  goal  <lin> <col>   novo goal (antes ou durante a corrida)
  resume              esquece TODOS os obstaculos descobertos de uma vez e
                      replaneja
  status              imprime o estado atual
  help | quit
"""

CONTROLS = """
CONTROLES DA JANELA
  botao esquerdo   equivalente a 'start <lin> <col>' da celula clicada
  botao direito    equivalente a 'goal  <lin> <col>' da celula clicada
  setas            move a camera pan/tilt do kart (real + sim)
  C                centraliza a camera (olhando pra frente)
  ESC              sair
"""

COMMANDS = {
    "ok":     (0, 0),
    "start":  (2, 2),
    "goal":   (2, 2),
    "resume": (0, 0),
    "status": (0, 0),
    "quit":   (0, 0),
}

# ---------------------------------------------------------------------------
# Front distance sensor -> live maze
# ---------------------------------------------------------------------------
# A `front_distance_cm` telemetry reading at or below this is a HIT (see
# LiveMazeRunner._scan_obstacle). A hit no longer blocks a cell on its own:
# while the kart moves it only makes the cell a SUSPECT, and only a STEADY
# kart, sitting in the cell right before it and facing it, gets to mark it
# (see the OBSTACLE_* knobs below). The maze's own walls trip this too, which
# is harmless -- they are already walls, so the mark is a no-op.
OBSTACLE_DISTANCE_CM = 90.0

# The sensor measures the NEAR face of whatever reflected, and the thing
# occupies the space behind that face, never in front of it. So the cell an
# obstacle is blamed on is projected this much past the hit point.
#
# Without it a cell-sized box (or a wall) has its face exactly on a cell
# boundary, and the projection is a coin flip between the right cell and the
# OPEN one before it: kart centred in a 0.30 m cell, open cell ahead, wall
# after that -> reading 0.45 - 0.113 = 33.7 cm, rounding to 33 or 34 decides
# whether the open cell gets marked. Sized above the 1 cm quantisation plus a
# few cm of pose error, and well under half a cell, so a hit in the middle of
# a cell still lands in that cell.
OBSTACLE_HIT_DEPTH_M = 0.05

# "Facing the next cell" means the heading is within this of a grid axis.
# Same value as RANGE_FIX_ALIGN_TOLERANCE_DEG, for the same reason: further
# off-axis the beam is as much about the neighbouring row as about the cell
# straight ahead.
OBSTACLE_ALIGN_TOLERANCE_DEG = 12.0

# "Steady": no move in flight (`goal_active` low) and both measured wheel
# speeds under this.
OBSTACLE_STEADY_SPEED_MPS = 0.02

# How long a steady kart must keep reading the SAME verdict about the cell
# ahead -- blocked or clear -- before it is acted on. Any frame that moves,
# turns off-axis or disagrees starts it over. Telemetry repeats its cached
# reading between sensor updates, so this is time and not a frame count:
# it has to span several real readings to filter a single stray echo.
OBSTACLE_CONFIRM_S = 0.4

# How long the runner holds in front of a suspect waiting for that verdict
# before it gives up and drives on. Reached when the steady kart never gets
# a usable reading of the cell (no echo, misaligned, stopped in the wrong
# cell). The suspect is then ignored for the next leg so the kart cannot loop
# stop -> hold -> go -> stop on it; if something really is there, the
# server's own front guard still stops the kart, and the steady check maps it
# from wherever it stopped.
OBSTACLE_SUSPECT_TIMEOUT_S = 3.0

# How far ahead of the kart's tracked CENTER the front sensor actually sits.
# The reading is measured from the sensor, so this is added back before the
# hit point is projected onto the grid; otherwise every obstacle lands about
# half a kart-length too close and can be blamed on the wrong cell.
SENSOR_FORWARD_OFFSET_M = 0.113

# The other half of every front reading: whatever the beam passed THROUGH on
# its way to the hit point has to be empty, or it would have stopped earlier.
# Cells discovered blocked that now read as clear are un-discovered, which is
# what lets a run resume by itself once an obstacle is physically taken away.
# A cell is only cleared after this many consecutive clear looks, so one bad
# reading (or one frame of heading jitter) cannot erase a real obstacle.
SENSOR_CLEAR_CONFIRMATIONS = 6

# While waiting for a route to open up, how often to re-run the planner even
# though nothing in the live maze changed. Maze changes retry immediately; this
# is the floor that also covers a leg that came up short for a reason the maze
# never recorded (a stall, a wheel slip), which no amount of waiting reveals.
LIVE_RETRY_INTERVAL_S = 2.0

# An identical front reading that survives this much travel is FROZEN, not
# real: readings are whole cm, so closing even a few cm on something inside
# OBSTACLE_DISTANCE_CM has to change the number. Sized well above odometry
# jitter for a parked kart (which must stay trusted -- it is how a kart
# stopped in front of a real obstacle still maps it), and well under the
# travel needed to shift the projection into the next cell.
SENSOR_FROZEN_TRAVEL_M = 0.03

# ---------------------------------------------------------------------------
# Front sensor validity (see runner/range_fix.is_valid_reading)
#
# The kart reports "no measurement" in several distinguishable ways, and the
# 1-D pose correction must reject every one of them -- taking any as a distance
# corrupts the pose instead of fixing it:
#
#   None  the server's front_distance_cm expired (front_distance_ttl_s passed
#         with nothing valid), or there is no sensor at all. Rendered "-- cm".
#   255   the RPi.GPIO fallback's failed-read sentinel. Not a distance.
#   300   the pigpio reader's max range, reported both for genuinely far/clear
#         AND for a latched sensor fault -- "I see nothing", not "wall at 3 m".
# ---------------------------------------------------------------------------
SENSOR_INVALID_CM = frozenset({255.0, 300.0})

# HC-SR04 blind zone: below this the echo overlaps the trigger burst and the
# number is unreliable.
SENSOR_MIN_RANGE_CM = 3.0

# Anything at or past this is the sensor's max range, i.e. "nothing there".
SENSOR_MAX_RANGE_CM = 300.0

# ---------------------------------------------------------------------------
# 1-D longitudinal pose correction (runner/range_fix.py)
#
# Odometry closes the loop on rotation (gyro yaw is ground truth) but never on
# translation, so wheel slip accumulates in the pose without bound. The maze
# file's walls are a map we already own, so a front reading compared against a
# raycast of that map measures the pose error along the beam directly.
# ---------------------------------------------------------------------------
# Only correct when the heading is within this of a grid axis. Off-axis, the
# ray grazes wall corners and the predicted range becomes hypersensitive to
# heading error -- the residual would then be measuring YAW error while being
# applied as a TRANSLATION, which actively makes the pose worse.
RANGE_FIX_ALIGN_TOLERANCE_DEG = 12.0

# Usable band for the PREDICTED range. Closer than the minimum, the geometry is
# dominated by which part of the wall face the beam catches; further than the
# maximum, a fraction of a degree of heading error moves the prediction more
# than the drift being corrected.
RANGE_FIX_MIN_PREDICTED_M = 0.08
RANGE_FIX_MAX_PREDICTED_M = 1.20

# Trust ceiling on the MEASURED range: beyond this the reading is not used to
# correct anything, neither the belief nor the kart's position.
#
# Separate from the band above, and stricter, because it bounds a different
# thing: that band says where a PREDICTION is geometrically meaningful, this
# says how far the SENSOR itself is believed. An HC-SR04's cone widens with
# distance (~15 degrees), so a far reading may be the first echo off a side
# wall or a chamfer rather than the surface straight ahead -- and the error
# that misattribution introduces grows with range. Close in, the cone is narrow
# enough that the echo almost certainly came from what the ray points at.
#
# It also caps the damage from heading error: at 12 degrees off-axis (the
# alignment tolerance) a 0.45 m reading is off by 0.45*(1/cos(12deg)-1) ~= 10 mm,
# while at 1.20 m the same angle costs ~27 mm -- more than the whole drift
# being corrected.
#
# Sized against the cell geometry, which is what makes 0.45 the useful value
# here rather than 0.30. With 0.25 m cells and the sensor 0.13 m ahead of the
# tracked centre, the wall the beam can actually see sits at:
#
#     2 cells ahead:  2*0.25 - 0.125 - 0.13 = 0.245 m
#     3 cells ahead:  3*0.25 - 0.125 - 0.13 = 0.495 m
#
# A 0.30 ceiling therefore only ever admitted the 2-cell case, and with barely
# 5 cm of headroom before an offset kart's reading fell outside it. 0.45 keeps
# the 2-cell wall comfortably inside and still excludes the 3-cell one, where
# heading error starts to dominate the residual.
RANGE_FIX_MAX_TRUSTED_M = 0.5

# Residual smaller than this is sensor quantisation, not drift: the kart
# reports whole centimetres, so +-1 cm carries no information.
RANGE_FIX_MIN_CORRECTION_M = 0.02

# Residual larger than this is not drift either -- it means the beam hit a
# different wall than the map predicts (wrong cell, an unmapped object in the
# corridor, or a pose already lost beyond what this can repair). Applying it
# would teleport the kart; refusing leaves a merely-drifting pose, which is the
# smaller error. Sized under half a cell so a whole-cell disagreement is always
# rejected.
RANGE_FIX_MAX_CORRECTION_M = 0.15

# ---------------------------------------------------------------------------
# Physical recentring (runner/recenter.py)
#
# Instead of only moving the BELIEF onto the kart (range_fix shifting the
# anchor), drive the KART back onto the position the map predicts. Closed on
# the sensor: command a move, re-measure, repeat -- so convergence does not
# depend on the move being accurate, only on it going mostly the right way.
# ---------------------------------------------------------------------------
# Displacement from the predicted position that triggers a physical move. Below
# this the kart is left alone (and range_fix's own noise floor of 2 cm means
# anything smaller is not measurable anyway, since the kart reports whole cm).
RECENTER_THRESHOLD_M = 0.02

# Cap on a single corrective move. The measurement could be wrong; this bounds
# how far a bad one can drive the kart before it is re-measured.
RECENTER_MAX_STEP_M = 0.12

# Floor on a single corrective move. Below this the kart cannot reliably
# execute the command at all (stiction, the position PID's own tolerance), so
# issuing it would burn an attempt on a move that never completes.
RECENTER_MIN_STEP_M = 0.015

# Speed for corrective moves. Deliberately slower than the normal drive speed:
# these are centimetre-scale, and a slow approach both slips less and keeps the
# encoders inside the rate they can count.
RECENTER_SPEED = 0.12

# Attempts before giving up and falling back to correcting the anchor only. A
# kart that cannot close the gap physically -- wheels against a wall, a surface
# it cannot grip -- must not stall the whole run over 2 cm.
RECENTER_MAX_ATTEMPTS = 3

# Pause after a move before re-measuring. An ultrasonic reading taken while the
# chassis is still rocking is what would make this oscillate instead of
# converge.
RECENTER_SETTLE_S = 0.35

# Max wait for `goal_active` to go True after issuing a move, before assuming
# the move was too brief for telemetry to have caught the pulse. Same guard as
# Leg's START_GRACE_S, and for the same reason: the first frame after a command
# can still be a sample from before the server processed it.
RECENTER_START_GRACE_S = 0.4


# ---------------------------------------------------------------------------
# Angular sweep + 2-DOF pose fit (runner/scan_fit.py, runner/sweep.py)
#
# range_fix and Recenter both close the LONGITUDINAL axis against a single
# forward ray. That ray cannot observe lateral offset at all -- and lateral is
# the component that actually puts the kart into a wall, since a corridor is
# only one cell wide. Readings taken at several HEADINGS do see it: the profile
# of range-versus-angle shifts when the kart moves sideways, which no single
# reading can show.
#
# Every number below was measured against tcc.maze (6x10, 0.30 m cells) with
# 1 cm sensor noise and whole-centimetre quantisation, over the solved path.
# ---------------------------------------------------------------------------
# Half-span of the sweep, and the step between samples.
#
# The span is set by how many samples survive, not by taste. Measured on
# tcc.maze's solved path -- mean usable samples per cell, and cells that reach
# SCAN_FIT_MIN_SAMPLES (of 17):
#
#     +-45 / 15 deg -> 2.5 samples/cell,  7 of 17 usable
#     +-60 / 15 deg -> 3.9 samples/cell,  9 of 17 usable
#     +-75 / 15 deg -> 4.9 samples/cell, 14 of 17 usable
#     +-90 / 15 deg -> 5.9 samples/cell, 14 of 17 usable
#
# +-75 is where the yield stops improving for free: it costs the same 11
# rotations as +-60 at this step and reaches 14 cells instead of 9, while +-90
# adds two more rotations for no extra cells. Accuracy is unchanged by the
# wider span (both +-60 and +-75 measure 0.00 cm median / 0.50 cm p90 lateral
# error with 1 cm sensor noise, zero gross failures), so the wider one is
# strictly better here.
#
# Note what the outermost samples are worth on real hardware, though: at 0.30 m
# cells a side wall 0.15 m away subtends 2.5 cm of range at 75 deg, inside the
# HC-SR04's blind zone (SENSOR_MIN_RANGE_CM), and ultrasonic echo off a flat
# wall is largely specular past ~15 deg of incidence. The simulation's ideal
# raycasts are optimistic about those angles; if the bench shows the wide
# samples returning nothing, drop back to +-60 and expect the lower yield
# rather than chasing it with a finer step.
#
# 15 deg steps over 10: finer steps do raise the count (7.6/cell at +-75/10,
# 15 of 17 cells) but at 31 rotations per sweep instead of 11, and each rotation
# is an open-loop turn whose own error enters the fit through the sample's
# recorded heading.
SWEEP_HALF_SPAN_DEG = 90.0
SWEEP_STEP_DEG = 45.0

# Skip a side the map says is open: a sample that predicts no wall is dropped
# by the fit anyway (scan_fit._residuals), so sweeping into open space spends
# rotations for nothing. A side counts as worth sweeping if the map predicts a
# wall within this range on that side.
SWEEP_SIDE_PROBE_M = 0.45

# Pause after each rotation before reading. Same reasoning as
# RECENTER_SETTLE_S: an ultrasonic reading taken while the chassis is still
# rocking is noise, and here it would be noise fed straight into a least-
# squares fit that assumes the pose was static.
SWEEP_SETTLE_S = 0.30

# Readings averaged at each angle. The sensor quantises to whole centimetres
# and the fit is trying to resolve a few millimetres, so averaging a handful of
# samples at one heading recovers resolution the single reading throws away.
SWEEP_SAMPLES_PER_ANGLE = 6

# Max wait for `goal_active` after a sweep rotation -- same stale-telemetry
# guard as Leg.START_GRACE_S and RECENTER_START_GRACE_S.
SWEEP_START_GRACE_S = 0.4

# Lateral recentring at the end of a sweep (see runner/sweep.py's docstring):
# from the last angle, nearly side-on to the corridor, drive forward/back so the
# kart's centre lands on the cell's centre line, THEN turn back to the leg
# heading. Off, the lateral error is only ever corrected in the belief, as
# before. The move reuses Recenter's limits -- RECENTER_THRESHOLD_M (smallest
# lateral error worth moving for), RECENTER_MIN_STEP_M / RECENTER_MAX_STEP_M
# (move clamp) and RECENTER_SPEED -- since it is the same kind of
# centimetre-scale corrective move.
SWEEP_RECENTER_LATERAL = True

# Smallest angle between the last sweep heading and the leg heading at which
# the move is attempted. A move of d only reaches the lateral axis by
# d*sin(angle): at 45 deg that is already 1.4x the lateral error in travel, as
# much of it along the leg as across it; closer to the leg the move is mostly
# longitudinal and not worth the time. The sweep normally ends at
# SWEEP_HALF_SPAN_DEG (plan_angles finishes on its wider end), so this only
# bites when a border cell has cut the span short.
SWEEP_RECENTER_MIN_ANGLE_DEG = 45.0

# Minimum interval between sweeps, in legs -- a FLOOR, not a target. 1 means
# "every boundary that has geometry to measure against", which is what keeping
# the kart off the walls actually requires.
#
# This was 3, on the reasoning that the absolute goto had made lateral drift
# slow. That reasoning was wrong, and the measurement says so. `goto` closes its
# loop on ODOMETRY, so it cannot see a heading error: if the kart turns 2 degrees
# more than commanded, odometry records the commanded heading, the belief and
# the target agree perfectly, and goto reports success while the real kart walks
# off at an angle. Worst real position error over tcc.maze's 17-cell path, with
# per-rotation heading error, relative legs vs absolute goto:
#
#     1 deg/turn ->  3.8 cm both      3 deg/turn -> 11.3 cm both
#     2 deg/turn ->  7.6 cm both      5 deg/turn -> 18.9 cm both
#
# Identical. Against heading error the goto buys nothing, because each 30 cm leg
# displaces sin(err)*0.30 sideways (0.52 cm per degree) and nothing proprioceptive
# observes it. Only an exteroceptive measurement does -- the sweep.
#
# Worst error with the sweep, by interval (clearance to a wall is 15 cm minus
# half the kart, about 8 cm):
#
#     interval   2 deg    3 deg    5 deg
#     never      7.6 cm  11.3 cm  18.9 cm
#     every 3    5.4 cm   8.1 cm  13.1 cm   <- still hits at 3 deg
#     every 1    3.1 cm   7.2 cm   7.5 cm   <- holds to 5 deg
#
# Hence 1. The cost is bounded by the geometry gate in
# LiveMazeRunner._sweep_worth_trying, which skips boundaries where the map
# foresees too little to fit: 13 attempts instead of 16 on this path, for the
# same accuracy as sweeping unconditionally.
#
# Do not "optimise" this back to an every-N cadence. Measured, a cadence of 2
# was WORSE than 3 (12.8 cm vs 4.8 cm) because of which cells the count lands
# on -- the result aliases against the maze layout, so the knob is not
# monotonic and tuning it is tuning noise.
#
# Counted in CELLS driven, not in Leg objects: every measurement above was
# taken when one leg was one cell, and a leg can now span up to LEG_MAX_CELLS
# (below). Counting legs would quietly stretch the interval up to that many
# times further in distance.
SWEEP_EVERY_N_LEGS = 2

# Longest straight run driven as ONE leg. Consecutive path cells that keep
# the same heading are merged into a single goto -- (r, c) -> (r, c+3) heading
# east is one move to the far cell's centre, with no stop at the two cells in
# between. Longer straight runs are cut into pieces of at most this many
# cells. 1 restores the old cell-by-cell driving.
#
# Starts equal to SWEEP_EVERY_N_LEGS so a full straight run ends where a
# sweep is due, but it is its own knob: tying it to the sweep interval would
# make tuning one silently change the other.
LEG_MAX_CELLS = 3

# ---------------------------------------------------------------------------
# The fit itself (runner/scan_fit.py)
# ---------------------------------------------------------------------------
# Half-width of the (dx, dy) search window. Just over half a cell: a genuine
# error larger than this is a lost pose rather than drift, and a wider window
# mostly lets the search find a convincing wrong answer against the next cell's
# walls.
SCAN_FIT_MAX_SHIFT_M = 0.12

# Two-pass grid: coarse over the window, then fine around the winner. The
# product is ~250 candidate positions and a few hundred raycasts each -- tens
# of milliseconds, which is why the sweep runs the fit on a worker thread
# rather than inline (see runner/sweep.py).
SCAN_FIT_COARSE_STEP_M = 0.03
SCAN_FIT_FINE_STEP_M = 0.005

# Fewest samples that may produce a fit. Two readings at distinct headings are
# the formal minimum for 2 DOF; 4 is where the measured error stops having a
# tail. On tcc.maze with 1 cm noise, median lateral error and the rate of
# gross (>3 cm) failures:
#
#     min 3 -> 0.27 cm median,  5 gross in 494 fits
#     min 4 -> 0.25 cm median,  0 gross in 289 fits
#
# The cost of the higher bar is fits refused, not fits made badly, and a
# refused fit leaves the belief alone -- the safe direction.
SCAN_FIT_MIN_SAMPLES = 3

# Fraction of the worst residuals dropped before summing. ZERO, deliberately,
# and this one is counter-intuitive enough to be worth the measurement.
#
# Trimming is the right instinct with many samples: one reading through an
# opening the map does not model is off by a whole cell, and squaring makes it
# worth hundreds of good ones. But a sweep yields ~4 samples, and a 2-DOF fit
# needs 2 -- so dropping 25% of four samples throws away one of the readings
# that was pinning the answer, leaving the fit underdetermined and free to
# slide. Measured on tcc.maze, 1 cm noise, p90 of the lateral error and gross
# failures out of 289 fits:
#
#     trim 0.00 -> p90 0.82 cm,   0 gross
#     trim 0.15 -> p90 1.27 cm,  19 gross
#     trim 0.25 -> p90 8.68 cm,  42 gross
#
# An 8.68 cm p90 error is worse than the drift being corrected. Robustness
# here comes from SCAN_FIT_MIN_SAMPLES and the observability gate instead; if
# the sweep is ever made dense enough to have samples to spare, revisit this
# with the same measurement.
SCAN_FIT_TRIM_FRACTION = 0.0

# ---------------------------------------------------------------------------
# Observability gate (runner/scan_fit.py: longitudinal_is_observable)
#
# In a straight corridor with no wall ahead in range, longitudinal position is
# UNOBSERVABLE: the side walls look identical wherever the kart sits along
# them. Measured directly -- the trimmed cost is exactly 0.0 across the whole
# +-16 cm window, so the fit's longitudinal number is whatever the search
# happened to touch first. On tcc.maze's open cells it came out at -16 to
# -18 cm against a true +2 cm.
#
# Acting on that would drive the kart up to 16 cm the wrong way to correct 2.
# So the longitudinal component is reported only when the cost actually RISES
# as the candidate moves along the heading, by at least this much relative to
# the cost at the optimum. Lateral is unaffected: it is well observed in
# exactly the geometry where longitudinal is not (side walls are what a sweep
# sees best), and it is the component with no other corrector.
SCAN_FIT_OBSERVABLE_MIN_RISE = 1e-6

# How far along the heading the gate probes for that rise. Large enough to
# leave the flat floor of a genuine minimum, small enough to stay inside the
# search window.
SCAN_FIT_OBSERVABLE_PROBE_M = 0.04

# Reject a fit that does not improve the residuals by at least this factor:
# rms_after < rms_before * this. A fit that explains the readings no better
# than the uncorrected pose did is not evidence of an error, it is the search
# finding noise -- and shifting the belief on it adds error.
SCAN_FIT_MIN_IMPROVEMENT = 0.9

# Ceiling on the lateral correction applied to the belief in one sweep. Same
# role as RANGE_FIX_MAX_CORRECTION_M: beyond this the fit is claiming the kart
# is most of a cell from where odometry says, which is a wrong-cell hypothesis
# rather than drift, and acting on it would teleport the belief.
SCAN_FIT_MAX_LATERAL_M = 0.10

# ---------------------------------------------------------------------------
# THE OPEN RISK IN THIS WHOLE FEATURE: the beam is a cone, not a ray
#
# Everything above is measured against raycasts -- one ray per sample, hitting
# exactly what it points at. A real HC-SR04 returns the NEAREST echo anywhere in
# a ~15 degree cone, and `scan_fit` predicts ranges along the ray. Modelling the
# cone (nearest of several rays across +-7.5 deg) and re-running the tcc.maze
# study, lateral error against a true offset:
#
#     cone half-angle   median   p90     fits with >3 cm error
#     0.0 deg (ray)     0.25 cm  0.81 cm    0.0%
#     2.0 deg           0.63 cm  1.41 cm    1.1%
#     4.0 deg           1.28 cm  3.08 cm   11.5%
#     7.5 deg (real)    2.48 cm  4.82 cm   37.8%
#
# At the real cone width better than a third of fits are grossly wrong, and a
# gross fit is acted on -- the caller shifts the belief by it. Raising
# RANGE_FIX_MAX_TRUSTED_M to 0.90 was tried: it lifts the tcc.maze yield from
# 7/17 to 12/17 cells and costs nothing under ideal rays, but under the cone
# model it is no better (46/107 gross vs 37/98), because the cone, not the
# range ceiling, is what dominates. So the ceiling stays at 0.45.
#
# What this means practically: the numbers above are an upper bound on how well
# the sweep can work, and the bench will land somewhere between the ray and
# cone rows. Before trusting the lateral correction on hardware, measure the
# real sensor's spread by parking the kart at a known offset and sweeping --
# if it looks like the 7.5 deg row, the fit needs a cone-aware forward model
# (predict the min over a fan, not a single ray) rather than tighter gates.
# That is a bigger change than this one and is why it has not been made here.
# ---------------------------------------------------------------------------

# Frames spent looking at one sweep heading before moving on. Telemetry repeats
# its cached front reading between sensor updates, so "wait until we have N
# distinct readings" can wait forever -- and a heading pointed at open space
# never yields one at all. This bounds both cases; a heading that produced at
# least one real reading still contributes a sample from however many it got.
SWEEP_MAX_LOOKS_PER_ANGLE = 30

# How long the fit's search parks the worker thread at a row boundary, so the
# render loop gets the GIL. Measured against a CPU-bound worker, worst wait for
# the main thread: no yield 17.5 ms, time.sleep(0) 26.0 ms (worse -- it does not
# force a switch in CPython 3.14 and adds a syscall), time.sleep(5e-4) 2.8 ms.
# Small enough that the fit still finishes in well under a second.
SCAN_FIT_YIELD_S = 0.0005