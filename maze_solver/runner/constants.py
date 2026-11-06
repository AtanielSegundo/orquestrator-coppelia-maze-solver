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
                      (e, parado no goal, confirma o caminho do novo goal)
  start <lin> <col>   antes do 'ok': novo start | ao vivo: novo waypoint
  goal  <lin> <col>   novo goal (antes ou durante a corrida)
                      parado no goal: start = celula do kart, planeja e
                      espera 'ok' (start/goal fazem o mesmo nesse caso)
  resume              esquece TODOS os obstaculos descobertos de uma vez e
                      replaneja
  camfix              mede a pose com a camera ali mesmo, sem aplicar nada
                      (bancada; imagens de debug em maze_solver/logs/camera)
  status              imprime o estado atual
  help | quit
"""

CONTROLS = """
CONTROLES DA JANELA
  botao esquerdo   equivalente a 'start <lin> <col>' da celula clicada
  botao direito    equivalente a 'goal  <lin> <col>' da celula clicada
                   (parado no goal, qualquer botao escolhe o novo goal)
  setas            move a camera pan/tilt do kart (real + sim)
  C                centraliza a camera (olhando pra frente)
  ESC              sair
"""

COMMANDS = {
    "ok":     (0, 0),
    "start":  (2, 2),
    "goal":   (2, 2),
    "resume": (0, 0),
    "camfix": (0, 0),
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
OBSTACLE_DISTANCE_CM = 105.0

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
OBSTACLE_CONFIRM_S = 1.0

# How long the runner holds in front of a suspect waiting for that verdict
# before it gives up and drives on. Reached when the steady kart never gets
# a usable reading of the cell (no echo, misaligned, stopped in the wrong
# cell). The suspect is then ignored for the next leg so the kart cannot loop
# stop -> hold -> go -> stop on it; if something really is there, the
# server's own front guard still stops the kart, and the steady check maps it
# from wherever it stopped.
OBSTACLE_SUSPECT_TIMEOUT_S = 4.0

# How far ahead of the kart's tracked CENTER the front sensor actually sits.
# The reading is measured from the sensor, so this is added back before the
# hit point is projected onto the grid; otherwise every obstacle lands about
# half a kart-length too close and can be blamed on the wrong cell.
SENSOR_FORWARD_OFFSET_M = 0.10

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
RANGE_FIX_MAX_CORRECTION_M = 0.20

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
RECENTER_MAX_STEP_M = 0.20

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
RECENTER_MAX_ATTEMPTS = 6

# Pause after a move before re-measuring. An ultrasonic reading taken while the
# chassis is still rocking is what would make this oscillate instead of
# converge.
RECENTER_SETTLE_S = 0.5

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
SWEEP_EVERY_N_LEGS = 1

# Longest straight run driven as ONE leg. Consecutive path cells that keep
# the same heading are merged into a single goto -- (r, c) -> (r, c+3) heading
# east is one move to the far cell's centre, with no stop at the two cells in
# between. Longer straight runs are cut into pieces of at most this many
# cells. 1 restores the old cell-by-cell driving.
#
# Starts equal to SWEEP_EVERY_N_LEGS so a full straight run ends where a
# sweep is due, but it is its own knob: tying it to the sweep interval would
# make tuning one silently change the other.
LEG_MAX_CELLS = 2

# ---------------------------------------------------------------------------
# Closest obstacle as lateral reference (runner/closest_obstacle.py)
#
# FEATURE FLAG. True replaces the angular sweep above: at the same boundary
# (every SWEEP_EVERY_N_LEGS cells, after Recenter) the kart turns +-90 to face
# the closest .maze wall on the axis orthogonal to the leg, reads it, drives
# alpha * (measured - expected) toward/away from it, re-reads, shifts the
# anchor by what is left, and turns back. Turns alternate +90 / -90 between
# corrections. No usable wall on either side -> nothing happens and the next
# boundary tries again; the interval only restarts after a real reading.
#
# Reuses the sweep's settle/sampling knobs (SWEEP_SETTLE_S,
# SWEEP_SAMPLES_PER_ANGLE, SWEEP_MAX_LOOKS_PER_ANGLE, SWEEP_START_GRACE_S),
# Recenter's move speed/clamp, and SCAN_FIT_MAX_LATERAL_M as the "wrong wall"
# ceiling on the measured error.
#
# The wall must be READABLE from the cell: expected range between
# SENSOR_MIN_RANGE_CM and RANGE_FIX_MAX_TRUSTED_M. A wall in the ADJACENT
# cell is expected at cell/2 - SENSOR_FORWARD_OFFSET_M: 5 cm at 0.30 m cells
# and 0.10, readable; 2 cm at 0.13, inside the blind zone, so it would not
# qualify. A wall one open cell further (35 cm at 0.10) always does -- see
# NEXT_BEST_CLOSEST_AS_REFERENCE. Turned off, the sweep runs as before.
# ---------------------------------------------------------------------------
CLOSEST_OBSTACLE_AS_REFERENCE = True

# Fraction of the measured error actually driven. 1.0 drives the whole of it;
# below 1 under-corrects on purpose (useful if the move tends to overshoot).
# The anchor correction does not depend on it -- it comes from the re-read.
CLOSEST_OBSTACLE_ALPHA = 1.0

# Only move when alpha * error is larger than this. Below it the move is
# smaller than the sensor's whole-cm resolution can justify (and, like
# RECENTER_MIN_STEP_M, close to what the kart can execute at all).
CLOSEST_OBSTACLE_MINIMAL_DELTA_CM = 3.0

# Visit BOTH sides (+90 and -90) on every correction instead of only the
# preferred one -- each only if it has a readable obstacle; a side with none
# is never turned to. The anchor shift found on the first side is folded into
# the pose the second side is judged from, so the two refine one estimate
# rather than double-counting it. Costs a 180 deg turn (as two 90 deg steps)
# between the sides.
CLOSEST_OBSTACLE_ALWAYS_CHECKS_THE_TWO = False

# Postpone a due correction to a BETTER reference ahead on the path. Only
# acts with CLOSEST_OBSTACLE_AS_REFERENCE on.
#
# "Better" = a readable .maze wall with exactly ONE empty cell between it and
# the kart's cell, on the axis orthogonal to the heading the kart arrives
# with: e.g. kart at (r, c) heading north, (r, c-1) empty, (r, c-2) wall. At
# 0.30 m cells that reads 0.45 - SENSOR_FORWARD_OFFSET_M (35 cm at 0.10),
# against 5 cm for an adjacent wall, right at the edge of the blind zone.
#
# When the correction is due and the current cell has no better reference,
# the next MAX_NEXT_BEST_CELLS cells of the plan are checked. If one has it,
# legs drop to ONE cell (LEG_MAX_CELLS is overridden) until that cell is
# reached; there the correction runs facing that better wall -- even if it is
# the same turn direction as last time -- then LEG_MAX_CELLS goes back to its
# value and the SWEEP_EVERY_N_LEGS count restarts. Nothing better ahead: the
# correction runs here, as without this flag. A replan drops the plan for it.
NEXT_BEST_CLOSEST_AS_REFERENCE = True
MAX_NEXT_BEST_CELLS = 8

# ---------------------------------------------------------------------------
# The fit itself (runner/scan_fit.py)
# ---------------------------------------------------------------------------
# Half-width of the (dx, dy) search window. Just over half a cell: a genuine
# error larger than this is a lost pose rather than drift, and a wider window
# mostly lets the search find a convincing wrong answer against the next cell's
# walls.
SCAN_FIT_MAX_SHIFT_M = 0.20

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
SCAN_FIT_MIN_SAMPLES = 4

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


# ---------------------------------------------------------------------------
# Camera pose fix (runner/camera_fix.py, runner/camera_fit.py)
#
# The camera sees where each wall stands on the floor (the floor-contact
# profile, camera_link/contact_profile.py); that moves in the image when the
# kart moves sideways or turns, and the pan servo turns the CAMERA instead of
# the kart to look at the side walls. It resolves millimetres where the sonar
# resolves centimetres.
#
# It works WITH the sonar correctors, not instead of them. At every boundary:
#
#     Recenter -> [closest obstacle reference | sweep, when due] -> camera
#
# The sonar correctors keep their own schedule and make the BIG corrections
# (centimetres, physical moves); on the real tcc.maze, CLOSEST_OBSTACLE_AS_
# REFERENCE alone holds for about three full 12-15 cell runs before the error
# it leaves behind starts to matter. The camera runs last and makes only the
# SMALL ones (CAMERA_FIX_SMALL_*): what the sonar leaves below its own
# thresholds and quantisation, plus the heading error of every turn made at
# the boundary -- the error that makes those runs drift in the first place.
# ---------------------------------------------------------------------------
CAMERA_FIX_ENABLED = True

# Run one fix right after 'ok', before the first leg: it corrects the
# operator's placement, and doubles as a health check of the calibration (a
# kart placed carefully on the start cell should measure close to zero).
CAMERA_FIX_AT_START = True

# Hold the tilt servo at CAMERA_FIX_TILT_PCT for the whole run instead of
# returning to the level home tilt between fixes. Tilt repeatability then stops
# mattering: the head never leaves the tilt it was calibrated at.
CAMERA_FIX_HOLD_TILT = True

# Calibration file read by default (runner/camera_setup.py), relative to the
# orquestrator directory. Missing is fine: the hardware defaults below are
# used and the HUD says so.
CAMERA_CALIB_DEFAULT_PATH = "camera_link/camera_calib.json"

# -- Hardware: Raspberry Pi Camera rev 1.3 (OV5647) --------------------------
# Server/proc_camera.py asks Picamera2 for a 400x300 video stream. libcamera
# serves that from the OV5647's 640x480 mode, which reads a 2560x1920 window
# of the sensor (binned) and the ISP scales it to 400x300. Pixel pitch is
# 1.4 um and the lens is the stock 3.6 mm, so one stream pixel is
# 2560*1.4/400 = 8.96 um of sensor and
#
#     f = 3.6 mm / 8.96 um = 401.8 px   (HFOV 52.9 deg, VFOV 41 deg)
#
# which matches the datasheet's 53.5 x 41.4 deg for the full sensor. The mode
# table is the kart's own (`rpicam-hello --list-cameras`): 640x480 reads
# (16, 0)/2560x1920. Were Picamera2 to pick the 1296x972 mode instead (the whole
# 2592 px width), f would be 397 px -- 1.2% off, inside what intrinsics
# calibration corrects anyway. These are
# the defaults until `camera_calibrate.py intrinsics` measures the real lens;
# the stock lens is close to distortion-free at this resolution.
#
# The kart's module is a clone sold as "OV5647 5MP, 3.6 mm, 60 deg". The 3.6 mm
# is what is used here; the 60 deg fits neither axis (60 deg horizontal would be
# f = 346 px, 60 deg diagonal f = 433 px, while 3.6 mm gives 53 x 41 deg, 64
# diagonal), so it is taken as a rounded marketing figure. The lens is a screw
# mount whose focus also shifts f a little -- measure it, and do not turn the
# lens after calibrating.
CAMERA_STREAM_SIZE = (400, 300)
CAMERA_FOCAL_MM = 3.6
CAMERA_PIXEL_UM = 1.4
CAMERA_SENSOR_CROP_PX = 2560

# -- Mount ------------------------------------------------------------------
# Lens position relative to the kart's odometry centre (the (0, 0) the pose
# tracks): 13 cm ahead, on the centre line, 10 cm above the floor -- the same
# direction as the sonar. Measured at the fix tilt.
CAMERA_LENS_FORWARD_M = 0.13
CAMERA_LENS_LATERAL_M = 0.0
CAMERA_LENS_HEIGHT_M = 0.10

# Distance from the pan servo's axis to the lens, along the optical axis. A
# side view moves the lens sideways by arm*sin(pan) -- 1.8 cm at 45 deg with
# 2.5 cm -- which is the size of the error being measured, so it is modelled.
# MEASURE IT: ruler from the pan shaft to the lens.
CAMERA_PAN_ARM_M = 0.025

# -- Servo head -------------------------------------------------------------
# One CMD_SERVO unit is one servo degree: Server/servo.py drives
# 500 + (angle + 10) / 0.09 us, i.e. 11.1 us per unit on a 2000 us / 180 deg
# servo. Home values are CameraMirror's (camera level and centred).
CAMERA_PAN_HOME_PCT = 50
CAMERA_TILT_HOME_PCT = 37
CAMERA_SERVO_DEG_PER_PCT = 1.0
# Direction conventions. Check both on the bench with the overlay tool -- a
# wrong sign shows as a mirrored or inverted model, not as a subtle error.
# Tilt measured on the real kart (2026-11-04): 37 is level, 62 looks down at
# the floor and 12 looks UP at the ceiling, so a HIGHER tilt is down -- the
# opposite of the arrow keys' "DOWN lowers tilt" in main.py.
CAMERA_PAN_LEFT_IS_HIGHER = True
CAMERA_TILT_DOWN_IS_LOWER = False

# Tilt held during fixes. Level, the floor contacts nearer than ~0.33 m fall
# below the image, and those are the ones with the best resolution: with the
# lens only 10 cm up, a contact moves f*h/r^2 px per metre -- 4 px/cm at
# 0.3 m but 0.5 px/cm at 0.9 m. Pitched 25 deg down (62 = 37 + 25) the image
# spans the floor from ~0.10 m ahead of the lens to past the range limit.
CAMERA_FIX_TILT_PCT = 62

# Views, as (name, pan in degrees relative to home, + = left). The side views
# look straight at the current cell's own side walls, which is where lateral
# offset shows best (a wall's contact row moves directly with it); the forward
# view defines yaw. HFOV is ~53 deg, so +-45 covers +-72 deg around the leg.
CAMERA_FIX_VIEWS = (("direita", -45.0), ("frente", 0.0), ("esquerda", 45.0))

# Rows at the bottom of each view excluded from the profile (kart body, sonar
# module). Per view in the calibration file once measured.
CAMERA_FIX_IGNORE_BOTTOM_PX = 20

# Every view is approached from the same side (from this many units below the
# lowest pan), so servo backlash lands each view in the same place every time.
CAMERA_FIX_PREPOSITION_PCT = 6

# -- Timing -----------------------------------------------------------------
# Same reasoning as RECENTER_SETTLE_S: let the chassis stop rocking.
CAMERA_FIX_KART_SETTLE_S = 0.30
# Head ringing after a pan move. Re-measure on the bench.
CAMERA_FIX_SERVO_SETTLE_S = 0.50
# Capture -> JPEG -> WiFi -> decode. A frame is only used if it was DECODED at
# least this long after the head settled, so it was taken after it too.
CAMERA_FIX_FRAME_LATENCY_S = 0.20
# Wait for the kart to report the servo command (servo telemetry) at most
# this long; a server that does not publish it just skips the wait.
CAMERA_FIX_SERVO_ACK_TIMEOUT_S = 1.0
# A view that has not produced its frames by then is skipped.
CAMERA_FIX_VIEW_TIMEOUT_S = 2.5
# Frames combined per view (median profile): one blurred frame cannot move it.
CAMERA_FIX_FRAMES_PER_VIEW = 3

# -- Profile and model ------------------------------------------------------
# Beyond this floor range (from the point under the lens) a contact carries
# almost no information at this camera height (see CAMERA_FIX_TILT_PCT), and
# the map stops being a trustworthy description of the room.
CAMERA_FIX_MAX_FLOOR_RANGE_M = 0.90
CAMERA_FIX_COLUMN_STEP_PX = 4
CAMERA_FIX_ROW_STEP_PX = 2
CAMERA_FIX_MIN_RUN_PX = 3
# Bearing resolution of the visibility raycast: 0.15 deg is 2.4 mm at 0.9 m
# and 0.8 mm at 0.3 m, under a pixel everywhere in range.
CAMERA_FIX_BEARING_STEP_DEG = 0.15

# Wall/floor colour model (camera_link/floor_segmenter.py). The margin is how
# far from a PREDICTED boundary a pixel must be to train on, larger than what
# the pose error moves a boundary: ~4 px per cm near the kart, and f*tan(yaw)
# sideways for a heading error -- 21 px at 3 deg, which is what a turn leaves
# (the 12 px it was sized at for translation alone let vertical wall edges
# into the wrong class in the closed loop).
CAMERA_SEG_COMPONENTS = 2
CAMERA_SEG_L_WEIGHT = 1.0 / 3.0
CAMERA_SEG_LABEL_MARGIN_PX = 24
# Self-training rounds after the seed (floor_segmenter.FloorSegmenter.train):
# the model's own confident pixels (|LLR| > 2) retrain it, so the labels come
# from colour and the map only says where the maze is.
CAMERA_SEG_REFINE_ITERS = 2
CAMERA_SEG_MIN_PIXELS = 300
CAMERA_SEG_MIN_ACCURACY = 0.85

# -- Cost -------------------------------------------------------------------
# Residual per column = gap between the measured and the predicted contact
# intervals, in units of PIXEL_SIGMA; Huber beyond HUBER_K, flat beyond
# TRUNCATE_K so a person or a bag in view is a bounded outlier.
CAMERA_FIX_PIXEL_SIGMA_PX = 1.5
CAMERA_FIX_HUBER_K = 2.0
CAMERA_FIX_TRUNCATE_K = 6.0
# Neighbouring columns see the same wall edge, so their errors are not
# independent. The cost and the information are divided by this.
CAMERA_FIX_COLUMN_CORRELATION = 3.0
# Systematic row error assumed per VIEW (px), on top of the column noise:
# a whole view's boundary can sit a pixel high or low (segmentation, JPEG's
# half-resolution chroma, tilt repeatability). Propagated into the reported
# std (camera_fit._bias_cov), so a geometry where one view's offset can be
# traded for lateral or yaw is reported unobservable instead of confidently
# wrong. 0.7 px: the median per-view offset measured in the closed loop after
# the segmentation fixes, which ranged 0.2 - 1.0 px.
CAMERA_FIX_VIEW_BIAS_PX = 0.7

# -- Search -----------------------------------------------------------------
# Window: under half a cell, like SCAN_FIT_MAX_SHIFT_M -- a larger error is a
# lost pose, and a wider window mostly finds the next cell's walls.
CAMERA_FIX_SEARCH_HALF_M = 0.10
CAMERA_FIX_SEARCH_HALF_YAW_DEG = 8.0
CAMERA_FIX_COARSE_STEP_M = 0.02
CAMERA_FIX_COARSE_STEP_YAW_DEG = 1.0
# The coarse grid scores with the pixel sigma multiplied by this. Near a side
# wall a contact moves ~4 px per cm of lateral, so a 2 cm step moves it ~8 px:
# at the real sigma every candidate but the exact one is past the truncation,
# the cost is a spike the grid steps over, and the coarse optimum is whichever
# wrong candidate happens to clip fewest columns. Measured on the synthetic
# corridor: at scale 1 the coarse stage missed a 3 cm / 2 deg offset outright;
# widened, the basin is several grid steps wide and the fine stage (real sigma)
# then resolves it.
CAMERA_FIX_COARSE_SIGMA_SCALE = 4.0
# Local minima of the coarse stage finished independently by Levenberg-
# Marquardt; the cheapest result wins (see camera_fit.CameraFitter._basins).
CAMERA_FIX_FINE_STARTS = 3
# LM iterations per sigma level (coarse-scaled, 2x, real).
CAMERA_FIX_LM_ITERATIONS = 12
# Prior on the kart's yaw error. Honest, not tight: turns err by 1-5 deg (see
# SWEEP_EVERY_N_LEGS), and against a single wall seen face-on lateral and yaw
# trade off -- a tight prior would turn real yaw error into a confident wrong
# lateral.
CAMERA_FIX_YAW_PRIOR_DEG = 3.0
# Weak prior on translation, so a direction the data cannot see (longitudinal
# in a corridor) settles at zero instead of wherever the search started.
CAMERA_FIX_TRANSLATION_PRIOR_M = 0.08
# Side views' pan is uncertain by about this much (servo repeatability); the
# fit absorbs it per view instead of reading it as kart yaw. 0 disables.
CAMERA_FIX_PAN_NUISANCE_SIGMA_DEG = 1.0
CAMERA_FIX_PAN_NUISANCE_HALF_DEG = 2.0

# -- Gates ------------------------------------------------------------------
CAMERA_FIX_MIN_HITS_PER_VIEW = 12
CAMERA_FIX_MIN_HITS_TOTAL = 30
CAMERA_FIX_MIN_INLIER_FRACTION = 0.6
CAMERA_FIX_MAX_RMS_PX = 2.0
# Marginal standard deviation above which an axis is reported unobservable.
CAMERA_FIX_MAX_STD_LAT_M = 0.008
CAMERA_FIX_MAX_STD_LONG_M = 0.010
CAMERA_FIX_MAX_STD_YAW_DEG = 0.6
CAMERA_FIX_MAX_LATERAL_M = SCAN_FIT_MAX_LATERAL_M
CAMERA_FIX_MAX_LONG_M = 0.10
CAMERA_FIX_MAX_YAW_DEG = 7.0
# A second lateral hypothesis at least this far away must cost clearly more
# (in noise units) than the best one, or lateral is ambiguous.
CAMERA_FIX_AMBIGUITY_SEPARATION_M = 0.03
CAMERA_FIX_AMBIGUITY_MIN_DELTA = 4.5
# The fit must explain the image this much better than the uncorrected pose.
CAMERA_FIX_MIN_IMPROVEMENT = 0.8
# Below these -- or below twice the component's own standard deviation, when
# that is larger -- the pose is "confirmed": nothing is applied. A dead-band,
# so a well-centred kart is not re-anchored by noise at every cell; small
# because fine corrections are the whole job now (see CAMERA_FIX_SMALL_*).
CAMERA_FIX_CONFIRM_M = 0.004
CAMERA_FIX_CONFIRM_YAW_DEG = 0.25
CAMERA_FIX_YAW_GAIN = 1.0

# -- Small vs big (runner/arrival_policy.py) ---------------------------------
# The largest correction the camera applies from one fix.
#
# Translation is shared with the sonar correctors. A camera reading beyond
# these is not applied; with CAMERA_FIX_ESCALATES the closest obstacle
# reference / sweep runs right away, at this same boundary -- the tool that
# moves the kart physically and has been proven on the real maze. Only when it
# cannot run here (no usable side wall, a better one scheduled ahead) does the
# camera apply its reading clamped to these, so a known error is not left
# standing and a wrong fit cannot move the belief by more than this.
#
# 3 cm sits where the two meet: the closest obstacle reference only moves the
# kart above CLOSEST_OBSTACLE_MINIMAL_DELTA_CM (3 cm) and only re-anchors
# above RANGE_FIX_MIN_CORRECTION_M (2 cm), so everything under 3 cm is left
# to the camera.
#
# Yaw is the camera's alone -- no sonar corrector observes heading -- so it is
# never escalated, only clamped to CAMERA_FIX_SMALL_YAW_DEG per fix. 3 deg is
# about one turn's worth of error (1-5 deg/turn, see SWEEP_EVERY_N_LEGS); a
# larger accumulated error is worked down over successive boundaries. That is
# also why the search window (CAMERA_FIX_SEARCH_HALF_YAW_DEG) and the
# plausibility gate (CAMERA_FIX_MAX_YAW_DEG) reach past it: a heading 5 deg off
# has to be MEASURABLE to be worked down at all -- measured in the closed loop,
# with a 6 deg window an accurate -5.5 deg reading was refused on the window
# edge, and from then on nothing could correct the heading.
CAMERA_FIX_SMALL_LATERAL_M = 0.03
CAMERA_FIX_SMALL_LONG_M = 0.03
CAMERA_FIX_SMALL_YAW_DEG = 3.0
CAMERA_FIX_ESCALATES = True

# -- Step mode (replaces the gates and the small-vs-big table above) ---------
# The camera only ever NUDGES the belief: every fix applies a bounded fraction
# of its estimate, whatever the fit's confidence, and never escalates. The
# reference swapping (closest obstacle reference / sweep) already removes the
# gross errors; inside the 2 cm radius around the cell centre the kart is
# still well localised, so a step of at most CAMERA_FIX_STEP_M can never push
# a well-placed kart out of it on its own, and a consistent reading is worked
# down over successive boundaries. Full precision is traded for that bound.
#
# With step mode on, the fit's quality gates (observability, inliers, window
# edge, ambiguity, magnitude, improvement) are only reported, not enforced;
# a fit is refused only when there is no data at all (no usable view, too few
# contacts). False restores the gated behaviour above.
CAMERA_FIX_STEP_MODE = True
# Fraction of the estimate applied per fix, per axis.
CAMERA_FIX_STEP_GAIN_LONG = 0.5
CAMERA_FIX_STEP_GAIN_LAT = 0.5
CAMERA_FIX_STEP_GAIN_YAW = 0.5
# Largest step per fix: translation (norm of long+lat) and yaw. Both halved
# (1 cm -> 0.5 cm, 2 deg -> 1 deg) after bench runs on the real kart
# (2026-11-05). Trade-off: in tests/test_camera_closed_loop.py (3 deg/turn) a
# 1 deg yaw cap let the departure from the centre line reach ~2.5 cm, where
# 2 deg kept it under 2 cm (that test was tuned with 1 cm / 2 deg).
CAMERA_FIX_STEP_M = 0.005
CAMERA_FIX_STEP_YAW_DEG = 1.0
# Sign of the applied yaw step: +1 corrects in the direction measured (what
# the simulated closed loop wants); -1 turns a +theta reading into a -theta
# correction, for trying an inverted sign on the real kart.
CAMERA_FIX_STEP_YAW_SIGN = -1.0

# -- Abort ------------------------------------------------------------------
# The kart is presumed still from capture to the end of the fix.
CAMERA_FIX_KART_MOVED_M = 0.005
CAMERA_FIX_KART_MOVED_DEG = 0.5

# -- Physical centring after a fix (runner/center_move.py) -------------------
# Smallest corrected lateral offset from the cell centre worth driving out.
CAMERA_FIX_LATERAL_MOVE_M = 0.03
# When to drive it out:
#   "always"        every fix over the threshold
#   "before_turns"  only when the next leg starts with a turn of 45 deg or more,
#                   or at the goal -- a straight next leg already removes the
#                   offset with its own absolute goto (~6 deg turn1 for 3 cm
#                   over 30 cm), while centring in place costs two turns
#   "never"
# "never" now that the camera only applies small corrections: physical moves
# for large offsets are the closest obstacle reference's (it drives toward or
# away from the side wall), and an offset under CAMERA_FIX_SMALL_LATERAL_M is
# removed by the next leg's absolute goto anyway.
CAMERA_FIX_MOVE_POLICY = "never"
# How: "goto" sends the absolute goto to the centre straight away, which for a
# mostly-lateral offset is turn ~90, creep, turn ~90 back -- the very turns
# that create heading error. "backoff" first reverses CAMERA_FIX_BACKOFF_M
# (gyro-held heading), so the goto approaches the centre at ~15 deg instead.
CAMERA_FIX_MOVE_MODE = "backoff"
CAMERA_FIX_BACKOFF_M = 0.12
# Arrival tolerance of the centring goto, like Leg.ARRIVE_TOLERANCE_M.
CAMERA_FIX_MOVE_ARRIVE_M = 0.03
# After the centring move, look again (belief only, never a second move).
CAMERA_FIX_REVERIFY = True

# -- Diagnostics ------------------------------------------------------------
# Per-view PNGs (frame, measured contacts, model before/after) and the per-
# arrival CSV, under maze_solver/logs/.
CAMERA_FIX_DEBUG_DUMP = True
ARRIVAL_LOG_ENABLED = True


# ---------------------------------------------------------------------------
# Camera pose fix v2 (runner/camera_fix_v2.py, runner/edge_match.py,
# runner/camera_cells.py)
#
# Replaces the per-boundary v1 fix above when CAMERA_V2_ENABLED. The head
# looks UP (box tops) and DOWN (floor contacts) at the centre pan, then at the
# right pan, then at the left pan. Every box edge seen is compared with the
# same edge in the maze ray caster's image at the believed pose: per point,
# how far the real edge sits from the simulated one along the edge normal.
#
# Grouping: the centre pan's points split into the LEFT and RIGHT half planes
# of the image; each side pan is one group; upper and lower edges stay apart.
# A shift common to the two halves (and to the two side pans) is yaw; halves
# moving apart / together (and the side pans moving opposite ways) is a
# longitudinal offset. The numbers come from each point's sensitivity to yaw,
# longitudinal and lateral, computed from the simulation itself.
#
# Then a STEP: the yaw is corrected by TURNING the kart (belief re-anchored to
# the measured heading, then a turn back onto the grid axis), the longitudinal
# only by re-anchoring the belief; lateral is estimated as a nuisance and left
# to the closest obstacle reference. Measure again, until under the acceptance
# thresholds or out of iterations.
#
# Slow (6 views per iteration, plus turns), so it is SCHEDULED like
# NEXT_BEST_CLOSEST_AS_REFERENCE: due every CAMERA_V2_EVERY_N_CELLS cells, run
# at a cell whose geometry suits it -- the current one if it does, else the
# first on the plan within CAMERA_V2_MAX_NEXT_BEST_CELLS (legs drop to one cell
# until there). Nothing suitable ahead: it stays due and the next boundary
# looks again.
# ---------------------------------------------------------------------------
CAMERA_V2_ENABLED = True
CAMERA_V2_AT_START = True

# Head positions. Pans are the calibration's pan views; tilts are named
# TiltCalibration entries -- "baixo" is the held tilt (CAMERA_FIX_TILT_PCT),
# "cima" this many units. Visiting order is the order below, every time, so
# servo hysteresis lands each view where `extrinsics` calibrated it.
# 8: on the real kart, 15 left the top of the box two cells ahead (~32 deg up
# from the lens) at the very top of the frame -- the tilt servo turns only
# ~0.5 deg per unit up there. Until `extrinsics` measures it, this tilt's pitch
# is derived with `tilt_deg_per_pct`, which is far off; the search of
# `extrinsics` (--span-tilt) is wide for that reason.
CAMERA_V2_TILT_UP_PCT = 8
CAMERA_V2_PAN_ORDER = ("frente", "direita", "esquerda")
CAMERA_V2_TILT_ORDER = ("cima", "baixo")
# The pan whose points split into half planes (the one facing the leg).
CAMERA_V2_SPLIT_PAN = "frente"

# -- Scheduling (runner/camera_cells.py) ------------------------------------
CAMERA_V2_EVERY_N_CELLS = 3
CAMERA_V2_MAX_NEXT_BEST_CELLS = MAX_NEXT_BEST_CELLS
# A good cell, judged from the heading the kart arrives with: the cell ahead
# open (the camera needs distance), at least this many original walls among
# front-left / front-right one cell ahead and the three cells two ahead, and
# -- with REQUIRE_CORRIDOR -- walls on both sides of the kart's cell or of the
# cell ahead.
CAMERA_V2_MIN_FRONT_OBSTACLES = 2
CAMERA_V2_REQUIRE_CORRIDOR = True

# -- Edges and matching (runner/edge_match.py) ------------------------------
# One simulated edge point every this many pixels along each edge.
CAMERA_V2_EDGE_STEP_PX = 3
# Edge points this close to the image border are not used.
CAMERA_V2_EDGE_BORDER_PX = 6
# Edges of boxes further than this from the lens are not used (small, and the
# segmentation of a far box is unreliable).
CAMERA_V2_MAX_EDGE_RANGE_M = 2.0
# How far along the normal the real edge is searched for, px. Covers ~4 deg of
# yaw at f ~ 380 px; a larger error is the closest obstacle reference's.
CAMERA_V2_SEARCH_PX = 30
# Robust loss on the per-point offsets: Huber beyond this, ignored beyond
# CAMERA_V2_OUTLIER_K times it.
CAMERA_V2_HUBER_PX = 2.0
CAMERA_V2_OUTLIER_K = 3.0
# Neighbouring points see the same edge; the information is divided by this.
CAMERA_V2_POINT_CORRELATION = 3.0
CAMERA_V2_MIN_POINTS_PER_GROUP = 15
CAMERA_V2_MIN_POINTS = 60
# A group whose points the fitted pose leaves this far off (median, px) is an
# outlier (a person, a bag, a misread colour) and the fit is redone without it.
CAMERA_V2_GROUP_MAX_RESIDUAL_PX = 4.0
# Gauss-Newton passes per measurement: render at the estimate, re-match.
CAMERA_V2_INNER_ITERATIONS = 3
# Colour model: walls vs EVERYTHING ELSE (floor, the room above the boxes).
CAMERA_V2_SEG_COMPONENTS = 3

# -- Priors -----------------------------------------------------------------
CAMERA_V2_YAW_PRIOR_DEG = 5.0
CAMERA_V2_LONG_PRIOR_M = 0.05
CAMERA_V2_LAT_PRIOR_M = 0.02
# Side pans' servo repeatability, shared by both tilts of that pan.
# 2 deg: the pan servo lands in ~10 deg jumps (feature sweep on the real kart);
# even with the fixed visiting order a side view can land a degree or two off.
CAMERA_V2_PAN_NUISANCE_SIGMA_DEG = 2.0
# A pitch error shared by every view (calibration drift, kart not level).
CAMERA_V2_PITCH_NUISANCE_SIGMA_DEG = 1.0
# Coarse yaw grid tried before the fine fit: +- this, in these steps.
CAMERA_V2_COARSE_YAW_DEG = 4.0
CAMERA_V2_COARSE_YAW_STEP_DEG = 1.0
# An axis whose standard deviation is above this is not stepped.
CAMERA_V2_MAX_STD_YAW_DEG = 1.0
CAMERA_V2_MAX_STD_LONG_M = 0.02

# -- The measure -> step loop (runner/camera_fix_v2.py) ----------------------
# The job is not precision: it is to stop the residual error the reference
# swapping leaves from ACCUMULATING over a long run until the kart grazes a
# wall. So two measurements at most, and "good enough" accepted early.
CAMERA_V2_MAX_ITERATIONS = 2
# Accepted (no more steps) when both are under these.
CAMERA_V2_ACCEPT_YAW_DEG = 1.0
CAMERA_V2_ACCEPT_LONG_M = 0.01
# Step per iteration: gain x estimate, clamped.
CAMERA_V2_YAW_GAIN = 1.0
CAMERA_V2_LONG_GAIN = 1.0
CAMERA_V2_MAX_YAW_STEP_DEG = 3.0
CAMERA_V2_MAX_LONG_STEP_M = 0.01
# Turns smaller than this are not issued (the belief is still corrected; the
# next leg's absolute goto takes the rest).
CAMERA_V2_MIN_TURN_DEG = 0.8
# Sign of the applied yaw step. +1 is what the geometry says; if a real run
# keeps stopping with "yaw divergiu", the sign is the first suspect.
CAMERA_V2_YAW_SIGN = 1.0
# The yaw reading after a step must not grow by more than this (deg), or the
# loop stops: a wrong sign or a wrong match would otherwise be amplified.
CAMERA_V2_DIVERGE_DEG = 0.5
