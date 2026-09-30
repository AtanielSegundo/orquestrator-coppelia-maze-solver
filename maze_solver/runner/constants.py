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
# A `front_distance_cm` telemetry reading at or below this marks whichever
# cell the obstacle actually falls in as blocked in the LIVE maze (see
# LiveMazeRunner._scan_obstacle). The maze's own walls trip this too, which is
# harmless -- they are already walls, so the mark is a no-op.
OBSTACLE_DISTANCE_CM = 45.0

# How far ahead of the kart's tracked CENTER the front sensor actually sits.
# The reading is measured from the sensor, so this is added back before the
# hit point is projected onto the grid; otherwise every obstacle lands about
# half a kart-length too close and can be blamed on the wrong cell.
SENSOR_FORWARD_OFFSET_M = 0.13

# The other half of every front reading: whatever the beam passed THROUGH on
# its way to the hit point has to be empty, or it would have stopped earlier.
# Cells discovered blocked that now read as clear are un-discovered, which is
# what lets a run resume by itself once an obstacle is physically taken away.
# A cell is only cleared after this many consecutive clear looks, so one bad
# reading (or one frame of heading jitter) cannot erase a real obstacle.
SENSOR_CLEAR_CONFIRMATIONS = 8

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
# alignment tolerance) a 0.30 m reading is off by 0.30*(1/cos(12deg)-1) ~= 7 mm,
# while at 1.20 m the same angle costs ~27 mm -- more than the whole drift
# being corrected.
RANGE_FIX_MAX_TRUSTED_M = 0.30

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
