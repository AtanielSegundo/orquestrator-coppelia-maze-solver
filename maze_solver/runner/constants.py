from runner.rgb_to_norm import rgb_to_norm

# ---------------------------------------------------------------------------
# Kart connection (same defaults as control_websocket.py)
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
  resume              tenta de novo depois de um obstaculo (estado blocked);
                      tambem limpa os obstaculos que o sensor descobriu
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

# An identical front reading that survives this much travel is FROZEN, not
# real: readings are whole cm, so closing even a few cm on something inside
# OBSTACLE_DISTANCE_CM has to change the number. Sized well above odometry
# jitter for a parked kart (which must stay trusted -- it is how a kart
# stopped in front of a real obstacle still maps it), and well under the
# travel needed to shift the projection into the next cell.
SENSOR_FROZEN_TRAVEL_M = 0.03
