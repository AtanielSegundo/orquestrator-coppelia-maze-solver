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
  resume              tenta de novo depois de um obstaculo (estado blocked)
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
