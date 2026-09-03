import os

from dataclasses import dataclass

_SRC  = os.path.dirname(os.path.abspath(__file__))   # maze_solver/src
_APP  = os.path.dirname(_SRC)                        # maze_solver
_ROOT = os.path.dirname(_APP)                        # orquestrator

@dataclass
class PATHS:
    SRC    = _SRC
    APP    = _APP
    ROOT   = _ROOT
    COPPELIA_BINARY = r"C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu\coppeliaSim.exe"
    KART_SHADOW = os.path.join(_ROOT, "kart_shadow.ttt")
    ASSETS = os.path.join(_ROOT, "assets")
    MAZES  = os.path.join(_ROOT, "saecollab", "mazes")

@dataclass
class LOG:
    WARNING = 0x4
    LEVEL   = WARNING
    
@dataclass
class WINDOW:
    WIDTH  = 1280 
    HEIGHT = 720
    TITLE  = "Maze_Solver"

@dataclass
class HUD:
    HEIGHT = 118
    PAD    = 16
    LINES  = (10, 40, 62, 84)   # deslocamento vertical de cada linha do painel

@dataclass
class SIM:
    SPEEDS        = (0.02, 0.05, 0.10, 0.20, 0.40, 0.80)   # segundos por passo
    DEFAULT_SPEED = 2
    DEFAULT_MAZE  = os.path.join(PATHS.MAZES, "small_eg.maze")
    KART_CELLS    = 1.4        # maior lado do kart, em celulas
