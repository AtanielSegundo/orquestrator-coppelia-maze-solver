import os
import pyray as pr

from dataclasses import dataclass

from src.definitions import PATHS

@dataclass
class HUDColors:
    background   = pr.Color(24, 26, 32, 255)
    goal_cell    = pr.GREEN
    open_cell    = pr.WHITE
    wall_cell    = pr.RED
    obstacle_cell = pr.Color(240, 140, 40, 255)   # found at runtime, not in the .maze
    start_cell   = pr.BLUE
    outline_cell = pr.BLACK
    path_cell    = pr.Color(90,  150, 240, 110)
    trail_cell   = pr.Color(250, 204, 62,  110)
    panel        = pr.Color(18,  20,  25,  255)
    panel_line   = pr.Color(60,  64,  76,  255)
    text         = pr.Color(238, 240, 246, 255)
    text_dim     = pr.Color(150, 154, 166, 255)
    text_ok      = pr.Color(96,  214, 140, 255)
    text_fail    = pr.Color(238, 104, 104, 255)

@dataclass
class Kart:
    file_path = os.path.join(PATHS.ASSETS, "kart_no_bg.png")
    tint = pr.WHITE
