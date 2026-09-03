from typing import *
import sys
import pyray as pr

from saecollab.env.MazeEnv import (
    MazeEnv,
    GridCell
)
from src.styles import HUDColors
from src.definitions import HUD

class KartView():
    def __init__(self,path:str,tint:pr.Color,
                 angle_offset_to_match_ccw = 270
                 ):
        self._cpu_image   = pr.load_image(path)    
        self._gpu_texture = pr.load_texture_from_image(self._cpu_image)
        self._reference_angle = 0
        self._tint        = tint
        self._scale       = 1.0
        self._src_rect    = pr.Rectangle(0,0,self._gpu_texture.width,self._gpu_texture.height)
        self._dst_rect    = pr.Rectangle(0,
                                         0,
                                         self._scale*self._gpu_texture.width,
                                         self._scale*self._gpu_texture.height)    
        
        self._border = pr.get_image_alpha_border(self._cpu_image, 0.0)   # -> 165, 79, 216, 365
        self._pivot_u = (self._border.x + self._border.width  / 2) / self._gpu_texture.width   # 0.533
        self._pivot_v = (self._border.y + self._border.height / 2) / self._gpu_texture.height  # 0.511
        
        self._angle_offset_to_match_ccw = angle_offset_to_match_ccw
        
        self.recompute()
                
    def set_offset(self,x:int,y:int):
        self._dst_rect.x = x
        self._dst_rect.y = y
    
    def set_offset_centered(self):
        self.set_offset(pr.get_screen_width() // 2, pr.get_screen_height() // 2)
        
    def set_rotation(self,angle:float):
        self._reference_angle = self._angle_offset_to_match_ccw - angle
        
    def set_scale(self,scale:float):
        self._dst_rect.width  = scale * (self._dst_rect.width  / self._scale)
        self._dst_rect.height = scale * (self._dst_rect.height / self._scale)
        self._scale = scale
        self.recompute()
    
    def set_scale_to_pixels(self,pixels:float):
        """Escala o sprite para que o seu maior lado *visivel* (a alpha border,
        nao a textura inteira) ocupe `pixels` na tela."""
        biggest = max(self._border.width, self._border.height)
        if biggest > 0:
            self.set_scale(pixels / biggest)
    
    def recompute(self):
        self.origin = pr.Vector2(self._pivot_u * self._dst_rect.width,self._pivot_v * self._dst_rect.height)
         
    def render(self): 
        pr.draw_texture_pro(
            self._gpu_texture,
            self._src_rect,
            self._dst_rect,
            self.origin,
            self._reference_angle,
            self._tint
        )

class MazeView:
    def __init__(self,maze_path:str):
        self._env = MazeEnv(maze_path)
        if not self._env.file_loaded:
            raise FileNotFoundError(maze_path)
        self.set_start_goal(self._env.agent_start,self._env.agent_goal)
        self.recompute()

    def set_maze(self,maze_path:str) -> bool:
        m = MazeEnv(maze_path)
        if not m.file_loaded:
            print(f"[ERROR] Cant Load {maze_path}")
            return False
        del self._env
        self._env = m
        self.set_start_goal(m.agent_start,m.agent_goal)
        self.recompute()
        return True
    
    def set_start_goal(self,
                       start:Tuple[int,int],
                       goal:Tuple[int,int]):
        self._start = start
        self._goal  = goal
    
    @staticmethod
    def is_cell_equal(a:Tuple[int,int],b:Tuple[int,int]):
        return (a[0] == b[0]) and (a[1] == b[1])
    
    @property
    def rows(self) -> int: return self._env.rows
    
    @property
    def cols(self) -> int: return self._env.cols
    
    @property
    def tag(self) -> str: return self._env.tag
    
    @property
    def start(self) -> Tuple[int,int]: return self._start
    
    @property
    def goal(self) -> Tuple[int,int]: return self._goal
    
    @property
    def env(self) -> MazeEnv: return self._env
    
    def in_bounds(self,cell:Tuple[int,int]) -> bool:
        return 0 <= cell[0] < self.rows and 0 <= cell[1] < self.cols
    
    def is_wall(self,cell:Tuple[int,int]) -> bool:
        return self._env.getStateGridCell(cell) is GridCell.WALL
    
    def recompute(self):
        w = pr.get_screen_width()  - 2 * HUD.PAD
        h = pr.get_screen_height() - HUD.HEIGHT - 2 * HUD.PAD
        self.cell = max(4, int(min(w / self._env.cols, h / self._env.rows)))
        gw = self.cell * self._env.cols
        gh = self.cell * self._env.rows
        self.ox = HUD.PAD + (w - gw) // 2
        self.oy = HUD.PAD + (h - gh) // 2

    def cell_rect(self, r: int, c: int) -> Tuple[int, int, int, int]:
        return self.ox + c * self.cell, self.oy + r * self.cell, self.cell, self.cell

    def center(self, r: int, c: int) -> Tuple[int, int]:
        return (self.ox + c * self.cell + self.cell // 2,
                self.oy + r * self.cell + self.cell // 2)

    def center_f(self, r: int, c: int) -> Tuple[float, float]:
        """Centro em float, para interpolar o kart entre duas celulas."""
        return (self.ox + (c + 0.5) * self.cell,
                self.oy + (r + 0.5) * self.cell)

    def mouse_cell(self) -> Optional[Tuple[int, int]]:
        m = pr.get_mouse_position()
        c = int((m.x - self.ox) // self.cell)
        r = int((m.y - self.oy) // self.cell)
        return (r, c) if self.in_bounds((r, c)) else None
    
    def render(self):
        for r in range(self._env.rows):
            for c in range(self._env.cols):
                x, y, w, h = self.cell_rect(r, c)
                wall = (self._env.getStateGridCell((r, c)) is GridCell.WALL)
                rect_color = HUDColors.wall_cell if wall else HUDColors.open_cell
                
                if self.is_cell_equal(self._start,(r,c)):
                    rect_color = HUDColors.start_cell
                if self.is_cell_equal(self._goal,(r,c)):
                    rect_color = HUDColors.goal_cell
                    
                pr.draw_rectangle(x, y, w, h, rect_color)
                
                if not wall and self.cell >= 10:
                    pr.draw_rectangle_lines(x, y, w, h, HUDColors.outline_cell)

    def render_cells(self,
                     cells:Iterable[Tuple[int,int]],
                     color:pr.Color,
                     pad_ratio:float = 0.0):
        """Overlay de celulas (caminho otimo, rastro do kart, ...)."""
        pad = int(self.cell * pad_ratio)
        for (r, c) in cells:
            x, y, w, h = self.cell_rect(r, c)
            pr.draw_rectangle(x + pad, y + pad, w - 2 * pad, h - 2 * pad, color)

class HUDView:
    """Painel inferior. Quem chama passa as linhas ja formatadas, uma tupla
    (texto, tamanho, cor) por linha, de cima para baixo."""
    
    def render(self, lines:Sequence[Tuple[str,int,pr.Color]]):
        y0 = pr.get_screen_height() - HUD.HEIGHT
        pr.draw_rectangle(0, y0, pr.get_screen_width(), HUD.HEIGHT, HUDColors.panel)
        pr.draw_line(0, y0, pr.get_screen_width(), y0, HUDColors.panel_line)
        for (text, size, color), dy in zip(lines, HUD.LINES):
            pr.draw_text(text, HUD.PAD, y0 + dy, size, color)
