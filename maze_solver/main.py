import os
import sys
import queue
import threading
import argparse
import multiprocessing

from typing import *

_HERE = os.path.dirname(os.path.abspath(__file__))   # maze_solver
_ROOT = os.path.dirname(_HERE)                       # orquestrator
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pyray as pr
import src.styles as styles
import src.definitions as definitions

from src.definitions import (
    SIM,
    PATHS,
    WINDOW
)
from src.renders import (
    HUDView,
    KartView,
    MazeView
)

from src.solver import (
    solve_bfs,
    heading_between,
    resolve_maze,
)

# ----/----/----/----/----

# 28 cm is the kart bigger side

CONTROLS = """
CONTROLES DA JANELA
  botao esquerdo   posiciona o START
  botao direito    posiciona o GOAL
  SPACE            play / pause
  N                um passo (funciona pausado)
  R                volta o kart ao start
  C                caminho da solucao por cima
  T                rastro do kart
  LEFT / RIGHT     mais lento / mais rapido
  ESC              sair
"""

HELP = """
COMANDOS DO PROMPT
  maze <arquivo>     troca o labirinto (aceita so o nome, ex: big_eg)
  start <lin> <col>  posiciona o start
  goal  <lin> <col>  posiciona o goal
  play | pause       roda / pausa o kart
  step [n]           avanca n passos (default 1)
  reset              volta o kart ao start
  speed <0..5>       0 = mais rapido, 5 = mais lento
  cell <metros>      lado real da celula, redimensiona o kart
  trail | path       liga/desliga rastro / caminho
  status             imprime o estado atual
  help | quit
"""

# nome -> (min_args, max_args)
COMMANDS = {
    "maze":   (1, 1),
    "start":  (2, 2),
    "goal":   (2, 2),
    "play":   (0, 0),
    "pause":  (0, 0),
    "step":   (0, 1),
    "reset":  (0, 0),
    "speed":  (1, 1),
    "cell":   (1, 1),
    "trail":  (0, 0),
    "path":   (0, 0),
    "status": (0, 0),
    "quit":   (0, 0),
}

# ----/----/----/----/----
# Solver
# ----/----/----/----/----
# solve_bfs / heading_between / DIRECTIONS now live in src.solver, shared with
# the CoppeliaSim + real-kart runner (coppelia_main.py).

class Episode:
    """Estado do percurso. Quem chama so faz set_start / set_goal, step e
    advance; o caminho e recalculado sozinho a cada mudanca."""

    def __init__(self,view:MazeView):
        self._view = view
        self.start = view.start
        self.goal  = view.goal
        self.reset()

    def reset(self):
        self.path    = solve_bfs(self._view, self.start, self.goal)
        self.index   = 0
        self.frac    = 0.0     # posicao dentro do trecho atual, em [0,1)
        self.heading = 0.0
        self._view.set_start_goal(self.start, self.goal)

    def rebind(self):
        """Chamado quando o maze muda: pega o start/goal do arquivo novo."""
        self.start = self._view.start
        self.goal  = self._view.goal
        self.reset()

    def set_start(self,cell:Tuple[int,int]) -> bool:
        if self._view.is_wall(cell) or cell == self.goal:
            return False
        self.start = cell
        self.reset()
        return True

    def set_goal(self,cell:Tuple[int,int]) -> bool:
        if self._view.is_wall(cell) or cell == self.start:
            return False
        self.goal = cell
        self.reset()
        return True

    @property
    def solvable(self) -> bool:
        return len(self.path) > 0

    @property
    def optimal(self) -> int:
        return max(0, len(self.path) - 1)

    @property
    def finished(self) -> bool:
        return (not self.solvable) or self.index >= self.optimal

    @property
    def cell(self) -> Tuple[int,int]:
        return self.path[self.index] if self.solvable else self.start

    @property
    def trail(self) -> List[Tuple[int,int]]:
        return self.path[:self.index + 1] if self.solvable else []

    def step(self,count:int = 1):
        for _ in range(max(1, count)):
            if self.finished:
                break
            self.index += 1
        self.frac = 0.0

    def advance(self,delta_time:float,step_time:float):
        """Anda no tempo, interpolando entre celulas para o kart deslizar."""
        if self.finished or step_time <= 0.0:
            self.frac = 0.0
            return
        self.frac += delta_time / step_time
        while self.frac >= 1.0 and not self.finished:
            self.frac -= 1.0
            self.index += 1
        if self.finished:
            self.frac = 0.0

    def position(self) -> Tuple[Tuple[float,float],float]:
        """Pixel do kart e o seu heading, ja interpolados."""
        if not self.solvable:
            return self._view.center_f(*self.start), self.heading

        here = self.path[self.index]
        nxt  = self.path[min(self.index + 1, self.optimal)]

        angle = heading_between(here, nxt)
        if angle is not None:
            self.heading = angle

        hx, hy = self._view.center_f(*here)
        nx, ny = self._view.center_f(*nxt)
        return (hx + (nx - hx) * self.frac,
                hy + (ny - hy) * self.frac), self.heading

# ----/----/----/----/----
# Orquestrador
# ----/----/----/----/----

class SolverOrquestrator:
    def __init__(self,k:KartView,m:MazeView,
                 kart_lenght_in_meters = 0.28,
                 maze_cell_lenght_in_meters = 0.3
                 ):
        self._kart_view = k
        self._maze_view = m
        self._kart_lenght = kart_lenght_in_meters
        self._hud_view  = HUDView()
        self._episode   = Episode(m)

        self.playing = False
        self.speed   = SIM.DEFAULT_SPEED
        self.show    = {"trail": True, "path": True}
        self.running = True

        self.set_cell_lenght(maze_cell_lenght_in_meters)

    # ---- escala fisica ----

    @property
    def kart_in_cells(self) -> float:
        """Quantas celulas o kart ocupa no maior lado."""
        return self._kart_lenght / self._cell_lenght

    def set_cell_lenght(self,lenght_meters:float):
        if lenght_meters <= 0.0:
            print("[ERROR] o lado da celula tem que ser > 0")
            return
        self._cell_lenght = lenght_meters
        self.recompute()

    def recompute(self):
        """Relayout do grid e do kart. Vale para resize, troca de maze e
        troca da escala fisica."""
        self._maze_view.recompute()
        self._kart_view.set_scale_to_pixels(self._maze_view.cell * self.kart_in_cells)

    def set_maze(self,maze_path:str) -> bool:
        if not self._maze_view.set_maze(maze_path):
            return False
        self._episode.rebind()
        self.playing = False
        self.recompute()
        return True

    # ---- comandos vindos do prompt ----

    def handle_command(self,name:str,args:List[str]):
        handler = getattr(self, f"_cmd_{name}", None)
        if handler is None:
            print(f"[ERROR] comando desconhecido: {name}")
            return
        try:
            handler(args)
        except ValueError:
            print(f"[ERROR] argumentos invalidos para '{name}'")

    def _cmd_quit(self,args):  self.running = False
    def _cmd_play(self,args):  self._set_playing(True)
    def _cmd_pause(self,args): self._set_playing(False)
    def _cmd_trail(self,args): self._toggle("trail")
    def _cmd_path(self,args):  self._toggle("path")

    def _cmd_reset(self,args):
        self._episode.reset()
        self.playing = False

    def _cmd_step(self,args):
        self.playing = False
        self._episode.step(int(args[0]) if args else 1)

    def _cmd_speed(self,args):
        i = int(args[0])
        if not 0 <= i < len(SIM.SPEEDS):
            print(f"[ERROR] speed fora de 0..{len(SIM.SPEEDS) - 1}")
            return
        self.speed = i
        print(f"[INFO] speed {i} ({SIM.SPEEDS[i] * 1000:.0f}ms por passo)")

    def _cmd_cell(self,args):
        self.set_cell_lenght(float(args[0]))
        print(f"[INFO] celula {self._cell_lenght:.3f}m, "
              f"kart ocupa {self.kart_in_cells:.2f} celula(s)")

    def _cmd_start(self,args):
        self._place("start", (int(args[0]), int(args[1])))

    def _cmd_goal(self,args):
        self._place("goal", (int(args[0]), int(args[1])))

    def _cmd_maze(self,args):
        path = resolve_maze(args[0])
        if path is None:
            print(f"[ERROR] maze nao encontrado: {args[0]}")
            return
        if self.set_maze(path):
            print(f"[INFO] maze {self._maze_view.tag} "
                  f"{self._maze_view.rows}x{self._maze_view.cols}")

    def _cmd_status(self,args):
        ep = self._episode
        print(f"[INFO] {self._maze_view.tag} "
              f"{self._maze_view.rows}x{self._maze_view.cols}"
              f"  start {ep.start}  goal {ep.goal}"
              f"  passos {ep.index}/{ep.optimal}"
              f"  {'rodando' if self.playing else 'pausado'}"
              f"{'  SEM CAMINHO' if not ep.solvable else ''}")

    def _toggle(self,flag:str):
        self.show[flag] = not self.show[flag]

    def _set_playing(self,value:bool):
        if value and self._episode.finished:
            self._episode.reset()
        self.playing = value

    def _place(self,which:str,cell:Tuple[int,int]):
        if not self._maze_view.in_bounds(cell):
            print(f"[ERROR] {cell} fora do grid "
                  f"{self._maze_view.rows}x{self._maze_view.cols}")
            return
        ok = (self._episode.set_start(cell) if which == "start"
              else self._episode.set_goal(cell))
        if not ok:
            print(f"[ERROR] {cell} e parede ou ja e o outro extremo")
            return
        self.playing = False
        if not self._episode.solvable:
            print(f"[WARN] sem caminho entre "
                  f"{self._episode.start} e {self._episode.goal}")

    # ---- input da janela ----

    def handle_input(self):
        if pr.is_window_resized():
            self.recompute()

        if pr.is_mouse_button_pressed(pr.MOUSE_BUTTON_LEFT):
            cell = self._maze_view.mouse_cell()
            if cell:
                self._place("start", cell)
        if pr.is_mouse_button_pressed(pr.MOUSE_BUTTON_RIGHT):
            cell = self._maze_view.mouse_cell()
            if cell:
                self._place("goal", cell)

        if pr.is_key_pressed(pr.KEY_SPACE): self._set_playing(not self.playing)
        if pr.is_key_pressed(pr.KEY_N):     self._cmd_step([])
        if pr.is_key_pressed(pr.KEY_R):     self._cmd_reset([])
        if pr.is_key_pressed(pr.KEY_T):     self._toggle("trail")
        if pr.is_key_pressed(pr.KEY_C):     self._toggle("path")
        if pr.is_key_pressed(pr.KEY_RIGHT): self.speed = max(0, self.speed - 1)
        if pr.is_key_pressed(pr.KEY_LEFT):  self.speed = min(len(SIM.SPEEDS) - 1,
                                                             self.speed + 1)

    # ---- update ----

    def update(self):
        if self.playing and not self._episode.finished:
            self._episode.advance(pr.get_frame_time(), SIM.SPEEDS[self.speed])
        if self._episode.finished:
            self.playing = False

    # ---- draw ----

    def hud_lines(self) -> Sequence[Tuple[str,int,pr.Color]]:
        ep = self._episode

        if not ep.solvable:
            status, color = (f"SEM CAMINHO entre {ep.start} e {ep.goal}",
                             styles.HUDColors.text_fail)
        elif ep.finished:
            status, color = (f"GOAL em {ep.index} passos",
                             styles.HUDColors.text_ok)
        elif self.playing:
            status, color = "rodando...", styles.HUDColors.text
        else:
            status, color = ("pausado  -  SPACE para dar play",
                             styles.HUDColors.text_dim)

        second = (f"passos {ep.index}/{ep.optimal}"
                  f"    celula {ep.cell}"
                  f"    vel {SIM.SPEEDS[self.speed] * 1000:.0f}ms")

        third = (f"{self._maze_view.tag}  "
                 f"{self._maze_view.rows}x{self._maze_view.cols}"
                 f"  |  start {ep.start}  goal {ep.goal}"
                 f"  |  celula {self._cell_lenght:.2f}m"
                 f"  |  kart {self._kart_lenght:.2f}m "
                 f"({self.kart_in_cells:.2f} cel)")

        on = lambda flag: "on" if self.show[flag] else "off"
        fourth = (f"L-click start   R-click goal   SPACE play   N passo   "
                  f"R reset   C caminho[{on('path')}]   T rastro[{on('trail')}]"
                  f"   <- -> vel")

        return (
            (status, 22, color),
            (second, 18, styles.HUDColors.text_dim),
            (third,  16, styles.HUDColors.text_dim),
            (fourth, 16, styles.HUDColors.panel_line),
        )

    def render(self):
        pr.begin_drawing()
        pr.clear_background(styles.HUDColors.background)

        self._maze_view.render()

        if self.show["path"] and self._episode.solvable:
            self._maze_view.render_cells(self._episode.path,
                                         styles.HUDColors.path_cell)
        if self.show["trail"]:
            self._maze_view.render_cells(self._episode.trail,
                                         styles.HUDColors.trail_cell, 0.25)

        (x, y), angle = self._episode.position()
        self._kart_view.set_offset(x, y)
        self._kart_view.set_rotation(angle)
        self._kart_view.render()

        self._hud_view.render(self.hud_lines())

        pr.end_drawing()

    # ---- loop ----

    def drain(self,commands:Optional[Any]):
        """Consome tudo que o prompt mandou desde o frame anterior."""
        if commands is None:
            return
        while True:
            try:
                name, args = commands.get_nowait()
            except queue.Empty:
                return
            self.handle_command(name, args)

    def run(self,commands:Optional[Any] = None):
        while self.running and not pr.window_should_close():
            self.drain(commands)
            self.handle_input()
            self.update()
            self.render()

# ----/----/----/----/----
# Processos
# ----/----/----/----/----

def render_solver(maze_path:str,
                  cell_lenght:float,
                  kart_lenght:float,
                  commands:Optional[Any] = None):
    pr.set_config_flags(pr.ConfigFlags.FLAG_WINDOW_RESIZABLE)
    pr.set_trace_log_level(definitions.LOG.LEVEL)

    pr.init_window(WINDOW.WIDTH,
                   WINDOW.HEIGHT,
                   WINDOW.TITLE)
    pr.set_target_fps(60)

    try:
        kart   = KartView(styles.Kart.file_path, styles.Kart.tint)
        maze_v = MazeView(maze_path)

        orquestrator = SolverOrquestrator(kart, maze_v,
                                          kart_lenght_in_meters=kart_lenght,
                                          maze_cell_lenght_in_meters=cell_lenght)
        orquestrator.run(commands)
    finally:
        pr.close_window()

def cli_requests(commands:Any):
    """Roda no processo PRINCIPAL de proposito: no Windows um Process filho
    nasce com o stdin apontando para os.devnull, entao input() la dentro so
    devolve EOF. O filho fica com a janela, que nao precisa de stdin."""
    print(HELP)
    while True:
        try:
            raw = input("[maze] ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        parts = raw.split()
        name, args = parts[0].lower(), parts[1:]

        if name in ("help", "ajuda", "h", "?"):
            print(HELP)
            print(CONTROLS)
            continue
        if name in ("sair", "exit", "q"):
            name = "quit"
        if name not in COMMANDS:
            print(f"[ERROR] comando desconhecido: {name}  (help para a lista)")
            continue

        low, high = COMMANDS[name]
        if not low <= len(args) <= high:
            print(f"[ERROR] '{name}' espera de {low} a {high} argumento(s)")
            continue

        commands.put((name, args))
        if name == "quit":
            break

def watch_render(process:multiprocessing.Process):
    """A janela pode fechar sozinha (ESC / X) enquanto o prompt esta preso em
    input(). Sem isso o processo principal ficaria orfao esperando uma linha."""
    process.join()
    print("\n[INFO] janela fechada, encerrando.")
    sys.stdout.flush()
    os._exit(0)

# ----/----/----/----/----
# CLI
# ----/----/----/----/----
# resolve_maze now lives in src.solver.

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Orquestrador do maze: o kart percorre a solucao, "
                    "controlado pela janela ou pelo prompt.",
        epilog=CONTROLS + HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-m", "--maze", default=SIM.DEFAULT_MAZE,
                   help="arquivo .maze (ou so o nome, ex: big_eg)")
    p.add_argument("--cell-lenght", type=float, default=0.3,
                   help="lado real da celula em metros (default: %(default)s)")
    p.add_argument("--kart-lenght", type=float, default=0.28,
                   help="maior lado do kart em metros (default: %(default)s)")
    p.add_argument("--no-cli", action="store_true",
                   help="so a janela, sem o prompt nem o processo extra")
    return p

if __name__ == "__main__":
    args = build_parser().parse_args()

    maze_path = resolve_maze(args.maze)
    
    if maze_path is None:
        print(f"[ERROR] maze nao encontrado: {args.maze}")
        print(f"        procurei tambem em {PATHS.MAZES}")
        sys.exit(-1)

    if args.no_cli:
        render_solver(maze_path, args.cell_lenght, args.kart_lenght)
        sys.exit(0)

    context  = multiprocessing.get_context("spawn")
    commands = context.Queue()

    p_rendering = context.Process(
        target=render_solver,
        args=(maze_path, args.cell_lenght, args.kart_lenght, commands),
        daemon=True,
    )
    p_rendering.start()

    threading.Thread(target=watch_render,
                     args=(p_rendering,),
                     daemon=True).start()

    try:
        cli_requests(commands)
    
    except Exception as e:
        print(f"[ERROR] cli requests closed: {e}")
    
    finally:
        p_rendering.join(timeout=2.0)
        if p_rendering.is_alive():
            p_rendering.terminate()
            p_rendering.join()
