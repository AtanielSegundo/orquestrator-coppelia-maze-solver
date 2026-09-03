"""
Maze solving + file resolution, shared by the Raylib-only simulator
(maze_solver/main.py) and the CoppeliaSim + real-kart runner
(maze_solver/coppelia_main.py) so both consume the exact same grid, path and
heading convention. Each piece lives in its own module; this re-exports them
so `from src.solver import solve_bfs, heading_between, resolve_maze` keeps
working.
"""
from src.solver.constants import DIRECTIONS
from src.solver.solve_bfs import solve_bfs
from src.solver.heading_between import heading_between
from src.solver.resolve_maze import resolve_maze

__all__ = ["DIRECTIONS", "solve_bfs", "heading_between", "resolve_maze"]
