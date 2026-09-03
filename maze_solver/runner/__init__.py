"""
CoppeliaSim + real-Lafvin-kart maze runner. See runner/main.py for the full
flow description. Each class/function/constant lives in its own module; the
runnable entrypoint is maze_solver/coppelia_main.py (a thin launcher into this
package).
"""
from runner.main import main

__all__ = ["main"]
