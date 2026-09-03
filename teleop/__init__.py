"""
Gamepad/keyboard teleop bridge for the REAL Lafvin PiKart, mirrored live into
CoppeliaSim. See teleop/main.py for the full pipeline description. Each
constant/function/class lives in its own module; the runnable entrypoint is
control_websocket.py at the project root (a thin launcher into this package).
"""
from teleop.main import main

__all__ = ["main"]
