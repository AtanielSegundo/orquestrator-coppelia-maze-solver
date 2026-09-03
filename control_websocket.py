#!/usr/bin/python3
"""
Launcher for the gamepad/keyboard teleop bridge (real Lafvin PiKart, mirrored
live into CoppeliaSim). Implementation lives in teleop/, split one
class/function/constant group per file; this stays a thin, directly-runnable
entrypoint so `python control_websocket.py` keeps working.
"""
from teleop.main import main

if __name__ == "__main__":
    main()
