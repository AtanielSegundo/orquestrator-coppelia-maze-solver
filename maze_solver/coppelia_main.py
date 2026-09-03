#!/usr/bin/python3
"""
Launcher for the CoppeliaSim + Lafvin-kart maze runner. Implementation lives
in maze_solver/runner/, split one class/function/constant group per file;
this stays a thin, directly-runnable entrypoint so `python coppelia_main.py`
keeps working from any working directory.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))   # maze_solver
_ROOT = os.path.dirname(_HERE)                        # orquestrator
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from runner.main import main

if __name__ == "__main__":
    main()
