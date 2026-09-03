"""
Shared "shadow" plumbing for mirroring the REAL Lafvin PiKart into a
simulation: the WebSocket link to the kart (telemetry in, commands out), the
2D rigid-pose algebra used to place it, and the CoppeliaSim pose mirror.

Split out of control_websocket.py so both the teleop bridge (gamepad ->
`drive` twist) and the maze-solving runner (`turn` / `drive_distance` per
cell) can share one tested implementation instead of two copies drifting
apart. Each piece lives in its own module (one class/function per file); this
just re-exports them so `from shadow import KartLink, SimMirror, ...` keeps
working.

Deps:  pip install websocket-client
"""
from shadow.compose import compose
from shadow.inverse import inverse
from shadow.wrap import wrap
from shadow.grid_heading_to_world_theta import grid_heading_to_world_theta
from shadow.world_theta_to_grid_heading import world_theta_to_grid_heading
from shadow.kart_link import KartLink
from shadow.sim_mirror import SimMirror

__all__ = [
    "compose", "inverse", "wrap",
    "grid_heading_to_world_theta", "world_theta_to_grid_heading",
    "KartLink", "SimMirror",
]
