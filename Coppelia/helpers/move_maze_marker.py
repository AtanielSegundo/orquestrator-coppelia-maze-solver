from Coppelia.helpers.cell_to_world import cell_to_world


def move_maze_marker(sim, tag: str, name: str, origin, cell_cube_size: float,
                     cell, thin: float = 0.01, eps: float = 1e-3) -> None:
    """Reposition the `{tag}_start` / `{tag}_goal` marker box render_maze()
    drew (see its own start/goal loop) onto `cell`. render_maze() only draws
    them once, from the .maze file's baked-in start/goal, so this is what
    keeps the Coppelia scene in sync whenever the operator picks a different
    start/goal at runtime (mouse click or the `start`/`goal` prompt
    commands) -- without it those boxes go stale, still marking the maze
    file's original cells."""
    try:
        obj = sim.getObject(f"/{tag}_{name}")
    except Exception as e:
        print(f"[ERROR] marker '{tag}_{name}' not found: {e}")
        return
    x, y, oz = cell_to_world(origin, cell_cube_size, cell)
    sim.setObjectPosition(obj, sim.handle_world, [x, y, oz + thin / 2 + eps])
