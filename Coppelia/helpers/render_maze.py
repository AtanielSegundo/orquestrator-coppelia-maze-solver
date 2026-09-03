from typing import Tuple

import numpy as np

from saecollab.env.MazeEnv import Grid, GridCell

from Coppelia.helpers.get_maze_tag import get_maze_tag
from Coppelia.helpers.maze_origin import maze_origin
from Coppelia.helpers.cell_to_world import cell_to_world


def render_maze(sim,
                maze_path: str,
                center: Tuple[float, float, float],
                cell_cube_size=0.3,                 # em metros
                tag=None,
                wall_color=(1.0, 0.0, 0.0),
                start_color=(0.0, 0.0, 1.0),        # azul
                goal_color=(0.0, 1.0, 0.0)):        # verde
    """Returns (container, origin) so callers can also place objects on
    specific cells with cell_to_world(origin, cell_cube_size, cell) -- or
    (None, None) if the maze file failed to load."""
    grid = Grid(maze_path)
    if not grid.file_loaded:
        print(f"[ERROR] Maze not loaded: {maze_path}")
        return None, None

    tag = tag or get_maze_tag(maze_path)

    origin = maze_origin(grid.rows, grid.cols, center, cell_cube_size)
    ox, oy, oz = origin

    container = sim.createDummy(0)
    sim.setObjectAlias(container, tag)
    sim.setObjectPosition(container, sim.handle_world, [ox, oy, oz])

    def add_box(size, alias, world_pos, color, respondable):
        h = sim.createPrimitiveShape(sim.primitiveshape_cuboid, size)
        sim.setObjectAlias(h, alias)
        sim.setObjectParent(h, container, True)
        sim.setObjectPosition(h, sim.handle_world, world_pos)
        sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, list(color))
        sim.setObjectInt32Param(h, sim.shapeintparam_static, 1)
        sim.setObjectInt32Param(h, sim.shapeintparam_respondable, respondable)
        return h

    def cell_xy(r, c):
        x, y, _ = cell_to_world(origin, cell_cube_size, (r, c))
        return x, y

    for r in range(grid.rows):
        for c in range(grid.cols):
            if grid.grid[r, c] != GridCell.WALL.value:
                continue
            x, y = cell_xy(r, c)
            add_box([cell_cube_size] * 3,
                    f"{tag}_{r}_{c}",
                    [x, y, oz + cell_cube_size / 2],
                    wall_color, respondable=1)

    eps = 1e-3

    thin = 0.01
    for value, color, name in ((GridCell.START.value, start_color, "start"),
                               (GridCell.GOAL.value,  goal_color,  "goal")):
        coords = np.where(grid.grid == value)
        if coords[0].size == 0:
            continue
        r, c = int(coords[0][0]), int(coords[1][0])
        x, y = cell_xy(r, c)
        add_box([cell_cube_size, cell_cube_size, thin],
                f"{tag}_{name}",
                [x, y, oz + thin / 2 + eps],
                color, respondable=0)

    return container, origin
