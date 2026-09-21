from typing import Tuple

from Coppelia.helpers.cell_to_world import cell_to_world


def add_maze_obstacle(sim,
                      tag: str,
                      origin: Tuple[float, float, float],
                      cell_cube_size: float,
                      cell: Tuple[int, int],
                      color=(0.95, 0.55, 0.15),
                      eps: float = 1e-3):
    """Drop a box on `cell` for an obstacle DISCOVERED at runtime by the kart's
    front distance sensor.

    render_maze() only builds the walls the .maze file knew about, so this is
    what puts a newly-found one into the scene -- deliberately a different
    colour from the file's walls, matching the Raylib view's obstacle_cell, so
    the two kinds stay tellable apart. Aliased `{tag}_obs_{r}_{c}` and
    parented to the maze container, so remove_object(sim, tag) still takes the
    whole maze (discoveries included) down with it on the next rebuild.

    Returns the new shape's handle, or None if that cell already has one.
    """
    r, c = cell
    alias = f"{tag}_obs_{r}_{c}"
    try:
        sim.getObject(f"/{alias}")
        return None                     # already standing -- nothing to do
    except Exception:                   # noqa: BLE001
        pass

    x, y, oz = cell_to_world(origin, cell_cube_size, cell)
    h = sim.createPrimitiveShape(sim.primitiveshape_cuboid,
                                 [cell_cube_size, cell_cube_size, cell_cube_size])
    sim.setObjectAlias(h, alias)
    try:
        sim.setObjectParent(h, sim.getObject(f"/{tag}"), True)
    except Exception as e:              # noqa: BLE001
        print(f"[WARN] obstacle {alias} left unparented ({e})")
    sim.setObjectPosition(h, sim.handle_world,
                          [x, y, oz + cell_cube_size / 2 + eps])
    sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, list(color))
    sim.setObjectInt32Param(h, sim.shapeintparam_static, 1)
    sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 1)
    return h
