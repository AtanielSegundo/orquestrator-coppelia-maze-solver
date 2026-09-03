from typing import Tuple

from Coppelia.sim import CoppeliaSimulation
from Coppelia.helpers.remove_object import remove_object


def change_floor(sim:CoppeliaSimulation,
                 dimensions: Tuple[float, float],
                 thickness: float = .5):
    """change_floor

    Args:
        sim (_type_): Coppelia Simulator
        dimensions (Tuple[float,float]): Floor Dimensions in Meters
    """
    x, y = dimensions

    remove_object(sim,"Floor")

    floor = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [x, y, thickness])
    sim.setObjectAlias(floor, "Floor")

    sim.setObjectPosition(floor, sim.handle_world, [0, 0, -thickness / 2])
    sim.setShapeColor(floor, None, sim.colorcomponent_ambient_diffuse, [0.7, 0.7, 0.7])

    sim.setObjectInt32Param(floor, sim.shapeintparam_static, 1)
    sim.setObjectInt32Param(floor, sim.shapeintparam_respondable, 1)

    return floor
