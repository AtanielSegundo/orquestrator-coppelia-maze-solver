from Coppelia.sim import CoppeliaSimulation


def remove_object(sim:CoppeliaSimulation,alias:str) -> bool:
    try:
        base = sim.getObject(f"/{alias}")
    except Exception as e:
        print(f"[ERROR] Failed to Remove Object {alias}: {e}")
        return False

    tree = sim.getObjectsInTree(base, sim.handle_all, 0)
    sim.removeObjects(tree)
