def local_bounding_box(sim, handle):
    """Combined 2D (X, Y) bounding box of `handle` and every descendant in its
    tree, expressed in HANDLE's OWN local frame. Returns (min_x, max_x,
    min_y, max_y); all zero if nothing in the tree has usable geometry.

    Needed whenever `handle` itself is a dummy/reference frame with no shape
    of its own (its own objbbox_* params come back degenerate/zero) but its
    visual mesh lives on a child shape -- e.g. a robot's "_base" dummy, kept
    separate from the chassis mesh so the base's origin can sit wherever the
    kinematics need it (see kart_base). Querying only `handle`'s own bbox in
    that case silently finds nothing, which is exactly what made
    set_planar_pose()/SimMirror apply no correction at all and place the
    model by its raw (off-center) origin.

    Each descendant's OWN bbox is local to THAT object, so each of its 4
    corners is transformed into `handle`'s frame via that object's matrix
    relative to `handle` (sim.getObjectMatrix(obj, handle)) before folding
    into the running min/max. Objects with no bbox (not a shape) are skipped.
    """
    try:
        tree = list(sim.getObjectsInTree(handle, sim.handle_all, 0))
    except Exception:
        tree = []
    if handle not in tree:
        tree.append(handle)

    xs, ys = [], []
    for obj in tree:
        try:
            bb_min_x = sim.getObjectFloatParam(obj, sim.objfloatparam_objbbox_min_x)
            bb_max_x = sim.getObjectFloatParam(obj, sim.objfloatparam_objbbox_max_x)
            bb_min_y = sim.getObjectFloatParam(obj, sim.objfloatparam_objbbox_min_y)
            bb_max_y = sim.getObjectFloatParam(obj, sim.objfloatparam_objbbox_max_y)
        except Exception:
            continue                       # not a shape / no bbox -- skip

        m = sim.getObjectMatrix(obj, handle)   # obj's pose, in HANDLE's own frame
        for lx, ly in ((bb_min_x, bb_min_y), (bb_min_x, bb_max_y),
                      (bb_max_x, bb_min_y), (bb_max_x, bb_max_y)):
            xs.append(m[0] * lx + m[1] * ly + m[3])
            ys.append(m[4] * lx + m[5] * ly + m[7])

    if not xs:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs), max(xs), min(ys), max(ys))
