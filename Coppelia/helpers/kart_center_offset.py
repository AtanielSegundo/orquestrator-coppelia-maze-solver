# Manual fudge (meters), in the kart's OWN body frame (local +X = the
# direction kart_base is authored facing, local +Y = its left), ADDED on top
# of local_bounding_box()'s auto-computed origin-to-center offset.
#
# Tune this BY HAND if the kart still looks off-center in CoppeliaSim after
# the automatic bbox correction -- e.g. an accessory (the camera mount)
# extends the aggregated bounding box asymmetrically, skewing the computed
# center away from the body's true visual middle in just one axis. Nudge one
# number at a time and re-run: +dx shifts the effective center further along
# the kart's own forward axis, +dy further along its own left axis.
KART_CENTER_OFFSET_LOCAL = (0.03,0.168)
