from teleop.rgb_to_norm import rgb_to_norm

# ---------------------------------------------------------------------------
# Kart connection
# ---------------------------------------------------------------------------
KART_HOST   = "192.168.0.238"
KART_PORT   = 8080
WS_URL      = f"ws://{KART_HOST}:{KART_PORT}/ws"
VIDEO_URL   = f"http://{KART_HOST}:{KART_PORT}/video"
COMMAND_URL = f"http://{KART_HOST}:{KART_PORT}/command"    # fire-and-forget camera servo POSTs

# Who pulls the camera stream:
#   "sim"    -> sim_video_feeder.py, loaded in the scene, pulls and writes the
#               sensor in-process; the frames never touch the remote API.
#   "client" -> this process pulls and pushes each frame over ZMQ (no scene
#               setup needed, but ~900 KB per frame through the control loop).
#   "off"    -> no video mirror at all.
VIDEO_BACKEND = "sim"

CAIXA_COR = rgb_to_norm((178, 131, 97))
