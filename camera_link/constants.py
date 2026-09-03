# Loopback control endpoint of the in-scene feeder (mirrors sim_video_feeder.py).
# Stay off CoppeliaSim's own ports: 19997 (legacy remote API), 23000 (ZMQ remote
# API) and 23050 (WS remote API server -- talking raw JSON at that one just
# floods the log with `simWS: handle_read_handshake error`). Also below 49152,
# so Windows cannot hand the port out as an ephemeral one behind our back.
FEEDER_HOST = "127.0.0.1"
FEEDER_PORT = 28080
