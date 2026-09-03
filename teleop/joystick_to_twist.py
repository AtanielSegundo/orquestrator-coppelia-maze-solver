def joystick_to_twist(axis_fwd, axis_turn, v_max, w_max):
    """Dominant-axis mapping -> body twist (linear m/s, angular rad/s).
       Y -> forward/back (pure translation); X -> turn in place (pure rotation).
       Mirrors joystick_to_wheels() but in twist space, since the kart's `drive`
       command takes a twist and runs its own inverse kinematics."""
    if abs(axis_turn) >= abs(axis_fwd):
        return 0.0, axis_turn * w_max           # pure spin
    return -axis_fwd * v_max, 0.0               # pure drive (stick up = forward)
