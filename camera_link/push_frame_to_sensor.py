import cv2


def push_frame_to_sensor(sim, sensor, frame, res_x, res_y, flip=True):
    """Write a decoded BGR frame into a CoppeliaSim vision sensor.

    Resizes to the sensor resolution, converts BGR->RGB (CoppeliaSim wants RGB)
    and, by default, flips vertically for the sensor's bottom-up image
    convention. options=0 -> RGB, 3 bytes/pixel; the buffer must be
    res_x*res_y*3, which the resized (res_y, res_x, 3) array satisfies exactly
    (pos/size default to the full frame).
    """
    img = cv2.resize(frame, (res_x, res_y))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if flip:
        img = cv2.flip(img, 0)
    sim.setVisionSensorImg(sensor, img.tobytes(), 0)
