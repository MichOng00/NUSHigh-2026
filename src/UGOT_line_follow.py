from constants import *
from ugot import ugot
import time
import numpy as np
import cv2

# Initialize UGOT
got = ugot.UGOT()
got.initialize(IP_ADDRESS)

got.load_models(["line_recognition"])
got.set_track_recognition_line(0)

def line_follow_SB(got=got, mult=0.25, speed=10):
    offset, type, x, y = got.get_single_track_total_info()
    rotation_speed = abs(int(offset * mult))

    if offset > 0:
        direction = 2
    else:
        direction = 3
    got.balance_move_turn(0, speed, direction, rotation_speed)

    return type, x, y

def line_follow_WL(got=got, mult=0.25, speed=10):
    offset, type, x, y = got.get_single_track_total_info()
    rotation_speed = abs(int(offset * mult))

    if offset > 0:
        direction = 2
    else:
        direction = 3
    got.wheelleg_move_turn(0, speed, direction, rotation_speed)

    return type, x, y

def line_follow_mec(got=got, mult=0.25, speed=10):
    offset, type, x, y = got.get_single_track_total_info()
    rotation_speed = int(offset * mult)

    got.mecanum_move_xyz(x_speed=0, y_speed=speed, z_speed=rotation_speed)

    return type, x, y

if __name__ == "__main__":
    got.open_camera()
    if ROBOT_TYPE == "SB":
        got.balance_start_balancing()
    elif ROBOT_TYPE == "WL":
        got.wheelleg_start_balancing()
    time.sleep(1)
    try:
        while True:
            frame = got.read_camera_data()
            if frame is not None:
                nparr = np.frombuffer(frame, np.uint8)
                data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                cv2.imshow("UGOT Camera", data)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            
            if ROBOT_TYPE == "SB":
                line_follow_SB(got, 0.25, 10)
            elif ROBOT_TYPE == "WL":
                line_follow_WL(got, 0.25, 10)
            elif ROBOT_TYPE == "mec":
                line_follow_mec(got, 0.2, 10)

            
    finally:
        if ROBOT_TYPE == "SB":
            got.balance_stop_balancing()
        elif ROBOT_TYPE == "WL":
            got.wheelleg_stop_balancing()
        elif ROBOT_TYPE == "mec":
            got.mecanum_stop()
        cv2.destroyAllWindows()