from constants import *
from ugot import ugot

# Initialize UGOT
got = ugot.UGOT()
got.initialize(IP_ADDRESS)

THRESHOLD = 30
TOLERANCE = 10
SPEED = 10

def constant_distance_SB(got, distance, speed):
    if distance > THRESHOLD + TOLERANCE:
        got.balance_move_speed(0, speed)
    elif distance < THRESHOLD - TOLERANCE:
        got.balance_move_speed(1, speed)
    else:
        got.balance_stop_balancing()
    
def constant_distance_WL(got, distance, speed):
    if distance > THRESHOLD + TOLERANCE:
        got.wheelleg_move_speed(0, speed)
    elif distance < THRESHOLD - TOLERANCE:
        got.wheelleg_move_speed(1, speed)
    else:
        got.wheelleg_stop_balancing()
    
def constant_distance_mec(got, distance, speed):
    if distance > THRESHOLD + TOLERANCE:
        got.mecanum_move_speed(0, speed)
    elif distance < THRESHOLD - TOLERANCE:
        got.mecanum_move_speed(1, speed)
    else:
        got.mecanum_stop()

if __name__ == "__main__":
    while True:
        distance = got.read_distance_data(51) # change based on the sensor port on main controller
        # print(f"Distance: {distance}")

        if ROBOT_TYPE == "SB":
            constant_distance_SB(got, distance, SPEED)
        elif ROBOT_TYPE == "WL":
            constant_distance_WL(got, distance, SPEED)
        elif ROBOT_TYPE == "mec":
            constant_distance_mec(got, distance, SPEED)