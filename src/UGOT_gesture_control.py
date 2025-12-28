"""Gesture Control for UGOT Robot

This module uses MediaPipe to detect hand gestures from a webcam to control the UGOT.
- RIGHT hand controls DIRECTION (forward/backward/left/right)
- LEFT hand controls SPEED (0-4 fingers = 0-40 speed)
"""
from constants import *
import cv2
import mediapipe as mp
from ugot import ugot
import time
import numpy as np

# ============================================================================
# SETUP
# ============================================================================
got = ugot.UGOT()
got.initialize(IP_ADDRESS)  # Change IP based on robot
got.open_camera()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Hand landmark indices (MediaPipe convention):
# Fingers: 8=index_tip, 6=index_pip, 12=middle_tip, 10=middle_pip, etc.
# Thumb: 4=thumb_tip, 3=thumb_ip

# ============================================================================
# HELPER FUNCTIONS: HAND GESTURE RECOGNITION
# ============================================================================

def count_fingers(lm):
    """
    Count how many fingers are extended on a hand.
    
    Only counts INDEX, MIDDLE, RING, PINKY (thumb is ignored for reliability).
    A finger is considered "up" if its tip is higher on screen than its PIP joint.
    
    Args:
        lm: MediaPipe hand_landmarks.landmark list (21 joints)
        
    Returns:
        int: Number of fingers up (0-4)
    """
    def is_finger_up(tip_idx, pip_idx):
        """
        Finger is up if tip.y < pip.y (smaller y = higher on screen in OpenCV).
        """
        return lm[tip_idx].y < lm[pip_idx].y

    index_up = is_finger_up(8, 6)
    middle_up = is_finger_up(12, 10)
    ring_up = is_finger_up(16, 14)
    pinky_up = is_finger_up(20, 18)

    fingers = [index_up, middle_up, ring_up, pinky_up]
    return sum(1 for f in fingers if f)


def classify_right_hand(lm):
    """
    Classify RIGHT hand gesture to determine robot direction.
    
    Gesture Mapping:
        - Fist (all fingers down, thumb not extended) → LEFT
        - 1 finger up (index only) → RIGHT
        - All 4 fingers up (index-pinky extended) → FORWARD
        - Thumb only (fist + thumb extended) → BACKWARD
        - Unrecognized → NONE
    
    Args:
        lm: MediaPipe hand_landmarks.landmark list
        
    Returns:
        str: Direction command ("Left", "Right", "Forward", "Backward", or "None")
    """
    def is_finger_up(tip_idx, pip_idx):
        """Helper: check if finger is extended."""
        return lm[tip_idx].y < lm[pip_idx].y

    # Thumb extended if thumb_tip.x < thumb_ip.x (for right hand)
    thumb_up = lm[4].x < lm[3].x

    # Check index-pinky fingers
    index_up = is_finger_up(8, 6)
    middle_up = is_finger_up(12, 10)
    ring_up = is_finger_up(16, 14)
    pinky_up = is_finger_up(20, 18)

    # Convert to binary pattern [0/1, 0/1, 0/1, 0/1]
    four_fingers = [index_up, middle_up, ring_up, pinky_up]
    four_bits = [1 if f else 0 for f in four_fingers]

    # === GESTURE PATTERNS ===
    
    # Fist: no index-pinky fingers up
    if four_bits == [0, 0, 0, 0]:
        return "Backward" if thumb_up else "Left"

    # Index only up → RIGHT
    if four_bits == [1, 0, 0, 0]:
        return "Right"

    # All 4 fingers (index-pinky) up → FORWARD
    if four_bits == [1, 1, 1, 1]:
        return "Forward"

    # Unrecognized pattern
    return "None"


def speed_from_fingers(fingers):
    """
    Map number of extended fingers to robot speed.
    
    Mapping:
        0 fingers (fist) → 0 (stop)
        1 finger → 10
        2 fingers → 20
        3 fingers → 30
        4 fingers → 40
    
    Args:
        fingers (int): Number of extended fingers (0-4)
        
    Returns:
        int: Speed value (0-50)
    """
    speed_table = {
        0: 0,
        1: 10,
        2: 20,
        3: 30,
        4: 40,
        5: 50,  # Safety fallback (should not occur)
    }
    return speed_table.get(fingers, 0)


# ============================================================================
# ROBOT CONTROL
# ============================================================================

def control_robot_SB(direction, speed, got):
    """
    Send movement commands to the UGOT robot based on direction and speed.
    
    Args:
        direction (str): Direction command ("Forward", "Backward", "Left", "Right", "None")
        speed (int): Speed value (0-50). Speed of 0 stops the robot.
    """
    # Stop if speed is 0 or direction is unrecognized
    if speed == 0 or direction == "None":
        got.balance_stop_balancing()
        return

    if direction == "Forward":
        got.balance_move_speed(0, speed)

    elif direction == "Backward":
        got.balance_move_speed(1, speed)

    elif direction == "Left":
        got.balance_turn_speed(2, speed * 2)

    elif direction == "Right":
        got.balance_turn_speed(3, speed * 2)

def control_robot_WL(direction, speed, got):
    """
    Send movement commands to the UGOT robot based on direction and speed.
    
    Args:
        direction (str): Direction command ("Forward", "Backward", "Left", "Right", "None")
        speed (int): Speed value (0-50). Speed of 0 stops the robot.
    """
    # Stop if speed is 0 or direction is unrecognized
    if speed == 0 or direction == "None":
        got.wheelleg_stop_balancing()
        return

    if direction == "Forward":
        got.wheelleg_move_speed(0, speed)

    elif direction == "Backward":
        got.wheelleg_move_speed(1, speed)

    elif direction == "Left":
        got.wheelleg_turn_speed(2, speed * 2)

    elif direction == "Right":
        got.wheelleg_turn_speed(3, speed * 2)

def control_robot_mec(direction, speed, got):
    """
    Send movement commands to the UGOT robot based on direction and speed.
    
    Args:
        direction (str): Direction command ("Forward", "Backward", "Left", "Right", "None")
        speed (int): Speed value (0-50). Speed of 0 stops the robot.
    """
    # Stop if speed is 0 or direction is unrecognized
    if speed == 0 or direction == "None":
        got.mecanum_stop()
        return

    if direction == "Forward":
        got.mecanum_move_speed(0, speed)

    elif direction == "Backward":
        got.mecanum_move_speed(1, speed)

    elif direction == "Left":
        got.mecanum_turn_speed(2, speed * 2)

    elif direction == "Right":
        got.mecanum_turn_speed(3, speed * 2)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def gesture_control(cap, got, robot_type = ROBOT_TYPE):
    """
    Main gesture control loop using MediaPipe hand detection.
    
    Reads from webcam, detects hand gestures, and sends commands to UGOT.
    - RIGHT hand: controls direction
    - LEFT hand: controls speed
    
    Exit: Press 'Q' to quit.
    """

    with mp_hands.Hands(
        max_num_hands=2,  # Detect both hands
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    ) as hands:

        print("🤖 TWO-HAND GESTURE CONTROL ACTIVE (Press Q to quit)")
        print("   RIGHT hand = direction (fist/1-finger/open/thumb)")
        print("   LEFT hand  = speed (0-4 fingers)")

        direction = "None"
        speed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Camera frame error")
                break

            # Flip horizontally so it acts like a mirror
            frame = cv2.flip(frame, 1)

            # MediaPipe requires RGB input
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # Reset controls each frame
            direction = "None"
            speed = 0

            # Process detected hands
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness
                ):
                    hand_label = handedness.classification[0].label  # "Left" or "Right"
                    lm = hand_landmarks.landmark

                    # Draw hand skeleton on frame
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                    # LEFT hand: map finger count to speed
                    if hand_label == "Left":
                        left_fingers = count_fingers(lm)
                        speed = speed_from_fingers(left_fingers)
                        cv2.putText(
                            frame, f"Left fingers: {left_fingers}",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 0), 2
                        )

                    # RIGHT hand: classify gesture to direction
                    elif hand_label == "Right":
                        direction = classify_right_hand(lm)
                        cv2.putText(
                            frame, f"Right: {direction}",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2
                        )

            # Execute robot movement based on detected gestures
            if robot_type == "SB":
                control_robot_SB(direction, speed, got)
            elif robot_type == "WL":
                control_robot_WL(direction, speed, got)
            elif robot_type == "mec":
                control_robot_mec(direction, speed, got)

            # Display current speed on screen
            cv2.putText(
                frame, f"Speed: {speed}",
                (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255), 2
            )

            cv2.imshow("Two-Hand UGOT Controller", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == "__main__":
    if ROBOT_TYPE == "SB":
        got.balance_start_balancing()
    elif ROBOT_TYPE == "WL":
        got.wheelleg_start_balancing()
    time.sleep(1)

    try:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Cannot open webcam")
        gesture_control(cap, got, ROBOT_TYPE)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if ROBOT_TYPE == "SB":
            got.balance_stop_balancing()
        elif ROBOT_TYPE == "WL":
            got.wheelleg_stop_balancing()
        elif ROBOT_TYPE == "mec":
            got.mecanum_stop()

        print("❌ Gesture control stopped.")