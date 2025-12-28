import cv2
import mediapipe as mp
import numpy as np
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ---------------------------------------------
# Gesture classifier (same as yours, reused)
# ---------------------------------------------
def classify_gesture(hand_landmarks):
    lm = hand_landmarks.landmark

    THUMB_TIP = 4
    THUMB_IP = 3

    INDEX_TIP = 8
    INDEX_PIP = 6
    MIDDLE_TIP = 12
    MIDDLE_PIP = 10
    RING_TIP = 16
    RING_PIP = 14
    PINKY_TIP = 20
    PINKY_PIP = 18

    def finger_up(tip, pip):
        return lm[tip].y < lm[pip].y

    thumb_up  = lm[THUMB_TIP].x < lm[THUMB_IP].x
    index_up  = finger_up(INDEX_TIP, INDEX_PIP)
    middle_up = finger_up(MIDDLE_TIP, MIDDLE_PIP)
    ring_up   = finger_up(RING_TIP, RING_PIP)
    pinky_up  = finger_up(PINKY_TIP, PINKY_PIP)

    pattern = [thumb_up, index_up, middle_up, ring_up, pinky_up]
    pattern = [1 if f else 0 for f in pattern]

    # Some simple patterns
    if pattern == [0,0,0,0,0]: return "Fist"
    if pattern[1:] == [1,1,1,1]: return "Open Palm"
    if pattern == [1,0,0,0,0]: return "Thumbs Up"
    if pattern == [0,1,0,0,0]: return "Pointing"

    return f"Pattern {pattern}"


# ---------------------------------------------
# Main: Air drawing
# ---------------------------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    # Drawing state
    draw_color_list = [
        (0,   0, 255),  # Red
        (0, 255,   0),  # Green
        (255, 0,   0),  # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 255) # White
    ]
    color_index = 0
    current_color = draw_color_list[color_index]

    brush_thickness = 6
    eraser_thickness = 40

    last_gesture = ""
    last_color_switch_time = 0
    color_switch_cooldown = 0.5  # seconds

    prev_x, prev_y = None, None
    mode = "draw"  # "draw" or "erase" or "idle"

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    ) as hands:

        # Canvas will be created after first frame (so it matches camera size)
        canvas = None

        print("🎨 Air-drawing started!")
        print("Gestures:")
        print("  👉 Pointing   = draw with current color")
        print("  ✋ Open Palm  = eraser mode")
        print("  👍 Thumbs Up  = switch color")
        print("  ✊ Fist       = lift pen / idle")
        print("Keyboard:")
        print("  S = save drawing as 'air_drawing.png'")
        print("  C = clear canvas")
        print("  Q = quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Camera disconnected")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Lazy init canvas
            if canvas is None:
                canvas = np.zeros_like(frame)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            gesture_text = "No hand"

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]

                # Draw skeleton on camera frame
                mp_drawing.draw_landmarks(
                    frame,
                    hand,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
                )

                # Classify gesture
                gesture_text = classify_gesture(hand)

                # -------------------------------
                # GESTURE → MODE changes
                # -------------------------------
                now = time.time()

                if gesture_text == "Thumbs Up":
                    # Color switching with cooldown, so it doesn't spam every frame
                    if now - last_color_switch_time > color_switch_cooldown:
                        color_index = (color_index + 1) % len(draw_color_list)
                        current_color = draw_color_list[color_index]
                        last_color_switch_time = now
                        print(f"🎨 Color changed to index {color_index}, BGR={current_color}")
                    mode = "idle"
                    prev_x, prev_y = None, None

                elif gesture_text == "Open Palm":
                    # Eraser mode
                    mode = "erase"

                elif gesture_text == "Pointing":
                    # Drawing mode
                    mode = "draw"

                elif gesture_text == "Fist":
                    # Lift pen / idle
                    mode = "idle"
                    prev_x, prev_y = None, None

                # -------------------------------
                # Get index fingertip position
                # -------------------------------
                INDEX_TIP = 8
                index_tip = hand.landmark[INDEX_TIP]
                cx = int(index_tip.x * w)
                cy = int(index_tip.y * h)

                # Visual marker for fingertip
                cv2.circle(frame, (cx, cy), 8, (255, 255, 255), -1)

                # -------------------------------
                # Drawing on canvas
                # -------------------------------
                if mode in ("draw", "erase"):
                    if prev_x is None or prev_y is None:
                        prev_x, prev_y = cx, cy

                    if mode == "draw":
                        cv2.line(canvas, (prev_x, prev_y), (cx, cy),
                                 current_color, brush_thickness)
                    elif mode == "erase":
                        cv2.line(canvas, (prev_x, prev_y), (cx, cy),
                                 (0, 0, 0), eraser_thickness)

                    prev_x, prev_y = cx, cy
                else:
                    # Not drawing now
                    prev_x, prev_y = None, None
            else:
                # No hand
                mode = "idle"
                prev_x, prev_y = None, None

            # ---------------------------------
            # Combine canvas and camera frame
            # ---------------------------------
            # Any non-black pixel from canvas will appear on top
            gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
            output = cv2.add(frame_bg, canvas_fg)

            # Show current gesture and mode + color indicator
            cv2.putText(output, f"{gesture_text} | Mode: {mode}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            cv2.rectangle(output, (10, 40), (60, 90), current_color, -1)
            cv2.putText(output, "Color", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            cv2.imshow("Air Drawing", output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                canvas = np.zeros_like(canvas)
                print("🧹 Canvas cleared")
            elif key == ord('s'):
                cv2.imwrite("air_drawing.png", canvas)
                print("💾 Saved drawing as air_drawing.png")

    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed.")


if __name__ == "__main__":
    main()
