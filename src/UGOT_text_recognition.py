from constants import *
from ugot import ugot
import time
import numpy as np
import cv2

# Initialize UGOT
got = ugot.UGOT()
got.initialize(IP_ADDRESS)

got.load_models(["word_recognition"])

def text_rec(got=got):
    text = got.get_words_result()

    return text
    

if __name__ == "__main__":
    got.open_camera()
    try:
        while True:
            text = text_rec()
            frame = got.read_camera_data()
            if frame is not None:
                nparr = np.frombuffer(frame, np.uint8)
                data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                cv2.putText(data, text, (50, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                cv2.imshow("UGOT Camera", data)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Display color on UGOT screen
            text = text.lower().replace(" ", "")
            if "white" in text:
                got.screen_display_background(1)
            elif "purple" in text:
                got.screen_display_background(2)
            elif "red" in text:
                got.screen_display_background(3)
            elif "orange" in text:
                got.screen_display_background(4)
            elif "yellow" in text:
                got.screen_display_background(5)
            elif "green" in text:
                got.screen_display_background(6)
            elif "cyan" in text:
                got.screen_display_background(7)
            elif "blue" in text:
                got.screen_display_background(8)
            else:
                got.screen_display_background(0)
            
    finally:
        cv2.destroyAllWindows()