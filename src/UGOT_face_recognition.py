from constants import *
from ugot import ugot
import time
import numpy as np
import cv2

# Initialize UGOT
got = ugot.UGOT()
got.initialize(IP_ADDRESS)

got.load_models(["face_recognition"])

# Print all previously recognised faces
print(got.face_recognition_get_all_names())

# Add a face
got.face_recognition_add_name("your name here")

# Delete a face
# got.face_recognition_delete_name("your name here")

def recognise_face(got=got):
    faces = got.get_face_recognition_total_info()
    print(faces)
    if faces:
        return faces[0]
    else:
        return [""]
    
if __name__ == "__main__":
    got.open_camera()
    try:
        while True:
            frame = got.read_camera_data()
            if frame is not None:
                nparr = np.frombuffer(frame, np.uint8)
                data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                cv2.imshow("UGOT Camera", data)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            face_info = recognise_face()
            # If recognised a specific face, do something
            if face_info[0] == "your name here":
                got.screen_display_background(5)
            else:
                got.screen_display_background(0)
    finally:
        cv2.destroyAllWindows()