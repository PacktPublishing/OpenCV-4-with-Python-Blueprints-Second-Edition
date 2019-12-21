import numpy as np
import cv2
import tensorflow.keras as K

def draw_box(frame: np.ndarray, box: np.ndarray) -> np.ndarray:
    h, w = frame.shape[0:2]
    pts = (box.reshape((2, 2)) * np.array([w, h])).astype(np.int)
    cv2.rectangle(frame, tuple(pts[0]), tuple(pts[1]), (0, 255, 0), 2)
    return frame

model = K.models.load_model("localization.h5")

cap = cv2.VideoCapture(0)

for _, frame in iter(cap.read, (False, None)):
    input = cv2.resize(frame, (224, 224))
    input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    box, = model.predict(input[None] / 255)
    draw_box(frame, box)
    cv2.imshow("res", frame)
    if(cv2.waitKey(1) == 27):
        break
